import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 全域型態與隨機種子設定 (預設 float64)
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 物理問題定義 (Problem Definition)
# ==========================================


class PoissonProblem1D:
    """定義一維泊松方程：u''(x) = f(x) 且 u(0) = u(1) = 0"""

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (np.pi**2) * torch.sin(np.pi * x)

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        return -np.sin(np.pi * x)


# ==========================================
# 2. 神經網路架構
# ==========================================

class SmoothNetwork(nn.Module):
    """標準單隱層特徵展開網路：S^N(x) = sum(c_i * tanh(w_i * x + b_i))"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.act = nn.Tanh()

        # 權重初始化
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

# ==========================================
# 3. 求解器基底與三種微分更新策略 (Solvers)
# ==========================================


class BaseSolver:

    def __init__(
        self,
        model: nn.Module,
        problem: PoissonProblem1D,
        lr: float = 1e-4,
        bc_weight: float = 10.0,
    ):
        self.model = model
        self.problem = problem
        self.lr = lr
        self.bc_weight = bc_weight
        self.loss_history = []
        self.time_history = []
        self.x_bc = torch.tensor([[0.0], [1.0]])

    def compute_second_derivative(self, x: torch.Tensor):
        """利用 PyTorch 自動微分計算二階導數 u_xx"""
        x.requires_grad_(True)
        u = self.model(x)

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            outputs=grad_u,
            inputs=x,
            grad_outputs=torch.ones_like(grad_u),
            create_graph=True,
        )[0]
        return u, u_xx

    def manual_gradient_step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.sub_(self.lr * p.grad)
                    p.grad.zero_()

    def step(self, x_batch: torch.Tensor) -> float:
        raise NotImplementedError

    def train(self, epochs: int, batch_size: int):
        total_time = 0.0
        for _ in range(epochs):
            x_batch = torch.rand(batch_size, 1)

            t_start = time.perf_counter()
            loss_val = self.step(x_batch)
            t_end = time.perf_counter()

            total_time += t_end - t_start
            self.loss_history.append(loss_val)
            self.time_history.append(total_time)


class PINNSolver(BaseSolver):
    """方法 1：標準 PINN"""

    def step(self, x_batch: torch.Tensor) -> float:
        # 1. 內部殘差 (PDE loss)
        _, u_xx = self.compute_second_derivative(x_batch)
        residual = u_xx - self.problem.f(x_batch)
        loss_pde = 0.5 * torch.mean(residual**2)

        # 2. 邊界條件殘差 (BC loss: u(0)=0, u(1)=0)
        u_bc = self.model(self.x_bc)
        loss_bc = 0.5 * torch.mean(u_bc**2)

        total_loss = loss_pde + self.bc_weight * loss_bc
        total_loss.backward()
        self.manual_gradient_step()
        return total_loss.item()


class QPDESolver(BaseSolver):
    """方法 2：QPDE 半梯度更新"""

    def step(self, x_batch: torch.Tensor) -> float:
        # 1. 內部半梯度
        u, u_xx = self.compute_second_derivative(x_batch)
        residual = u_xx - self.problem.f(x_batch)
        semi_grad_loss_pde = -torch.mean(residual.detach() * u)

        # 2. 邊界半梯度
        u_bc = self.model(self.x_bc)
        loss_bc = 0.5 * torch.mean(u_bc**2)

        semi_grad_loss = semi_grad_loss_pde + self.bc_weight * loss_bc
        semi_grad_loss.backward()
        self.manual_gradient_step()

        with torch.no_grad():
            true_loss = (
                0.5 * torch.mean(residual**2)
                + self.bc_weight * 0.5 * torch.mean(u_bc**2)
            ).item()
        return true_loss


class GaussNewtonSolver(BaseSolver):
    """方法 3：高斯—牛頓法"""

    def __init__(
        self,
        model: nn.Module,
        problem: PoissonProblem1D,
        damping: float = 1e-2,
        bc_weight: float = 10.0,
    ):
        super().__init__(model, problem, bc_weight=bc_weight)
        self.damping = damping

    def step(self, x_batch: torch.Tensor) -> float:
        batch_size = x_batch.shape[0]
        params = [p for p in self.model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)

        # 1. 計算內部物理殘差
        _, u_xx = self.compute_second_derivative(x_batch)
        res_pde = u_xx - self.problem.f(x_batch)

        # 2. 計算邊界殘差
        u_bc = self.model(self.x_bc)
        weight_factor = np.sqrt(self.bc_weight)
        res_bc = weight_factor * u_bc

        # 3. 建立內部 Jacobian
        J_pde = torch.zeros(
            batch_size, num_params, dtype=res_pde.dtype, device=x_batch.device
        )
        for i in range(batch_size):
            grad_params = torch.autograd.grad(
                u_xx[i],
                params,
                retain_graph=True,
                create_graph=False,
            )
            J_pde[i, :] = torch.cat([g.reshape(-1) for g in grad_params])

        # 4. 建立邊界 Jacobian
        J_bc = torch.zeros(2, num_params, dtype=res_bc.dtype,
                           device=x_batch.device)
        for i in range(2):
            grad_params = torch.autograd.grad(
                u_bc[i],
                params,
                retain_graph=(i < 1),
                create_graph=False,
            )
            J_bc[i, :] = weight_factor * torch.cat(
                [g.reshape(-1) for g in grad_params]
            )

        # 5. 拼接總殘差向量與總雅可比矩陣
        total_res = torch.cat([res_pde, res_bc], dim=0)  # [batch_size + 2, 1]
        # [batch_size + 2, num_params]
        total_J = torch.cat([J_pde, J_bc], dim=0)

        # 6. 組裝線性方程系統 (J^T J + lambda * I) * delta = - J^T * residual
        rhs = -torch.matmul(total_J.T, total_res)
        JTJ = torch.matmul(total_J.T, total_J)
        reg_I = self.damping * torch.eye(num_params, device=x_batch.device)
        A = JTJ + reg_I

        # 7. 求解增量 delta 並更新
        delta = torch.linalg.solve(A, rhs).squeeze(1)

        with torch.no_grad():
            offset = 0
            for p in params:
                p_len = p.numel()
                p.add_(delta[offset: offset + p_len].reshape(p.shape))
                offset += p_len

            true_total_loss = (
                0.5 * torch.mean(res_pde**2)
                + self.bc_weight * 0.5 * torch.mean(u_bc**2)
            ).item()

        return true_total_loss


# ==========================================
# 4. 主執行與視覺化 (Main Pipeline)
# ==========================================
def main():
    cfg = {
        "seed": 42,
        "epochs": 1000,
        "batch_size": 192,
        "hidden_dim": 64,
        "learning_rate": 1e-4,
        "damping": 1e-2,
        "bc_weight": 10.0,
        "n_eval_points": 200,
    }

    def init_model():
        torch.manual_seed(cfg["seed"])
        return SmoothNetwork(hidden_dim=cfg["hidden_dim"])

    problem = PoissonProblem1D()

    # 1. 實例化三個模型與求解器
    solvers = {
        "PINN (Full Gradient)": PINNSolver(
            init_model(),
            problem,
            lr=cfg["learning_rate"],
            bc_weight=cfg["bc_weight"],
        ),
        "QPDE (Semi-gradient)": QPDESolver(
            init_model(),
            problem,
            lr=cfg["learning_rate"],
            bc_weight=cfg["bc_weight"],
        ),
        "Gauss-Newton": GaussNewtonSolver(
            init_model(),
            problem,
            damping=cfg["damping"],
            bc_weight=cfg["bc_weight"],
        ),
    }

    # 2. 統一執行訓練
    for name, solver in solvers.items():
        solver.train(epochs=cfg["epochs"], batch_size=cfg["batch_size"])

    # 3. 評估資料產生
    x_test_np = np.linspace(0, 1, cfg["n_eval_points"]).reshape(-1, 1)
    x_test_torch = torch.tensor(x_test_np)
    u_exact = problem.exact_solution(x_test_np)

    preds = {}
    with torch.no_grad():
        for name, solver in solvers.items():
            preds[name] = solver.model(x_test_torch).numpy()

    # 4. 繪製統一比較圖 (1x3 排列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    styles = {
        "PINN (Full Gradient)": {
            "color": "#7570B3",
            "linestyle": ":",
            "alpha": 0.8,
        },
        "QPDE (Semi-gradient)": {
            "color": "#1B9E77",
            "linestyle": "-.",
            "alpha": 0.9,
        },
        "Gauss-Newton": {"color": "#D95F02", "linestyle": "--", "alpha": 0.9},
    }

    # 圖 1：解的擬合曲線
    axes[0].plot(
        x_test_np,
        u_exact,
        "k--",
        label="Exact $u^*(x)=-\sin(\pi x)$",
        linewidth=2.5,
    )
    for name, u_pred in preds.items():
        axes[0].plot(
            x_test_np, u_pred, label=name, linewidth=2.0, **styles[name]
        )
    axes[0].set_title("PDE Solution Approximation", fontsize=13)
    axes[0].set_xlabel("$x$", fontsize=11)
    axes[0].set_ylabel("$u(x)$", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 圖 2：殘差對訓練輪數 (Epoch)
    for name, solver in solvers.items():
        axes[1].plot(
            solver.loss_history,
            label=name,
            color=styles[name]["color"],
            alpha=0.7,
        )
    axes[1].set_title("Total Loss vs. Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Total Loss (PDE + BC)", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 圖 3：殘差對累積執行時間 (Wall-clock Time)
    for name, solver in solvers.items():
        axes[2].plot(
            solver.time_history,
            solver.loss_history,
            label=name,
            color=styles[name]["color"],
            alpha=0.7,
        )
    axes[2].set_title("Total Loss vs. Wall-clock Time", fontsize=13)
    axes[2].set_xlabel("Elapsed Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Total Loss (PDE + BC)", fontsize=11)
    axes[2].set_yscale("log")

    max_time_qpde = solvers["QPDE (Semi-gradient)"].time_history[-1]
    axes[2].set_xlim(0, max_time_qpde)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
