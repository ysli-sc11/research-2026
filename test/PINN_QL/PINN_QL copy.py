import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 設定隨機種子以保證實驗可重現
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 物理問題定義 (Problem Definition)
# ==========================================


class PoissonProblem1D:
    """定義一維泊松方程：u''(x) = f(x) 且 u(0) = u(1) = 0"""

    def __init__(self):
        pass

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """源項 f(x) = -pi^2 * sin(pi * x)"""
        return - (np.pi ** 2) * torch.sin(np.pi * x)

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        """真實解析解 u*(x) = sin(pi * x)"""
        return np.sin(np.pi * x)


# ==========================================
# 2. 神經網路與 Ansatz 架構 (Model)
# ==========================================
class SmoothNetwork(nn.Module):
    """
    標準單隱層特徵展開網路：
    S^N(x) = sum(c_i * tanh(w_i * x + b_i))
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.act = nn.Tanh()  # 滿足 C^3 有界且光滑的激勵函數

        # 初始化權重符合 Assumption 16
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.act(self.fc1(x))
        out = self.fc2(features)
        return out


class ConstrainedPINN(nn.Module):
    """
    結合邊界輔助函數的模型 (Ansatz)：
    u_theta(x) = eta(x) * S^N(x)
    其中 eta(x) = 4 * x * (1 - x) 保證邊界 u(0)=u(1)=0 嚴格成立
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def eta(self, x: torch.Tensor) -> torch.Tensor:
        return 4.0 * x * (1.0 - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eta(x) * self.backbone(x)


# ==========================================
# 3. 求解器基底與兩種微分策略 (Solvers)
# ==========================================
class BaseSolver:
    """求解器基類：負責資料取樣、微分計算、訓練循環與計時記錄"""

    def __init__(self, model: ConstrainedPINN, problem: PoissonProblem1D, lr: float = 1e-2):
        self.model = model
        self.problem = problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history = []
        self.time_history = []  # 記錄每個 epoch 結束時的累積秒數

    def compute_second_derivative(self, x: torch.Tensor):
        """利用 PyTorch 自動微分計算 u 對 x 的二階導數 u_xx"""
        x.requires_grad_(True)
        u = self.model(x)

        grad_u = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            outputs=grad_u, inputs=x,
            grad_outputs=torch.ones_like(grad_u),
            create_graph=True
        )[0]
        return u, u_xx

    def step(self, x_batch: torch.Tensor):
        """由子類實現具體的梯度或半梯度更新步驟"""
        raise NotImplementedError

    def train(self, epochs: int = 1500, batch_size: int = 64):
        total_time = 0.0
        for epoch in range(epochs):
            # 隨機在 (0, 1) 區間抽樣空間座標點 x
            x_batch = torch.rand(batch_size, 1)

            # 精確記錄 step 執行時間
            t_start = time.perf_counter()
            loss_val = self.step(x_batch)
            t_end = time.perf_counter()

            total_time += (t_end - t_start)
            self.loss_history.append(loss_val)
            self.time_history.append(total_time)


class PINNSolver(BaseSolver):
    """標準 PINN 求解器：使用殘差平方損失 (全梯度反向傳播)"""

    def step(self, x_batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        u, u_xx = self.compute_second_derivative(x_batch)
        residual = u_xx - self.problem.f(x_batch)

        # Loss = 0.5 * sum( (u_xx - f)^2 )
        # 梯度會直接流經 u_xx，反向傳播圖包含二階導數節點
        loss = 0.5 * torch.mean(residual ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class QPDESolver(BaseSolver):
    """論文 QPDE 求解器：使用半梯度 / Q-learning 定點更新"""

    def step(self, x_batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        u, u_xx = self.compute_second_derivative(x_batch)
        residual = u_xx - self.problem.f(x_batch)

        # 使用 detach() 截斷殘差的計算圖，只對 u 做參數梯度反向傳播
        semi_grad_loss = -torch.mean(residual.detach() * u)
        semi_grad_loss.backward()
        self.optimizer.step()

        # 記錄真實物理殘差平方以利公平比對
        with torch.no_grad():
            true_l2_residual = 0.5 * torch.mean(residual ** 2).item()
        return true_l2_residual


# ==========================================
# 4. 主執行與繪圖視覺化 (Comparison)
# ==========================================
def main():
    problem = PoissonProblem1D()
    epochs = 1500
    lr = 0.01

    # 1. 訓練 PINN
    torch.manual_seed(42)
    pinn_model = ConstrainedPINN(SmoothNetwork(hidden_dim=64))
    pinn_solver = PINNSolver(pinn_model, problem, lr=lr)
    pinn_solver.train(epochs=epochs)

    # 2. 訓練 QPDE (半梯度 Q-learning)
    torch.manual_seed(42)
    qpde_model = ConstrainedPINN(SmoothNetwork(hidden_dim=64))
    qpde_solver = QPDESolver(qpde_model, problem, lr=lr)
    qpde_solver.train(epochs=epochs)

    # 3. 評估資料產生
    x_test_np = np.linspace(0, 1, 200).reshape(-1, 1)
    x_test_torch = torch.tensor(x_test_np, dtype=torch.float32)
    u_exact = problem.exact_solution(x_test_np)

    with torch.no_grad():
        u_pinn = pinn_model(x_test_torch).numpy()
        u_qpde = qpde_model(x_test_torch).numpy()

    # 4. 繪製三張格式統一的比較圖 (1x3 排列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 圖 1：解的擬合曲線
    axes[0].plot(x_test_np, u_exact, 'k--',
                 label="Exact $u^*(x)=\sin(\pi x)$", linewidth=2.5)
    axes[0].plot(x_test_np, u_pinn, 'C0-',
                 label="PINN (Full Gradient)", alpha=0.8, linewidth=2)
    axes[0].plot(x_test_np, u_qpde, 'C1-.',
                 label="QPDE (Semi-gradient)", alpha=0.9, linewidth=2)
    axes[0].set_title("PDE Solution Approximation", fontsize=13)
    axes[0].set_xlabel("$x$", fontsize=11)
    axes[0].set_ylabel("$u(x)$", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 圖 2：殘差對訓練輪數 (Epoch)
    axes[1].plot(pinn_solver.loss_history, 'C0-', label="PINN", alpha=0.7)
    axes[1].plot(qpde_solver.loss_history, 'C1-', label="QPDE", alpha=0.7)
    axes[1].set_title("Residual vs. Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Mean Squared Residual", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 圖 3：殘差對累積執行時間 (Wall-clock Time)
    axes[2].plot(pinn_solver.time_history, pinn_solver.loss_history,
                 'C0-', label="PINN", alpha=0.7)
    axes[2].plot(qpde_solver.time_history, qpde_solver.loss_history,
                 'C1-', label="QPDE", alpha=0.7)
    axes[2].set_title("Residual vs. Wall-clock Time", fontsize=13)
    axes[2].set_xlabel("Elapsed Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Mean Squared Residual", fontsize=11)
    axes[2].set_yscale("log")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
