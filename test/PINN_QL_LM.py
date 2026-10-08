import numpy as np
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod


# ============================================================
# 1. Poisson Problem
# ============================================================

class Poisson1D:

    def __init__(self, n_collocation=63):
        self.n_collocation = n_collocation
        self.x = np.linspace(
            0.0,
            1.0,
            n_collocation + 2
        )[1:-1]

    def forcing(self, x):
        return np.pi**2 * np.sin(np.pi * x)

    def exact_solution(self, x):
        return -np.sin(np.pi * x)

    def residual(self, network, theta):
        u_xx = network.d2u_dx2(self.x, theta)
        return self.forcing(self.x) - u_xx

    def residual_norm(self, network, theta):
        r = self.residual(network, theta)
        return np.linalg.norm(r)


# ============================================================
# 2. Neural Network
# ============================================================

class NeuralNetwork1D:

    """
    """

    def __init__(self, N=20, seed=1234):
        self.N = N
        rng = np.random.default_rng(seed)
        self.c = 0.1 * rng.standard_normal(N)
        self.w = 0.1 * rng.standard_normal(N)
        self.b = 0.1 * rng.standard_normal(N)

    @property
    def n_parameters(self):
        return 3 * self.N

    def pack(self):
        return np.concatenate([
            self.c,
            self.w,
            self.b
        ])

    def unpack(self, theta):
        N = self.N
        c = theta[:N]
        w = theta[N:2*N]
        b = theta[2*N:3*N]
        return c, w, b

    def eta(self, x):
        return 4.0 * x * (1.0 - x)

    def eta_x(self, x):
        return 4.0 - 8.0 * x

    def eta_xx(self, x):
        return -8.0 * np.ones_like(x)

    """
    """

    def forward(self, x, theta):
        c, w, b = self.unpack(theta)
        z = x[:, None] * w[None, :] + b[None, :]
        h = np.tanh(z)
        S = np.sum(c[None, :] * h, axis=1)
        return self.eta(x) * S

    def du_dx(self, x, theta):
        c, w, b = self.unpack(theta)
        z = x[:, None] * w[None, :] + b[None, :]
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - tanh_z**2

        S = np.sum(
            c[None, :] * tanh_z,
            axis=1
        )

        S_x = np.sum(
            c[None, :] * w[None, :] * sech2_z,
            axis=1
        )

        return (
            self.eta_x(x) * S
            + self.eta(x) * S_x
        )

    def d2u_dx2(self, x, theta):
        c, w, b = self.unpack(theta)
        z = x[:, None] * w[None, :] + b[None, :]
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - tanh_z**2

        S = np.sum(
            c[None, :] * tanh_z,
            axis=1
        )

        S_x = np.sum(
            c[None, :] * w[None, :] * sech2_z,
            axis=1
        )

        S_xx = np.sum(
            c[None, :]
            * w[None, :]**2
            * (-2.0 * tanh_z * sech2_z),
            axis=1
        )

        return (
            self.eta_xx(x) * S
            + 2.0 * self.eta_x(x) * S_x
            + self.eta(x) * S_xx
        )

    def du_dtheta(self, x, theta):

        c, w, b = self.unpack(theta)
        z = x[:, None] * w[None, :] + b[None, :]
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - tanh_z**2
        eta = self.eta(x)

        d_c = eta[:, None] * tanh_z

        d_w = (
            eta[:, None]
            * c[None, :]
            * sech2_z
            * x[:, None]
        )

        d_b = (
            eta[:, None]
            * c[None, :]
            * sech2_z
        )

        return np.concatenate(
            [d_c, d_w, d_b],
            axis=1
        )

    def d2u_dtheta(self, x, theta):

        c, w, b = self.unpack(theta)
        z = x[:, None] * w[None, :] + b[None, :]
        t = np.tanh(z)
        s = 1.0 - t**2

        eta = self.eta(x)
        eta_x = self.eta_x(x)
        eta_xx = self.eta_xx(x)

        dS_dw = (
            c[None, :]
            * s
            * x[:, None]
        )

        dS_db = (
            c[None, :] * s
        )

        dSx_dc = (
            w[None, :] * s
        )

        dSx_dw = (
            c[None, :]
            * (
                s
                + w[None, :]
                * (-2.0 * t * s)
                * x[:, None]
            )
        )

        dSx_db = (
            c[None, :]
            * w[None, :]
            * (-2.0 * t * s)
        )

        q = -2.0 * t * s
        dq_dz = (
            -2.0
            * s
            * (1.0 - 3.0 * t**2)
        )

        dSxx_dc = (
            w[None, :]**2 * q
        )

        dSxx_dw = (
            c[None, :]
            * (
                2.0 * w[None, :] * q
                + w[None, :]**2
                * dq_dz
                * x[:, None]
            )
        )

        dSxx_db = (
            c[None, :]
            * w[None, :]**2
            * dq_dz
        )

        J_c = (
            eta_xx[:, None] * t
            + 2.0 * eta_x[:, None] * dSx_dc
            + eta[:, None] * dSxx_dc
        )

        J_w = (
            eta_xx[:, None] * dS_dw
            + 2.0 * eta_x[:, None] * dSx_dw
            + eta[:, None] * dSxx_dw
        )

        J_b = (
            eta_xx[:, None] * dS_db
            + 2.0 * eta_x[:, None] * dSx_db
            + eta[:, None] * dSxx_db
        )

        J = np.concatenate(
            [J_c, J_w, J_b],
            axis=1
        )

        return J


# ============================================================
# 3. Base Optimizer
# ============================================================

class OptimizerBase(ABC):

    def __init__(
        self,
        problem,
        network,
        theta0,
        alpha=1e-3,
        max_iter=100000,
        tolerance=1e-6
    ):
        self.problem = problem
        self.network = network

        self.theta = theta0.copy()

        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance

        # History
        self.iterations = []
        self.times = []
        self.residuals = []

        self.converged = False
        self.convergence_time = None
        self.convergence_iteration = None

    @abstractmethod
    def step(self):
        pass

    def record(self, iteration, elapsed_time):
        """
        """
        residual = self.problem.residual_norm(
            self.network,
            self.theta
        )

        self.iterations.append(iteration)
        self.times.append(elapsed_time)
        self.residuals.append(residual)

        if (
            not self.converged
            and residual <= self.tolerance
        ):
            self.converged = True
            self.convergence_time = elapsed_time
            self.convergence_iteration = iteration

    def run(self):

        start_time = time.perf_counter()

        self.record(
            iteration=0,
            elapsed_time=0.0
        )

        for k in range(1, self.max_iter + 1):

            self.step()

            elapsed_time = time.perf_counter() - start_time

            self.record(
                iteration=k,
                elapsed_time=elapsed_time
            )
            """
            """
            if self.converged:
                break

        return self.theta


# ============================================================
# 4. Gradient Method
# ============================================================

class GradientMethod(OptimizerBase):

    def step(self):

        r = self.problem.residual(
            self.network,
            self.theta
        )

        J = self.network.d2u_dtheta(
            self.problem.x,
            self.theta
        )

        gradient = -J.T @ r

        self.theta = (
            self.theta
            - self.alpha * gradient
        )


# ============================================================
# 5. Semi-Gradient Method
# ============================================================

class SemiGradientMethod(OptimizerBase):

    def step(self):

        r = self.problem.residual(
            self.network,
            self.theta
        )

        G = self.network.du_dtheta(
            self.problem.x,
            self.theta
        )

        gradient = G.T @ r

        self.theta = (
            self.theta
            - self.alpha * gradient
        )


# ============================================================
# 6. Damped Gauss-Newton Method
# ============================================================

class DampedGaussNewton(OptimizerBase):

    def __init__(
        self,
        problem,
        network,
        theta0,
        damping=1e-2,
        max_iter=1000,
        tolerance=1e-6
    ):

        super().__init__(
            problem=problem,
            network=network,
            theta0=theta0,
            alpha=None,
            max_iter=max_iter,
            tolerance=tolerance
        )

        self.damping = damping

    def step(self):

        r = self.problem.residual(
            self.network,
            self.theta
        )

        J = self.network.d2u_dtheta(
            self.problem.x,
            self.theta
        )

        n_parameters = len(self.theta)

        A = (
            J.T @ J
            + self.damping * np.eye(n_parameters)
        )

        rhs = J.T @ r
        delta = np.linalg.solve(A, rhs)
        self.theta = self.theta + delta


# ============================================================
# 7. Experiment Manager
# ============================================================

class Experiment:

    def __init__(
        self,
        N=64,
        n_collocation=63,
        seed=1234
    ):

        self.problem = Poisson1D(
            n_collocation=n_collocation
        )

        self.network = NeuralNetwork1D(
            N=N,
            seed=seed
        )

        self.theta0 = self.network.pack()
        self.results = {}

    def run_all(
        self,
        alpha_gradient=1e-3,
        alpha_semi=1e-3,
        damping=100,
        max_iter_gradient=10000,
        max_iter_semi=10000,
        max_iter_newton=1000,
        tolerance=1e-6
    ):

        gradient = GradientMethod(
            problem=self.problem,
            network=self.network,
            theta0=self.theta0,
            alpha=alpha_gradient,
            max_iter=max_iter_gradient,
            tolerance=tolerance
        )

        print("Running Gradient Method...")
        gradient.run()
        self.results["Gradient"] = gradient

        semi_gradient = SemiGradientMethod(
            problem=self.problem,
            network=self.network,
            theta0=self.theta0,
            alpha=alpha_semi,
            max_iter=max_iter_semi,
            tolerance=tolerance
        )

        print("Running Semi-gradient Method...")
        semi_gradient.run()
        self.results["Semi-gradient"] = semi_gradient

        newton = DampedGaussNewton(
            problem=self.problem,
            network=self.network,
            theta0=self.theta0,
            damping=damping,
            max_iter=max_iter_newton,
            tolerance=tolerance
        )

        print("Running Damped Gauss-Newton Method...")
        newton.run()
        self.results["Damped Gauss-Newton"] = newton


# ============================================================
# 8. Plot 1: Solution
# ============================================================

def plot_solution(
    experiment,
    n_plot=500
):
    problem = experiment.problem
    network = experiment.network
    x_plot = np.linspace(0.0, 1.0, n_plot)
    u_exact = problem.exact_solution(x_plot)

    style_map = {
        "Damped Gauss-Newton": {
            "ls": "--",
            "color": "#D95F02",
            "lw": 2.2,
            "zorder": 4
        },
        "Gradient": {
            "ls": ":",
            "color": "#7570B3",
            "lw": 2.5,
            "zorder": 3
        },
        "Semi-gradient": {
            "ls": "-.",
            "color": "#1B9E77",
            "lw": 2.0,
            "zorder": 2
        }
    }

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        x_plot,
        u_exact,
        label=r"Exact solution: $-\sin(\pi x)$",
        linestyle=(0, (5, 5)),
        color="black",
        linewidth=2.8,
        alpha=0.7,
        zorder=1
    )

    for name, optimizer in experiment.results.items():
        u_pred = network.forward(x_plot, optimizer.theta)

        style = style_map.get(
            name,
            {"ls": ":", "color": "gray", "lw": 2.0, "zorder": 2}
        )

        plt.plot(
            x_plot,
            u_pred,
            label=name,
            linestyle=style["ls"],
            color=style["color"],
            linewidth=style["lw"],
            zorder=style["zorder"]
        )

    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u(x)$", fontsize=12)
    plt.title("Neural Network Approximation of 1D Poisson Equation", fontsize=13)

    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    # plt.show()


# ============================================================
# 9. Plot 2: Residual vs Iteration
# ============================================================

def plot_residual_vs_iteration(
    experiment,
    tolerance=1e-6
):
    style_map = {
        "Damped Gauss-Newton": {
            "ls": "--",
            "color": "#D95F02",   # 橘紅
            "lw": 2.2,
            "marker": "o",
            "zorder": 4
        },
        "Gradient": {
            "ls": ":",
            "color": "#7570B3",   # 紫藍
            "lw": 2.5,
            "marker": "s",
            "zorder": 3
        },
        "Semi-gradient": {
            "ls": "-.",
            "color": "#1B9E77",   # 藍綠
            "lw": 2.0,
            "marker": "^",
            "zorder": 2
        }
    }

    plt.figure(figsize=(9, 6))

    for name, optimizer in experiment.results.items():
        residuals = np.array(optimizer.residuals)
        iterations = np.array(optimizer.iterations)

        style = style_map.get(
            name,
            {"ls": ":", "color": "gray", "lw": 2.0, "marker": "o", "zorder": 2}
        )

        plt.semilogy(
            iterations,
            residuals,
            label=name,
            linestyle=style["ls"],
            color=style["color"],
            linewidth=style["lw"],
            zorder=style["zorder"]
        )

        indices = np.where(
            residuals <= tolerance
        )[0]

        if len(indices) > 0:
            idx = indices[0]

            r_conv = residuals[idx]
            iter_conv = optimizer.iterations[idx]
            x_conv = iterations[idx]

            plt.scatter(
                x_conv,
                r_conv,
                s=60,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=1.2,
                zorder=style["zorder"] + 5
            )

            plt.annotate(
                f"iter = {iter_conv}",
                xy=(x_conv, r_conv),
                xytext=(10, 15),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->",
                    color=style["color"],
                    lw=1.2
                ),
                fontsize=9,
                fontweight="bold",
                color=style["color"]
            )

            print(
                f"{name}: "
                f"residual <= {tolerance:.0e}, "
                f"iteration = {iter_conv}"
            )
        else:
            print(
                f"{name}: "
                f"did NOT reach {tolerance:.0e}"
            )

    plt.axhline(
        tolerance,
        linestyle=(0, (4, 4)),
        color="black",
        linewidth=1.2,
        alpha=0.7,
        label=r"$10^{-6}$ target"
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(r"Residual norm $\|r\|_2$ (log)", fontsize=12)
    plt.title("Convergence History: Residual vs. Iteration", fontsize=13)

    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    # plt.show()


# ============================================================
# 10. Print Final Results
# ============================================================

def print_summary(experiment):

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 75)

    for name, optimizer in experiment.results.items():

        if not optimizer.residuals:
            print(f"\n{name}")
            print("-" * 75)
            print("No iteration data recorded.")
            continue

        final_residual = optimizer.residuals[-1]
        final_iteration = optimizer.iterations[-1]
        final_time = optimizer.times[-1]

        print(f"\n{name}")
        print("-" * 75)

        print(f"Final residual   : {final_residual:.6e}")
        print(f"Total iterations : {final_iteration}")
        print(f"Total run time   : {final_time:.6f} s")

        if optimizer.converged:
            print(f"Reached tol      : YES")
            print(f"Convergence iter : {optimizer.convergence_iteration}")
            print(f"Convergence time : {optimizer.convergence_time:.6f} s")
        else:
            print(f"Reached tol      : NO")

    print("\n" + "=" * 75)


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    experiment = Experiment(
        N=64,
        n_collocation=192,
        seed=42
    )

    experiment.run_all(
        alpha_gradient=1e-4,
        alpha_semi=1e-4,
        damping=1e-2,
        max_iter_gradient=1000,
        max_iter_semi=1000,
        max_iter_newton=1000,
        tolerance=1e-6
    )

    print_summary(experiment)
    plot_solution(experiment)
    plot_residual_vs_iteration(experiment, tolerance=1e-6)
    plt.show()
