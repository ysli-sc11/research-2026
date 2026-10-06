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
        alpha=1e-4,
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
    """
    Damped Gauss-Newton method.

    Solve

        (J^T J + lambda I) delta
            = J^T r

    and update

        theta_{k+1}
            = theta_k + delta
    """

    def __init__(
        self,
        problem,
        network,
        theta0,
        damping=1e-4,
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

        # Do NOT calculate inverse(A).
        #
        # Instead solve:
        #
        # A delta = rhs
        #
        delta = np.linalg.solve(A, rhs)

        self.theta = self.theta + delta


# ============================================================
# 7. Experiment Manager
# ============================================================

class Experiment:

    def __init__(
        self,
        N=20,
        n_collocation=101,
        seed=1234
    ):

        self.problem = Poisson1D(
            n_collocation=n_collocation
        )

        self.network = NeuralNetwork1D(
            N=N,
            seed=seed
        )

        # Same initial parameters for all methods
        self.theta0 = self.network.pack()

        self.results = {}

    def run_all(
        self,
        alpha_gradient=1e-4,
        alpha_semi=1e-4,
        damping=1e-4,
        max_iter_gradient=100000,
        max_iter_semi=100000,
        max_iter_newton=1000,
        tolerance=1e-6
    ):

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Semi-gradient
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Damped Gauss-Newton
        # ----------------------------------------------------

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

    x_plot = np.linspace(
        0.0,
        1.0,
        n_plot
    )

    u_exact = problem.exact_solution(x_plot)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_plot,
        u_exact,
        label="Exact solution",
        linewidth=2
    )

    for name, optimizer in experiment.results.items():

        u_pred = network.forward(
            x_plot,
            optimizer.theta
        )

        plt.plot(
            x_plot,
            u_pred,
            label=name
        )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")

    plt.title(
        "Neural Network Approximation of 1D Poisson Equation"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Plot 2: Residual vs cumulative execution time
# ============================================================

def plot_residual_vs_time(
    experiment,
    tolerance=1e-6
):

    plt.figure(figsize=(9, 6))

    for name, optimizer in experiment.results.items():

        times = np.array(optimizer.times)
        residuals = np.array(optimizer.residuals)
        iterations = np.array(optimizer.iterations)

        plt.semilogy(
            times,
            residuals,
            label=name
        )

        # ----------------------------------------------------
        # Find first iteration reaching tolerance
        # ----------------------------------------------------

        indices = np.where(
            residuals <= tolerance
        )[0]

        if len(indices) > 0:

            idx = indices[0]

            t_conv = times[idx]
            r_conv = residuals[idx]
            iter_conv = iterations[idx]

            plt.scatter(
                t_conv,
                r_conv,
                s=60,
                zorder=5
            )

            plt.annotate(
                f"iter = {iter_conv}\ntime = {t_conv:.4f} s",
                xy=(t_conv, r_conv),
                xytext=(10, 15),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->"
                )
            )

            print(
                f"{name}: "
                f"residual <= {tolerance:.0e}, "
                f"iteration = {iter_conv}, "
                f"time = {t_conv:.6f} s"
            )

        else:

            print(
                f"{name}: "
                f"did NOT reach {tolerance:.0e}"
            )

    plt.axhline(
        tolerance,
        linestyle="--",
        linewidth=1,
        label=r"$10^{-6}$ target"
    )

    plt.xlabel(
        "Cumulative execution time (s)"
    )

    plt.ylabel(
        r"$\|r\|_2$"
    )

    plt.title(
        "Residual vs. Cumulative Execution Time"
    )

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. Print Final Results
# ============================================================

def print_summary(experiment):

    print("\n")
    print("=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    for name, optimizer in experiment.results.items():

        final_residual = optimizer.residuals[-1]
        final_iteration = optimizer.iterations[-1]
        final_time = optimizer.times[-1]

        print(f"\n{name}")
        print("-" * 75)

        print(
            f"Final residual : {final_residual:.6e}"
        )

        print(
            f"Iterations     : {final_iteration}"
        )

        print(
            f"Time           : {final_time:.6f} s"
        )

        if optimizer.converged:

            print(
                f"Reached 1e-6   : YES"
            )

            print(
                f"Convergence iter: "
                f"{optimizer.convergence_iteration}"
            )

            print(
                f"Convergence time: "
                f"{optimizer.convergence_time:.6f} s"
            )

        else:

            print(
                f"Reached 1e-6   : NO"
            )


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------

    experiment = Experiment(
        N=20,
        n_collocation=101,
        seed=1234
    )

    # --------------------------------------------------------
    # Run all three methods
    #
    # These learning rates / damping are initial values.
    # They may need tuning depending on N and initialization.
    # --------------------------------------------------------

    experiment.run_all(
        alpha_gradient=1e-5,
        alpha_semi=1e-5,
        damping=1e-4,
        max_iter_gradient=100000,
        max_iter_semi=100000,
        max_iter_newton=10000,
        tolerance=1e-6
    )

    # --------------------------------------------------------
    # Print numerical results
    # --------------------------------------------------------

    print_summary(experiment)

    # --------------------------------------------------------
    # Figure 1
    # --------------------------------------------------------

    plot_solution(
        experiment
    )

    # --------------------------------------------------------
    # Figure 2
    # --------------------------------------------------------

    plot_residual_vs_time(
        experiment,
        tolerance=1e-6
    )
