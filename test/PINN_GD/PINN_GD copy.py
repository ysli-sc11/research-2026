import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------- 設定 ----------------
n = 64
m = 3 * n + 1
x = np.linspace(0.0, 1.0, m)          # collocation points
def f(x): return -np.pi**2 * np.sin(np.pi * x)


fx = f(x)
BETA = 1.0                             # 邊界懲罰權重，設 0 就是純 PDE 殘差

# ---------------- 模型 ----------------


def unpack(theta):
    return theta[:n], theta[n:2*n], theta[2*n:]


def u_xx(theta, x):
    c, w, b = unpack(theta)
    t = np.tanh(np.outer(x, w) + b)            # (m, n)
    g = t - t**3
    return -2.0 * (g * (c * w**2)).sum(axis=1)


def u(theta, x):
    c, w, b = unpack(theta)
    return np.tanh(np.outer(x, w) + b) @ c

# ---------------- Loss 與梯度 ----------------


def loss(theta):
    r = u_xx(theta, x) - fx
    L = 0.5 * np.sum(r**2)
    if BETA > 0:
        L += 0.5 * BETA * (u(theta, np.array([0.0]))[0]**2
                           + u(theta, np.array([1.0]))[0]**2)
    return L


def grad(theta):
    c, w, b = unpack(theta)
    t = np.tanh(np.outer(x, w) + b)
    s = 1 - t**2
    g = t - t**3
    gp = s * (1 - 3 * t**2)
    r = -2.0 * (g * (c * w**2)).sum(axis=1) - fx   # (m,)

    dc = -2.0 * (g * w**2)                          # (m, n)
    dw = -2.0 * c * (2 * w * g + (w**2) * gp * x[:, None])
    db = -2.0 * c * (w**2) * gp
    G = np.concatenate([r @ dc, r @ dw, r @ db])

    if BETA > 0:
        # u(0) = sum c tanh(b)
        t0 = np.tanh(b)
        s0 = 1 - t0**2
        u0 = c @ t0
        G[:n] += BETA * u0 * t0
        G[2*n:] += BETA * u0 * c * s0
        # u(1) = sum c tanh(w+b)
        t1 = np.tanh(w + b)
        s1 = 1 - t1**2
        u1 = c @ t1
        G[:n] += BETA * u1 * t1
        G[n:2*n] += BETA * u1 * c * s1
        G[2*n:] += BETA * u1 * c * s1
    return G


def jacobian(theta):
    c, w, b = unpack(theta)
    t = np.tanh(np.outer(x, w) + b)
    s = 1 - t**2
    g = t - t**3
    gp = s * (1 - 3 * t**2)

    dc = -2.0 * (g * w**2)
    dw = -2.0 * c * (2 * w * g + (w**2) * gp * x[:, None])
    db = -2.0 * c * (w**2) * gp
    J = np.hstack([dc, dw, db])                    # (m, 3n)

    if BETA > 0:
        sb = np.sqrt(BETA)
        t0 = np.tanh(b)
        s0 = 1 - t0**2
        t1 = np.tanh(w + b)
        s1 = 1 - t1**2
        zeros = np.zeros(n)
        row0 = np.concatenate([t0, zeros,  c * s0])   # ∂u(0)/∂θ
        row1 = np.concatenate([t1, c * s1, c * s1])   # ∂u(1)/∂θ
        J = np.vstack([J, sb * row0, sb * row1])
    return J


def cond_JtJ(theta):
    sv = np.linalg.svd(jacobian(theta), compute_uv=False)
    lam_max, lam_min = sv[0]**2, sv[-1]**2
    return lam_max / max(lam_min, 1e-300), lam_min     # κ(JᵀJ), λ_min(JᵀJ)

# ---------------- Wolfe 線搜尋 (Armijo + Curvature) ----------------


def wolfe_line_search(theta, p, L0, g0, c1=1e-4, c2=0.9, t0=1.0, max_iter=60):
    """二分/擴張法，找滿足
       Armijo:    L(θ+tp) <= L(θ) + c1 t g·p
       Curvature: ∇L(θ+tp)·p >= c2 g·p
    """
    dg0 = g0 @ p                       # < 0
    lo, hi, t = 0.0, np.inf, t0
    for _ in range(max_iter):
        th = theta + t * p
        Lt = loss(th)
        if not np.isfinite(Lt) or Lt > L0 + c1 * t * dg0:   # Armijo 不過 -> 步長太大
            hi = t
        else:
            gt = grad(th)
            if gt @ p < c2 * dg0:                            # Curvature 不過 -> 步長太小
                lo = t
            else:
                return t, Lt, gt
        t = 0.5 * (lo + hi) if np.isfinite(hi) else 2.0 * \
            lo if lo > 0 else 2.0 * t
    return t, Lt, grad(theta + t * p)   # 找不到就回傳最後一個


# ---------------- 初始化 ----------------
theta = np.concatenate([
    0.1 * np.random.randn(n),            # c
    np.random.uniform(-3, 3, n),         # w
    np.random.uniform(-3, 3, n),         # b
])

# ---------------- 梯度檢查 ----------------


def grad_check(theta, k=5, eps=1e-6):
    G = grad(theta)
    for i in np.random.choice(len(theta), k, replace=False):
        e = np.zeros_like(theta)
        e[i] = eps
        num = (loss(theta + e) - loss(theta - e)) / (2 * eps)
        print(f"idx {i:3d}: analytic={G[i]: .6e}  numeric={num: .6e}")


grad_check(theta)

# ---------------- 梯度下降 ----------------
max_iter, tol = 20000, 1e-8
cond_every = 1                                    # 每幾步算一次 SVD
print_every = 1                                  # 必須是 cond_every 的倍數
L = loss(theta)
G = grad(theta)
hist = [L]
gn_hist = [np.linalg.norm(G)]
kappa0, lam0 = cond_JtJ(theta)
cond_iters, cond_hist, lam_min_hist = [0], [kappa0], [lam0]

for k in range(max_iter):
    gn = np.linalg.norm(G)
    if gn < tol:
        break
    p = -G
    tk, L, G_new = wolfe_line_search(theta, p, L, G)
    theta = theta + tk * p
    G = G_new
    hist.append(L)
    gn_hist.append(np.linalg.norm(G))

    if (k + 1) % cond_every == 0:
        kappa, lam_min = cond_JtJ(theta)
        cond_iters.append(k + 1)
        cond_hist.append(kappa)
        lam_min_hist.append(lam_min)
        if (k + 1) % print_every == 0:
            print(f"iter {k+1:6d}  L={L:.4e}  |grad|={gn_hist[-1]:.3e}  "
                  f"t_k={tk:.3e}  cond={kappa:.3e}  lam_min={lam_min:.3e}")

# ---------------- 結果 ----------------
xt = np.linspace(0, 1, 400)
ut = u(theta, xt)
err = np.max(np.abs(ut - np.sin(np.pi * xt)))
print(f"final L = {L:.4e}, max|u_θ - sin(πx)| = {err:.3e}")

gn_arr = np.array(gn_hist)
gn_min = np.minimum.accumulate(gn_arr)             # min_{n<=k} ||∇L(θ_n)||
its = np.arange(len(gn_arr))
its1 = its[1:]                                     # log-log 圖不能有 iteration=0

# ---- Figure 1: Solution, Loss ----
fig1, ax1 = plt.subplots(1, 2, figsize=(11, 4))
ax1[0].plot(xt, np.sin(np.pi * xt), label="exact sin(πx)")
ax1[0].plot(xt, ut, "--", label="PINN")
ax1[0].legend()
ax1[0].set_title("Solution")
ax1[1].semilogy(hist)
ax1[1].set_title(r"Loss in $L_2$ Norm (semilog)")
ax1[1].set_xlabel("iteration")
plt.tight_layout()

# ---- Figure 2: 梯度範數 (4 張) ----
fig2, ax2 = plt.subplots(2, 2, figsize=(12, 8))
ax2[0, 0].semilogy(its, gn_arr)
ax2[0, 0].set_title(r"$\|\nabla L(\theta_k)\|_2$ (semilog)")
ax2[0, 0].set_xlabel("iteration")

ax2[0, 1].loglog(its1, gn_arr[1:])
ax2[0, 1].set_title(r"$\|\nabla L(\theta_k)\|_2$ (log-log)")
ax2[0, 1].set_xlabel("iteration")

ax2[1, 0].semilogy(its, gn_min)
ax2[1, 0].set_title(r"$\min_{n\leq k}\|\nabla L(\theta_n)\|_2$ (semilog)")
ax2[1, 0].set_xlabel("iteration")

ax2[1, 1].loglog(its1, gn_min[1:])
ax2[1, 1].set_title(r"$\min_{n\leq k}\|\nabla L(\theta_n)\|_2$ (log-log)")
ax2[1, 1].set_xlabel("iteration")
plt.tight_layout()

# ---- Figure 3: 條件數 ----
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.semilogy(cond_iters, cond_hist)
ax3.set_title(r"cond$(J^TJ)=\lambda_{max}/\lambda_{min}$ (semilog)")
ax3.set_xlabel("iteration")
plt.tight_layout()

plt.show()
