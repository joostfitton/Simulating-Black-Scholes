import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths, seed=13):
    """Euler-Maruyama scheme for the 3D SDE system:"""
    np.random.seed(seed)
    n_steps = int(T / dt)
    S_paths = np.full((n_paths, n_steps + 1), S0)
    sigma_paths = np.full((n_paths, n_steps + 1), sigma0)
    xi_paths = np.full((n_paths, n_steps + 1), xi0)
    sqrt_dt = np.sqrt(dt)

    for n in range(n_steps):
        dW1 = sqrt_dt * np.random.randn(n_paths)
        dW2 = sqrt_dt * np.random.randn(n_paths)

        S_paths[:, n+1] = (
            S_paths[:, n]
            + mu * S_paths[:, n] * dt
            + sigma_paths[:, n] * S_paths[:, n] * dW1
        )
        sigma_paths[:, n+1] = (
            sigma_paths[:, n]
            - (sigma_paths[:, n] - xi_paths[:, n]) * dt
            + p * sigma_paths[:, n] * dW2
        )
        xi_paths[:, n+1] = (
            xi_paths[:, n]
            + (1.0 / alpha) * (sigma_paths[:, n] - xi_paths[:, n]) * dt
        )

    return S_paths, sigma_paths, xi_paths

def milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths, seed=13):
    """ Milstein scheme for the 3D SDE system"""
    np.random.seed(seed)
    n_steps = int(T / dt)
    S_paths = np.full((n_paths, n_steps + 1), S0)
    sigma_paths = np.full((n_paths, n_steps + 1), sigma0)
    xi_paths = np.full((n_paths, n_steps + 1), xi0)
    sqrt_dt = np.sqrt(dt)

    for n in range(n_steps):
        dW1 = sqrt_dt * np.random.randn(n_paths)
        dW2 = sqrt_dt * np.random.randn(n_paths)

        S_paths[:, n+1] = (
            S_paths[:, n]
            + mu * S_paths[:, n] * dt
            + sigma_paths[:, n] * S_paths[:, n] * dW1
            + 0.5 * (sigma_paths[:, n]**2) * S_paths[:, n] * (dW1**2 - dt)
        )
        sigma_paths[:, n+1] = (
            sigma_paths[:, n]
            - (sigma_paths[:, n] - xi_paths[:, n]) * dt
            + p * sigma_paths[:, n] * dW2
            + 0.5 * (p**2) * sigma_paths[:, n] * (dW2**2 - dt)
        )
        xi_paths[:, n+1] = (
            xi_paths[:, n]
            + (1.0 / alpha) * (sigma_paths[:, n] - xi_paths[:, n]) * dt
        )

    return S_paths, sigma_paths, xi_paths

# Parameters
S0, sigma0, xi0 = 50.0, 0.20, 0.20
mu = 0.10
p_values = [0.25, 1.25, 2.25]
alpha_values = [0.02, 5, 20]
T, dt, n_paths = 1.0, 0.001, 1000

t_grid = np.linspace(0, T, int(T / dt) + 1)
alpha_p_combinations = [(p, alpha) for p in p_values for alpha in alpha_values]
titles = [f"p={p}, α={alpha}" for p, alpha in alpha_p_combinations]

#PLOTS FOR THE EULER–MARUYAMA METHOD

# Plot Stock Price S(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    S_paths, _, _ = euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(S_paths, axis=0), color='blue')
    ax.set_title(titles[idx])
    ax.set_ylabel("S(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Euler–Maruyama: Mean Stock Price S(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot Volatility σ(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, _ = euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(sigma_paths, axis=0), color='red')
    ax.set_title(titles[idx])
    ax.set_ylabel("σ(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Euler–Maruyama: Mean Volatility σ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot Long-Term Volatility ξ(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, _, xi_paths = euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(xi_paths, axis=0), color='green')
    ax.set_title(titles[idx])
    ax.set_ylabel("ξ(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Euler–Maruyama: Mean Long-Term Volatility ξ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot σ(t) and ξ(t) Together
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, xi_paths = euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(sigma_paths, axis=0), color='red', label='σ(t)')
    ax.plot(t_grid, np.mean(xi_paths, axis=0), color='green', linestyle='dashed', label='ξ(t)')
    ax.set_title(titles[idx])
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Euler–Maruyama: Mean σ(t) & ξ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot |ξ(t) - σ(t)|
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, xi_paths = euler_maruyama(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    diff = np.mean(xi_paths, axis=0) - np.mean(sigma_paths, axis=0)
    ax.plot(t_grid, diff, color='purple')
    ax.set_title(titles[idx])
    ax.set_ylabel("|ξ(t) - σ(t)|")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Euler–Maruyama: Mean |ξ(t) - σ(t)|", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#PLOTS FOR THE MILSTEIN METHOD

# Plot Stock Price S(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    S_paths, _, _ = milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(S_paths, axis=0), color='blue')
    ax.set_title(titles[idx])
    ax.set_ylabel("S(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Milstein: Mean Stock Price S(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot Volatility σ(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, _ = milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(sigma_paths, axis=0), color='red')
    ax.set_title(titles[idx])
    ax.set_ylabel("σ(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Milstein: Mean Volatility σ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot Long-Term Volatility ξ(t)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, _, xi_paths = milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(xi_paths, axis=0), color='green')
    ax.set_title(titles[idx])
    ax.set_ylabel("ξ(t)")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Milstein: Mean Long-Term Volatility ξ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot σ(t) and ξ(t) Together
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, xi_paths = milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    ax.plot(t_grid, np.mean(sigma_paths, axis=0), color='red', label='σ(t)')
    ax.plot(t_grid, np.mean(xi_paths, axis=0), color='green', linestyle='dashed', label='ξ(t)')
    ax.set_title(titles[idx])
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Milstein: Mean σ(t) & ξ(t)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot |ξ(t) - σ(t)|
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
for idx, (p, alpha) in enumerate(alpha_p_combinations):
    ax = axes[idx // 3][idx % 3]
    _, sigma_paths, xi_paths = milstein(S0, sigma0, xi0, mu, p, alpha, T, dt, n_paths)
    diff = np.mean(xi_paths, axis=0) - np.mean(sigma_paths, axis=0)
    ax.plot(t_grid, diff, color='purple')
    ax.set_title(titles[idx])
    ax.set_ylabel("|ξ(t) - σ(t)|")
    ax.grid(True)
axes[2, 1].set_xlabel("Time (Years)")
fig.suptitle("Milstein: Mean |ξ(t) - σ(t)|", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

def exact_solution_bs(S0, mu, sigma0, T, W_T):
    """ Exact solution S(T) for the standard Black–Scholes model"""
    return S0 * np.exp((mu - 0.5 * sigma0**2) * T + sigma0 * W_T)

def euler_scheme_bs(S0, mu, sigma0, T, dt, dW):
    """Euler scheme for the Black–Scholes SDE:"""
    N, M = dW.shape
    S = np.full(N, S0)
    for n in range(M):
        S += mu * S * dt + sigma0 * S * dW[:, n]
    return S

def milstein_scheme_bs(S0, mu, sigma0, T, dt, dW):
    """
    Milstein scheme for the Black–Scholes SDE"""
    N, M = dW.shape
    S = np.full(N, S0)
    for n in range(M):
        S_old = S
        # Basic increment
        S += mu * S_old * dt + sigma0 * S_old * dW[:, n]
        # Milstein correction
        S += 0.5 * (sigma0**2) * S_old * (dW[:, n]**2 - dt)
    return S

def strong_convergence_bs():
    """
    Strong convergence test for the Black–Scholes SDE using the exact
    solution as the reference.
    """
    #Model parameters
    S0 = 50.0
    mu = 0.10
    sigma0 = 0.20
    T = 1.0
    n_paths = 100_000

    #Fine grid for reference
    dt_fine = 1/2048
    M_fine = int(T / dt_fine)

    np.random.seed(13)
    # dW_fine has shape (n_paths, M_fine)
    dW_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))

    # For each path, the total Brownian increment W_T is sum(dW_fine along axis=1)
    W_T = dW_fine.sum(axis=1)

    #Exact solution at T
    S_exact = exact_solution_bs(S0, mu, sigma0, T, W_T)

    #Convergence grids
    dt_list = [1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]
    euler_errors = []
    milstein_errors = []

    for dt in dt_list:
        M = int(T / dt)          
        factor = int(dt / dt_fine)

        # Sum up blocks of dW_fine to get coarse increments
        dW_coarse = dW_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)

        # Euler on the coarse grid
        S_eul = euler_scheme_bs(S0, mu, sigma0, T, dt, dW_coarse)
        # Milstein on the coarse grid
        S_mil = milstein_scheme_bs(S0, mu, sigma0, T, dt, dW_coarse)

        # Strong errors: L^2 error vs exact solution
        err_eul = np.sqrt(np.mean((S_eul - S_exact)**2))
        err_mil = np.sqrt(np.mean((S_mil - S_exact)**2))

        euler_errors.append(err_eul)
        milstein_errors.append(err_mil)

    #Plot on log-log scale
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_list, euler_errors, 'o--', label="Euler")
    plt.loglog(dt_list, milstein_errors, 's--', label="Milstein")
    plt.xlabel("Δt")
    plt.ylabel("Strong Error in S(T)")
    plt.title("Strong Convergence (Exact Black–Scholes)")
    plt.grid(True, which='both')
    plt.legend()

    #Estimate slopes via linear regression in log-log space
    slope_eul, _ = np.polyfit(np.log(dt_list), np.log(euler_errors), 1)
    slope_mil, _ = np.polyfit(np.log(dt_list), np.log(milstein_errors), 1)
    print(f"Strong Convergence Slopes:\n"
          f"  Euler   ≈ {slope_eul:.2f}\n"
          f"  Milstein≈ {slope_mil:.2f}")

    plt.show()

def weak_convergence_bs():
    """
    Weak convergence test for the Black–Scholes SDE using the exact solution
    as the reference. We compare statistical functionals (e.g., variance, mean)
    of the numerical approximations vs. those of the exact solution.
    """
    #Model parameters
    S0 = 50.0
    mu = 0.10
    sigma0 = 0.20
    T = 1.0
    n_paths = 100_000

    # Fine grid for reference
    dt_fine = 1/1024
    M_fine = int(T / dt_fine)

    np.random.seed(13)
    dW_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))
    W_T = dW_fine.sum(axis=1)

    # Exact solution
    S_exact = exact_solution_bs(S0, mu, sigma0, T, W_T)
    mean_ref = np.mean(S_exact)
    var_ref = np.var(S_exact)

    dt_list = [1/8, 1/16, 1/32, 1/64, 1/128]

    euler_mean_err = []
    euler_var_err = []
    milstein_mean_err = []
    milstein_var_err = []

    for dt in dt_list:
        M = int(T / dt)
        factor = int(dt / dt_fine)

        dW_coarse = dW_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)

        # Euler and Milstein on the coarse grid
        S_eul = euler_scheme_bs(S0, mu, sigma0, T, dt, dW_coarse)
        S_mil = milstein_scheme_bs(S0, mu, sigma0, T, dt, dW_coarse)

        # Compute mean and variance
        mean_eul = np.mean(S_eul)
        var_eul = np.var(S_eul)
        mean_mil = np.mean(S_mil)
        var_mil = np.var(S_mil)

        # Weak errors (compare mean and variance to exact)
        euler_mean_err.append(abs(mean_eul - mean_ref))
        euler_var_err.append(abs(var_eul - var_ref))
        milstein_mean_err.append(abs(mean_mil - mean_ref))
        milstein_var_err.append(abs(var_mil - var_ref))

    #Plot variance convergence on log-log scale
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_list, euler_var_err, 'o--', label="Euler (variance)")
    plt.loglog(dt_list, milstein_var_err, 's--', label="Milstein (variance)")
    plt.xlabel("Δt")
    plt.ylabel("|Var[S(T)] - Var[S_exact(T)]|")
    plt.title("Weak Convergence (Variance) - Exact Black–Scholes")
    plt.grid(True, which='both')
    plt.legend()

    # Slopes in log-log scale
    slope_eul_var, _ = np.polyfit(np.log(dt_list), np.log(euler_var_err), 1)
    slope_mil_var, _ = np.polyfit(np.log(dt_list), np.log(milstein_var_err), 1)
    print(f"Weak Convergence (Variance) Slopes:\n"
          f"  Euler   ≈ {slope_eul_var:.2f}\n"
          f"  Milstein≈ {slope_mil_var:.2f}")

    plt.show()

#Run both experiments using the exact Black–Scholes solution as reference
strong_convergence_bs()
weak_convergence_bs()

def euler_scheme_system(S0, sigma0, xi0, mu, alpha, p, T, dt, dW1, dW2):
    """Euler-Maruyama scheme for the 3D system:"""
    n_paths, M = dW1.shape
    S = np.full(n_paths, S0)
    sigma = np.full(n_paths, sigma0)
    xi = np.full(n_paths, xi0)

    for n in range(M):
        dW1_n = dW1[:, n]
        dW2_n = dW2[:, n]

        S_old = S
        sigma_old = sigma
        xi_old = xi

        # Euler updates
        S = S_old + mu * S_old * dt + sigma_old * S_old * dW1_n
        sigma = sigma_old - (sigma_old - xi_old)*dt + p * sigma_old * dW2_n
        xi = xi_old + (1.0/alpha)*(sigma_old - xi_old)*dt

    return S, sigma, xi

def milstein_scheme_system(S0, sigma0, xi0, mu, alpha, p, T, dt, dW1, dW2):
    """Milstein scheme for the 3D system (S, sigma, xi), ignoring correlation between W1 and W2:"""
    n_paths, M = dW1.shape
    S = np.full(n_paths, S0)
    sigma = np.full(n_paths, sigma0)
    xi = np.full(n_paths, xi0)

    for n in range(M):
        dW1_n = dW1[:, n]
        dW2_n = dW2[:, n]

        S_old = S
        sigma_old = sigma
        xi_old = xi

        # Milstein increment for S
        S = (
            S_old
            + mu * S_old * dt
            + sigma_old * S_old * dW1_n
            + 0.5 * (sigma_old**2) * S_old * (dW1_n**2 - dt)
        )
        # Milstein increment for sigma
        sigma = (
            sigma_old
            - (sigma_old - xi_old)*dt
            + p * sigma_old * dW2_n
            + 0.5 * (p**2) * sigma_old * (dW2_n**2 - dt)
        )
        # Deterministic update for xi
        xi = xi_old + (1.0/alpha)*(sigma_old - xi_old)*dt

    return S, sigma, xi

def strong_convergence_system():
    """
    Strong convergence test for the 3D system using Milstein on a fine grid as reference.
    We only compare the S-component for simplicity.
    """
    #Model parameters
    S0, sigma0, xi0 = 50.0, 0.20, 0.20
    mu, alpha, p = 0.10, 10.0, 0.5
    T = 1.0
    n_paths = 100_000

    #Fine grid for reference (Milstein)
    dt_fine = 1/2048
    M_fine = int(T / dt_fine)

    np.random.seed(42)
    dW1_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))
    dW2_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))

    # Reference solution (S_ref) at final time T using Milstein on the fine grid
    S_ref, _, _ = milstein_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                         T, dt_fine, dW1_fine, dW2_fine)

    #Coarse step sizes for convergence study
    dt_list = [1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]
    euler_errors = []
    milstein_errors = []

    for dt in dt_list:
        M = int(T / dt)        
        factor = int(dt / dt_fine)

        # Block-sum the fine increments to get coarse increments
        dW1_coarse = dW1_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)
        dW2_coarse = dW2_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)

        # Euler on the coarse grid
        S_eul, _, _ = euler_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                          T, dt, dW1_coarse, dW2_coarse)
        # Milstein on the coarse grid
        S_mil, _, _ = milstein_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                             T, dt, dW1_coarse, dW2_coarse)

        # Strong error = RMS difference vs. the fine-grid Milstein reference
        err_eul = np.sqrt(np.mean((S_eul - S_ref)**2))
        err_mil = np.sqrt(np.mean((S_mil - S_ref)**2))

        euler_errors.append(err_eul)
        milstein_errors.append(err_mil)

    #Plot: strong errors in S(T)
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_list, euler_errors, 'o--', label="Euler (S-component)")
    plt.loglog(dt_list, milstein_errors, 's--', label="Milstein (S-component)")
    plt.xlabel("Δt")
    plt.ylabel("Strong Error in S(T)")
    plt.title("Strong Convergence (Reference = Fine Milstein)")
    plt.grid(True, which='both')
    plt.legend()

    #Slopes on log-log scale
    log_dt = np.log(dt_list)
    slope_eul, _ = np.polyfit(log_dt, np.log(euler_errors), 1)
    slope_mil, _ = np.polyfit(log_dt, np.log(milstein_errors), 1)

    print(f"Strong convergence slopes:")
    print(f"  Euler   ~ {slope_eul:.2f}")
    print(f"  Milstein~ {slope_mil:.2f}")
    plt.show()

def weak_convergence_system():
    """
    Weak convergence test for the 3D system using Milstein on a fine grid as reference.
    We compare the mean and variance of S(T).
    """
    #Model parameters
    S0, sigma0, xi0 = 50.0, 0.20, 0.20
    mu, alpha, p = 0.10, 10.0, 0.5
    T = 1.0
    n_paths = 200_000

    #Fine grid for reference
    dt_fine = 1/1024
    M_fine = int(T / dt_fine)

    np.random.seed(13)
    dW1_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))
    dW2_fine = np.random.normal(0.0, np.sqrt(dt_fine), (n_paths, M_fine))

    # Reference solution for S(T) using Milstein on fine grid
    S_ref, _, _ = milstein_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                         T, dt_fine, dW1_fine, dW2_fine)

    mean_ref = np.mean(S_ref)
    var_ref = np.var(S_ref)

    #Coarser dt list
    dt_list = [1/8, 1/16, 1/32, 1/64, 1/128]

    euler_mean_err = []
    euler_var_err = []
    milstein_mean_err = []
    milstein_var_err = []

    for dt in dt_list:
        M = int(T / dt)
        factor = int(dt / dt_fine)

        dW1_coarse = dW1_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)
        dW2_coarse = dW2_fine[:, :M*factor].reshape(n_paths, M, factor).sum(axis=2)

        # Euler on coarse grid
        S_eul, _, _ = euler_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                          T, dt, dW1_coarse, dW2_coarse)
        mean_eul = np.mean(S_eul)
        var_eul = np.var(S_eul)

        # Milstein on coarse grid
        S_mil, _, _ = milstein_scheme_system(S0, sigma0, xi0, mu, alpha, p,
                                             T, dt, dW1_coarse, dW2_coarse)
        mean_mil = np.mean(S_mil)
        var_mil = np.var(S_mil)

        # Weak errors in mean, variance vs. reference
        euler_mean_err.append(abs(mean_eul - mean_ref))
        euler_var_err.append(abs(var_eul - var_ref))
        milstein_mean_err.append(abs(mean_mil - mean_ref))
        milstein_var_err.append(abs(var_mil - var_ref))

    #Plot variance errors on log-log scale
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_list, euler_var_err, 'o--', label="Euler: variance error")
    plt.loglog(dt_list, milstein_var_err, 's--', label="Milstein: variance error")
    plt.xlabel("Δt")
    plt.ylabel("|Var[S(T)] - Var[S_ref(T)]|")
    plt.title("Weak Convergence in Variance (Reference = Fine Milstein)")
    plt.grid(True, which='both')
    plt.legend()

    # Slopes in log-log scale
    log_dt = np.log(dt_list)
    slope_eul_var, _ = np.polyfit(log_dt, np.log(euler_var_err), 1)
    slope_mil_var, _ = np.polyfit(log_dt, np.log(milstein_var_err), 1)

    print(f"Weak convergence slopes (variance):")
    print(f"  Euler   ~ {slope_eul_var:.2f}")
    print(f"  Milstein~ {slope_mil_var:.2f}")

    plt.show()

#Run the two experiments
strong_convergence_system()
weak_convergence_system()
