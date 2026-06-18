import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Fixed otimes (Table #1)
# ============================================================
otimes = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

def state_to_idx(val):
    return val + 1

def apply_otimes(a, b):
    return otimes[state_to_idx(a), state_to_idx(b)]

def f_to_array(f_values):
    f = np.zeros((3, 3, 3), dtype=int)
    for idx, val in enumerate(f_values):
        i = idx // 9
        j = (idx % 9) // 3
        k = idx % 3
        f[i, j, k] = val
    return f

def apply_rule(f, s_left, s_self, s_right, r):
    f_out = f[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
    sigma_new = apply_otimes(f_out, r)
    r_new = s_self
    return sigma_new, r_new

def F_index(sigma_i, r_i, sigma_next, r_next):
    idx = 0
    for val in [sigma_i, r_i, sigma_next, r_next]:
        idx = idx * 3 + state_to_idx(val)
    return idx

def build_constraint_matrix_L(f, L):
    states = [-1, 0, 1]
    n_configs = 3**(2*L)
    M = np.zeros((n_configs, 81), dtype=float)
    row = 0
    for config in product(states, repeat=2*L):
        sigmas = list(config[0::2])
        rs = list(config[1::2])
        sigmas_new = [0] * L
        rs_new = [0] * L
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i])
        for i in range(L):
            i_next = (i+1) % L
            M[row, F_index(sigmas_new[i], rs_new[i], 
                          sigmas_new[i_next], rs_new[i_next])] += 1
            M[row, F_index(sigmas[i], rs[i], 
                          sigmas[i_next], rs[i_next])] -= 1
        row += 1
    return M

def get_d_value(f, L=3):
    M = build_constraint_matrix_L(f, L)
    rank = np.linalg.matrix_rank(M)
    return 81 - rank

def compute_lyapunov(f, L, n_trials=10, n_steps=30):
    states = [-1, 0, 1]
    lyap_values = []
    for _ in range(n_trials):
        config1 = list(np.random.choice(states, 2*L))
        config2 = config1.copy()
        site = np.random.randint(0, 2*L)
        old_val = config2[site]
        new_vals = [v for v in states if v != old_val]
        config2[site] = np.random.choice(new_vals)
        distances = []
        for step in range(n_steps):
            d = sum(1 for i in range(2*L) if config1[i] != config2[i])
            distances.append(d)
            if d == 0:
                break
            for cfg in [config1, config2]:
                sigmas = cfg[0::2]
                rs = cfg[1::2]
                new_cfg = [0] * (2*L)
                for i in range(L):
                    s_left = sigmas[(i-1) % L]
                    s_self = sigmas[i]
                    s_right = sigmas[(i+1) % L]
                    new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                    new_cfg[2*i] = new_s
                    new_cfg[2*i+1] = new_r
                cfg[:] = new_cfg
        distances = np.array(distances)
        valid = (distances > 0) & (distances < 2*L*0.8)
        if np.sum(valid) >= 3:
            t = np.arange(len(distances))[valid]
            d = distances[valid]
            coeffs = np.polyfit(t, np.log(d + 1e-10), 1)
            lyap_values.append(coeffs[0])
        else:
            lyap_values.append(0.0)
    return np.mean(lyap_values)

# ============================================================
# Generate rules and compute Lyapunov exponents
# ============================================================

print("=" * 70)
print("LYAPUNOV EXPONENT: GAUSSIAN DISTRIBUTION")
print("=" * 70)

np.random.seed(42)
n_rules = 500
L = 6

lyap_values = []

print(f"Computing Lyapunov exponents for {n_rules} random rules (L={L})...")

for i in range(n_rules):
    f_vec = np.random.choice([-1, 0, 1], 27)
    f_arr = f_to_array(f_vec)
    lyap = compute_lyapunov(f_arr, L, n_trials=8, n_steps=25)
    lyap_values.append(lyap)
    if (i+1) % 100 == 0:
        print(f"  Progress: {i+1}/{n_rules}")

lyap_values = np.array(lyap_values)
mu = np.mean(lyap_values)
sigma = np.std(lyap_values)

print(f"\nResults:")
print(f"  Mean: {mu:.6f}")
print(f"  Std:  {sigma:.6f}")
print(f"  Fraction λ > 0: {np.sum(lyap_values > 0.01)/n_rules*100:.1f}%")

# ============================================================
# Plot: Gaussian fit
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Histogram
bins = np.linspace(mu - 4*sigma, mu + 4*sigma, 40)
ax.hist(lyap_values, bins=bins, density=True, color='steelblue', 
        edgecolor='white', alpha=0.75, label=f'Data (n={n_rules})')

# Gaussian fit
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
gaussian = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
ax.plot(x, gaussian, 'r-', linewidth=2.5, label=f'Gaussian fit\n$\mu={mu:.4f}$, $\sigma={sigma:.4f}$')

# Mean line
ax.axvline(x=mu, color='green', linestyle='--', linewidth=2, label=f'Mean = {mu:.4f}')

# Zero line
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='λ = 0')

ax.set_xlabel('Lyapunov Exponent λ', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
ax.set_title(f'Gaussian Distribution of Lyapunov Exponents\n'
             f'({n_rules} Random Rules, L={L})', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lyapunov_gaussian.png', dpi=200, bbox_inches='tight')
print("\nSaved: lyapunov_gaussian.png")
plt.show()