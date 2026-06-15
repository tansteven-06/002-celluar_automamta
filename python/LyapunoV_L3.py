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

# ============================================================
# Constraint matrix for d-value
# ============================================================

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

# ============================================================
# Lyapunov exponent calculation
# ============================================================

def compute_lyapunov(f, L, n_trials=10, n_steps=30):
    """
    Compute finite-time Lyapunov exponent by tracking
    how a small perturbation grows.
    """
    states = [-1, 0, 1]
    lyap_values = []
    
    for _ in range(n_trials):
        # Random initial config
        config1 = list(np.random.choice(states, 2*L))
        
        # Create perturbed config (flip one site)
        config2 = config1.copy()
        site = np.random.randint(0, 2*L)
        old_val = config2[site]
        new_vals = [v for v in states if v != old_val]
        config2[site] = np.random.choice(new_vals)
        
        distances = []
        
        for step in range(n_steps):
            # Compute Hamming distance
            d = sum(1 for i in range(2*L) if config1[i] != config2[i])
            distances.append(d)
            
            if d == 0:
                break
            
            # Evolve config1
            sigmas1 = config1[0::2]
            rs1 = config1[1::2]
            new1 = [0] * (2*L)
            for i in range(L):
                s_left = sigmas1[(i-1) % L]
                s_self = sigmas1[i]
                s_right = sigmas1[(i+1) % L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs1[i])
                new1[2*i] = new_s
                new1[2*i+1] = new_r
            config1 = new1
            
            # Evolve config2
            sigmas2 = config2[0::2]
            rs2 = config2[1::2]
            new2 = [0] * (2*L)
            for i in range(L):
                s_left = sigmas2[(i-1) % L]
                s_self = sigmas2[i]
                s_right = sigmas2[(i+1) % L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs2[i])
                new2[2*i] = new_s
                new2[2*i+1] = new_r
            config2 = new2
        
        # Fit exponential growth: d(t) = d0 * exp(lambda * t)
        distances = np.array(distances)
        valid = (distances > 0) & (distances < 2*L*0.8)  # Before saturation
        
        if np.sum(valid) >= 3:
            t = np.arange(len(distances))[valid]
            d = distances[valid]
            coeffs = np.polyfit(t, np.log(d + 1e-10), 1)
            lyap = coeffs[0]
        else:
            lyap = 0.0
        
        lyap_values.append(lyap)
    
    return np.mean(lyap_values)

# ============================================================
# Generate rules and compute Lyapunov exponents
# ============================================================

print("=" * 70)
print("LYAPUNOV EXPONENT DISTRIBUTION IN RULE SPACE")
print("=" * 70)

np.random.seed(42)
n_rules = 500
L = 6

results = []
print(f"\nComputing Lyapunov exponents for {n_rules} random rules (L={L})...")

for i in range(n_rules):
    f_vec = np.random.choice([-1, 0, 1], 27)
    f_arr = f_to_array(f_vec)
    d = get_d_value(f_arr, L=3)
    lyap = compute_lyapunov(f_arr, L, n_trials=8, n_steps=25)
    
    results.append({
        'd': d,
        'lyap': lyap,
        'f_vec': f_vec.copy()
    })
    
    if (i+1) % 50 == 0:
        print(f"  Progress: {i+1}/{n_rules}")

# ============================================================
# Extract data
# ============================================================

d_values = np.array([r['d'] for r in results])
lyap_values = np.array([r['lyap'] for r in results])

# Separate by d-value
d9_mask = d_values == 9
d_gt9_mask = d_values > 9

lyap_d9 = lyap_values[d9_mask]
lyap_dgt9 = lyap_values[d_gt9_mask]

print(f"\nResults:")
print(f"  d=9 rules: {np.sum(d9_mask)} ({np.sum(d9_mask)/n_rules*100:.1f}%)")
print(f"  d>9 rules: {np.sum(d_gt9_mask)} ({np.sum(d_gt9_mask)/n_rules*100:.1f}%)")
print(f"  Mean Lyapunov (d=9): {np.mean(lyap_d9):.4f} ± {np.std(lyap_d9):.4f}")
print(f"  Mean Lyapunov (d>9): {np.mean(lyap_dgt9):.4f} ± {np.std(lyap_dgt9):.4f}")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Histogram of all Lyapunov exponents
ax1 = axes[0, 0]
bins = np.linspace(-0.2, 0.4, 50)
ax1.hist(lyap_values, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='λ=0')
ax1.axvline(x=np.mean(lyap_values), color='green', linestyle='-', linewidth=2, 
            label=f'Mean = {np.mean(lyap_values):.4f}')
ax1.set_xlabel('Lyapunov Exponent λ', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
ax1.set_title(f'Lyapunov Exponent Distribution\n({n_rules} random rules, L={L})', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# 2. d=9 vs d>9 comparison
ax2 = axes[0, 1]
bins = np.linspace(-0.2, 0.4, 40)
ax2.hist(lyap_d9, bins=bins, alpha=0.7, color='#2196F3', label=f'd=9 (n={len(lyap_d9)})', edgecolor='white')
ax2.hist(lyap_dgt9, bins=bins, alpha=0.7, color='#E91E63', label=f'd>9 (n={len(lyap_dgt9)})', edgecolor='white')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Lyapunov Exponent λ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
ax2.set_title('Lyapunov Exponent: d=9 vs d>9', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Lyapunov vs d-value scatter
ax3 = axes[1, 0]
d_unique = sorted(set(d_values))
for d_val in d_unique:
    mask = d_values == d_val
    lyap_d = lyap_values[mask]
    if len(lyap_d) > 0:
        ax3.scatter([d_val] * len(lyap_d), lyap_d, alpha=0.3, s=15, 
                   color='steelblue', edgecolors='none')
        ax3.plot(d_val, np.mean(lyap_d), 'ro', markersize=8, markeredgecolor='darkred')

ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax3.set_xlabel('d (nullspace dimension)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Lyapunov Exponent λ', fontsize=12, fontweight='bold')
ax3.set_title('Lyapunov Exponent vs d-value', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Cumulative distribution
ax4 = axes[1, 1]
sorted_lyap = np.sort(lyap_values)
cdf = np.arange(1, len(sorted_lyap)+1) / len(sorted_lyap)
ax4.plot(sorted_lyap, cdf, 'b-', linewidth=2, label='All rules')
ax4.plot(np.sort(lyap_d9), np.arange(1, len(lyap_d9)+1)/len(lyap_d9), 
         'g-', linewidth=2, label=f'd=9')
if len(lyap_dgt9) > 0:
    ax4.plot(np.sort(lyap_dgt9), np.arange(1, len(lyap_dgt9)+1)/len(lyap_dgt9), 
             'r-', linewidth=2, label=f'd>9')
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax4.set_xlabel('Lyapunov Exponent λ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
ax4.set_title('Cumulative Distribution of Lyapunov Exponents', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Lyapunov Exponent Analysis in Ternary ERCA Rule Space', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lyapunov_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: lyapunov_distribution.png")

# ============================================================
# Statistical analysis
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

print(f"\nAll rules (n={len(lyap_values)}):")
print(f"  Mean: {np.mean(lyap_values):.6f}")
print(f"  Median: {np.median(lyap_values):.6f}")
print(f"  Std: {np.std(lyap_values):.6f}")
print(f"  Skewness: {((lyap_values - np.mean(lyap_values))**3).mean() / np.std(lyap_values)**3:.4f}")
print(f"  Fraction λ > 0: {np.sum(lyap_values > 0.01)/len(lyap_values)*100:.1f}%")
print(f"  Fraction λ ≈ 0: {np.sum(np.abs(lyap_values) < 0.01)/len(lyap_values)*100:.1f}%")

# Test for universality: fit to normal distribution
from scipy import stats
if 'stats' in dir():
    ks_stat, ks_p = stats.kstest(lyap_values, 'norm', args=(np.mean(lyap_values), np.std(lyap_values)))
    print(f"\nKolmogorov-Smirnov test for normality:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_p:.4f}")
    if ks_p > 0.05:
        print(f"  → Distribution is consistent with Gaussian")
    else:
        print(f"  → Distribution differs significantly from Gaussian")

plt.show()