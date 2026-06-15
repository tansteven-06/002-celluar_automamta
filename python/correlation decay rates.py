import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Setup
# ============================================================
otimes = np.array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]])

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

def get_d_value(f):
    from itertools import product
    def F_index(sigma_i, r_i, sigma_next, r_next):
        idx = 0
        for val in [sigma_i, r_i, sigma_next, r_next]:
            idx = idx * 3 + (val + 1)
        return idx
    
    states = [-1, 0, 1]
    L = 3
    n_configs = 3**(2*L)
    M = np.zeros((n_configs, 81), dtype=float)
    row = 0
    
    for config in product(states, repeat=2*L):
        sigmas = list(config[0::2])
        rs = list(config[1::2])
        
        sigmas_new = [0]*L
        rs_new = [0]*L
        for i in range(L):
            s_left = sigmas[(i-1)%L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1)%L]
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i])
        
        for i in range(L):
            i_next = (i+1)%L
            M[row, F_index(sigmas_new[i], rs_new[i], sigmas_new[i_next], rs_new[i_next])] += 1
            M[row, F_index(sigmas[i], rs[i], sigmas[i_next], rs[i_next])] -= 1
        row += 1
    
    rank = np.linalg.matrix_rank(M)
    return 81 - rank

def compute_correlation(f, L, n_samples=20, max_steps=50):
    """Compute time correlation function C(t)"""
    states = [-1, 0, 1]
    all_corr = []
    
    for _ in range(n_samples):
        config = list(np.random.choice(states, 2*L))
        sigmas_0 = np.array([config[2*i] for i in range(L)])
        mean_0 = np.mean(sigmas_0)
        std_0 = np.std(sigmas_0)
        
        if std_0 < 1e-10:
            continue
        
        sigmas_0_norm = (sigmas_0 - mean_0) / std_0
        
        corr_t = [1.0]  # C(0) = 1
        
        for t in range(1, max_steps+1):
            sigmas = config[0::2]
            rs = config[1::2]
            new_config = [0] * (2*L)
            for i in range(L):
                s_left = sigmas[(i-1) % L]
                s_self = sigmas[i]
                s_right = sigmas[(i+1) % L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                new_config[2*i] = new_s
                new_config[2*i+1] = new_r
            config = new_config
            
            sigmas_t = np.array([config[2*i] for i in range(L)])
            mean_t = np.mean(sigmas_t)
            std_t = np.std(sigmas_t)
            
            if std_t > 1e-10:
                sigmas_t_norm = (sigmas_t - mean_t) / std_t
                corr = np.mean(sigmas_0_norm * sigmas_t_norm)
            else:
                corr = 0
            corr_t.append(corr)
        
        all_corr.append(corr_t)
    
    return np.mean(all_corr, axis=0)

def fit_decay_rate(corr, t_max=20):
    """Fit C(t) = exp(-γ*t) for early times"""
    t = np.arange(t_max+1)
    c = corr[:t_max+1]
    
    # Only use positive values for log fit
    valid = c > 0.01
    if np.sum(valid) < 5:
        return -1
    
    t_valid = t[valid]
    c_valid = c[valid]
    
    coeffs = np.polyfit(t_valid, np.log(c_valid), 1)
    return -coeffs[0]  # γ

# ============================================================
# Find d=9 rules
# ============================================================

print("Finding d=9 rules...")
np.random.seed(42)

d9_rules = []
d9_f_arrays = []
attempts = 0

while len(d9_rules) < 15 and attempts < 5000:
    f_vec = np.random.choice([-1, 0, 1], 27)
    f_arr = f_to_array(f_vec)
    d = get_d_value(f_arr)
    
    if d == 9:
        d9_rules.append(f_vec)
        d9_f_arrays.append(f_arr)
        print(f"  Found #{len(d9_rules)} (attempt {attempts+1})")
    
    attempts += 1

print(f"\nFound {len(d9_rules)} d=9 rules")

# ============================================================
# Compute correlation decay rates
# ============================================================

L = 8
print(f"\nComputing correlation decay rates (L={L})...")

all_gamma = []
all_correlations = []

for idx, f_arr in enumerate(d9_f_arrays):
    print(f"  Rule {idx+1}/{len(d9_f_arrays)}...")
    corr = compute_correlation(f_arr, L, n_samples=15, max_steps=40)
    gamma = fit_decay_rate(corr, t_max=15)
    all_gamma.append(gamma)
    all_correlations.append(corr)
    print(f"    γ = {gamma:.4f}")

all_gamma = np.array(all_gamma)

# Remove failed fits
valid_gamma = all_gamma > 0
gamma_valid = all_gamma[valid_gamma]

print(f"\nValid fits: {np.sum(valid_gamma)}/{len(all_gamma)}")

# ============================================================
# Results
# ============================================================

print(f"\n{'='*70}")
print(f"RESULTS: Decay rate γ for {len(gamma_valid)} d=9 rules")
print(f"{'='*70}")
print(f"  γ values: {gamma_valid}")
print(f"  Mean γ = {np.mean(gamma_valid):.4f}")
print(f"  Std γ  = {np.std(gamma_valid):.4f}")
print(f"  Median γ = {np.median(gamma_valid):.4f}")
print(f"  CV = {np.std(gamma_valid)/np.mean(gamma_valid):.4f}")

# ============================================================
# Plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. All correlation functions
t = np.arange(41)
for idx, corr in enumerate(all_correlations):
    if idx < 10:  # Show first 10
        ax1.plot(t, np.abs(corr), '-', linewidth=1, alpha=0.7, 
                 label=f'γ={all_gamma[idx]:.3f}' if all_gamma[idx] > 0 else 'failed')

ax1.set_yscale('log')
ax1.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax1.set_ylabel('|C(t)|', fontsize=12, fontweight='bold')
ax1.set_title(f'Correlation Decay ({len(d9_rules)} d=9 rules)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='lower left')
ax1.grid(True, alpha=0.3)

# 2. Histogram of γ
ax2.hist(gamma_valid, bins=8, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(x=np.mean(gamma_valid), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(gamma_valid):.4f}')
ax2.axvline(x=np.median(gamma_valid), color='green', linestyle=':', linewidth=2, 
            label=f'Median = {np.median(gamma_valid):.4f}')
ax2.set_xlabel('Decay Rate γ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
ax2.set_title(f'Distribution of γ\nCV = {np.std(gamma_valid)/np.mean(gamma_valid):.3f}', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Universality of Correlation Decay in d=9 Rules', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_decay_universality.png', dpi=150, bbox_inches='tight')
print("\nSaved: correlation_decay_universality.png")

# ============================================================
# Conclusion
# ============================================================

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")

cv = np.std(gamma_valid) / np.mean(gamma_valid)

if cv < 0.3:
    print(f"\n✓ γ is UNIVERSAL across d=9 rules!")
    print(f"  γ = {np.mean(gamma_valid):.4f} ± {np.std(gamma_valid):.4f}")
else:
    print(f"\nγ shows significant variation (CV = {cv:.3f})")
    print(f"  NOT universal across d=9 rules.")

plt.show()