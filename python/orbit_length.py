import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter

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

def config_to_idx(config, L):
    idx = 0
    for val in config:
        idx = idx * 3 + (val + 1)
    return idx

def compute_orbit_lengths(f, L):
    states = [-1, 0, 1]
    n_states = 3**(2*L)
    visited = np.zeros(n_states, dtype=bool)
    orbit_lengths = []
    
    for config in product(states, repeat=2*L):
        idx = config_to_idx(config, L)
        if visited[idx]:
            continue
        
        current = config
        length = 0
        while not visited[config_to_idx(current, L)]:
            visited[config_to_idx(current, L)] = True
            length += 1
            
            sigmas = list(current[0::2])
            rs = list(current[1::2])
            new_config = []
            for i in range(L):
                s_left = sigmas[(i-1) % L]
                s_self = sigmas[i]
                s_right = sigmas[(i+1) % L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                new_config.append(new_s)
                new_config.append(new_r)
            current = tuple(new_config)
            
            if length > 50000:
                break
        
        orbit_lengths.append(length)
    
    return np.array(orbit_lengths)

def get_d_value(f):
    """Compute d-value for L=3"""
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

def fit_alpha(orbit_lengths, T_min=3):
    """Fit P(T) ~ T^{-alpha}"""
    counter = Counter(orbit_lengths)
    T_vals = np.array(sorted([t for t in counter.keys() if t >= T_min]))
    counts = np.array([counter[t] for t in T_vals])
    P = counts / np.sum(counts)
    
    valid = counts > 0
    log_T = np.log(T_vals[valid])
    log_P = np.log(P[valid])
    
    coeffs = np.polyfit(log_T, log_P, 1)
    return -coeffs[0]

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
# Compute orbit lengths for each rule (L=4)
# ============================================================

print(f"\nComputing orbit lengths (L=4)...")
all_alphas = []

for idx, f_arr in enumerate(d9_f_arrays):
    print(f"  Rule {idx+1}/{len(d9_f_arrays)}...")
    orbit_lengths = compute_orbit_lengths(f_arr, 4)
    alpha = fit_alpha(orbit_lengths)
    all_alphas.append(alpha)
    print(f"    Orbits: {len(orbit_lengths)}, α = {alpha:.4f}")

all_alphas = np.array(all_alphas)

# ============================================================
# Results
# ============================================================

print(f"\n{'='*70}")
print(f"RESULTS: α for {len(all_alphas)} d=9 rules")
print(f"{'='*70}")
print(f"  α values: {all_alphas}")
print(f"  Mean α = {np.mean(all_alphas):.4f}")
print(f"  Std α  = {np.std(all_alphas):.4f}")
print(f"  Median α = {np.median(all_alphas):.4f}")
print(f"  Min α = {np.min(all_alphas):.4f}")
print(f"  Max α = {np.max(all_alphas):.4f}")

# ============================================================
# Plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Histogram of α values
ax1.hist(all_alphas, bins=8, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(x=np.mean(all_alphas), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(all_alphas):.4f}')
ax1.axvline(x=np.median(all_alphas), color='green', linestyle=':', linewidth=2, 
            label=f'Median = {np.median(all_alphas):.4f}')
ax1.set_xlabel('α', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Rules', fontsize=13, fontweight='bold')
ax1.set_title(f'Distribution of α across {len(all_alphas)} d=9 Rules', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Overlay all power laws (normalized)
ax2.set_xlabel('Orbit Length T', fontsize=13, fontweight='bold')
ax2.set_ylabel('P(T) [arbitrary units, shifted for clarity]', fontsize=13, fontweight='bold')
ax2.set_title('Orbit Length Distributions (Shifted)', fontsize=13, fontweight='bold')

T_range = np.logspace(0, 3, 100)
for idx, (f_arr, alpha) in enumerate(zip(d9_f_arrays, all_alphas)):
    orbit_lengths = compute_orbit_lengths(f_arr, 4)
    counter = Counter(orbit_lengths)
    T_vals = np.array(sorted([t for t in counter.keys() if t >= 3]))
    counts = np.array([counter[t] for t in T_vals])
    P = counts / np.sum(counts)
    
    # Shift for visibility
    shift = idx * 0.5
    ax2.loglog(T_vals, P + shift, '-', linewidth=1, alpha=0.7, 
               label=f'α={alpha:.3f}' if idx < 5 else '')

ax2.grid(True, alpha=0.3)

plt.suptitle(f'Universality of Orbit Length Distribution in d=9 Rules\n'
             f'α = {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('alpha_universality.png', dpi=150, bbox_inches='tight')
print("\nSaved: alpha_universality.png")

# ============================================================
# Conclusion
# ============================================================

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")

cv = np.std(all_alphas) / np.mean(all_alphas)
print(f"\nCoefficient of variation: {cv:.4f}")

if cv < 0.2:
    print(f"\n✓ α is UNIVERSAL across d=9 rules!")
    print(f"  α = {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}")
    print(f"  This is a new universal exponent for ternary ERCA.")
else:
    print(f"\nα shows significant variation across rules.")
    print(f"  Not universal, but possibly correlated with other quantities.")

plt.show()