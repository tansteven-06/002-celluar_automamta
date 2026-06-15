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

def compute_lyapunov(f, L, n_trials=2, n_steps=20):
    """Fast Lyapunov computation with minimal trials"""
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
            
            if d >= 2*L * 0.8 or d == 0:
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
            lyap_values.append(max(coeffs[0], 0))
    
    return np.mean(lyap_values) if lyap_values else 0.0

# ============================================================
# Generate rules and test sparse L values up to 100
# ============================================================

np.random.seed(42)
n_rules = 100  # Small number of rules
L_values = [4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100,200,500,1000]

print(f"Generating {n_rules} rules...")
rules = []
for i in range(n_rules):
    f_vec = np.random.choice([-1, 0, 1], 27)
    rules.append(f_to_array(f_vec))

print(f"\nComputing Lyapunov at L = {L_values}")
print(f"({n_rules} rules, 2 trials each, {len(L_values)} L values)")
print("=" * 70)

data = np.zeros((n_rules, len(L_values)))

for i, f_arr in enumerate(rules):
    for j, L in enumerate(L_values):
        n_trials = 3 if L <= 30 else 2
        n_steps = 30 if L <= 30 else 40  # More steps for larger L
        data[i, j] = compute_lyapunov(f_arr, L, n_trials=n_trials, n_steps=n_steps)
    
    print(f"  Rule {i+1}/{n_rules}: L=4: {data[i,0]:.3f}, L=10: {data[i,3]:.3f}, L=100: {data[i,-1]:.3f}")

# ============================================================
# Results
# ============================================================

print(f"\n{'='*70}")
print(f"RESULTS: Mean Lyapunov exponent vs L")
print(f"{'='*70}")

means = np.array([np.mean(data[:, j]) for j in range(len(L_values))])
stds = np.array([np.std(data[:, j]) for j in range(len(L_values))])

for j, L in enumerate(L_values):
    print(f"  L={L:3d}: λ = {means[j]:.6f} ± {stds[j]:.6f}")

# ============================================================
# Fit: λ(L) = λ_∞ + c/L^α
# ============================================================

L_arr = np.array(L_values)

# Simple 1/L fit
L_inv = 1.0 / L_arr
coeffs_1 = np.polyfit(L_inv[1:], means[1:], 1)  # Skip L=4
lambda_inf_1 = coeffs_1[1]

print(f"\nFit λ = λ_∞ + c/L (L ≥ 6): λ_∞ = {lambda_inf_1:.6f}")

# Power-law fit: λ = λ_∞ + c/L^α
# Try different α
print(f"\nPower-law fits λ = c/L^α (assuming λ_∞ = 0):")
for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    L_pow = L_arr**(-alpha)
    coeffs = np.polyfit(L_pow, means, 1)
    residuals = means - (coeffs[0]*L_pow + coeffs[1])
    rms = np.sqrt(np.mean(residuals**2))
    print(f"  α = {alpha:.1f}: λ_∞ = {coeffs[1]:.6f}, RMS error = {rms:.6f}")

# ============================================================
# Plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. λ vs L (linear scale)
ax1.errorbar(L_arr, means, yerr=stds, fmt='ro-', linewidth=2, markersize=8, 
             capsize=5, label='Mean λ(L)')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('System Size L', fontsize=13, fontweight='bold')
ax1.set_ylabel('Lyapunov Exponent λ', fontsize=13, fontweight='bold')
ax1.set_title(f'Lyapunov Exponent vs L\n({n_rules} rules, L up to 100)', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. λ vs 1/L (linearized for extrapolation)
ax2.errorbar(L_inv, means, yerr=stds, fmt='ro', markersize=8, capsize=5, label='Data')
L_fine_inv = np.linspace(0, 0.25, 100)
ax2.plot(L_fine_inv, coeffs_1[0]*L_fine_inv + coeffs_1[1], 'b-', linewidth=2, 
         label=f'λ_∞ + c/L: λ_∞ = {lambda_inf_1:.5f}')
ax2.scatter([0], [lambda_inf_1], color='green', s=150, zorder=5, marker='*',
            label=f'λ_∞ = {lambda_inf_1:.5f}')
ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('1/L', fontsize=13, fontweight='bold')
ax2.set_ylabel('Lyapunov Exponent λ', fontsize=13, fontweight='bold')
ax2.set_title('Extrapolation to L → ∞', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Lyapunov Exponent in Thermodynamic Limit', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lyapunov_L100.png', dpi=150, bbox_inches='tight')
print("\nSaved: lyapunov_L100.png")

# ============================================================
# Conclusion
# ============================================================

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")

if lambda_inf_1 < 0.01:
    print(f"\nλ_∞ ≈ {lambda_inf_1:.6f} ≈ 0")
    print(f"The Lyapunov exponent VANISHES in the thermodynamic limit.")
    print(f"This suggests the system is NON-CHAOTIC for L → ∞,")
    print(f"or the chaos is SUBEXTENSIVE (e.g., only at boundaries).")
elif lambda_inf_1 > 0.02:
    print(f"\nλ_∞ ≈ {lambda_inf_1:.6f} > 0")
    print(f"The system remains CHAOTIC in the thermodynamic limit.")
    print(f"This is characteristic of EXTENSIVE SPATIOTEMPORAL CHAOS.")
else:
    print(f"\nλ_∞ ≈ {lambda_inf_1:.6f} (small but nonzero)")
    print(f"WEAK CHAOS in the thermodynamic limit.")

plt.show()