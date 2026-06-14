import numpy as np
from itertools import product

# ============================================================
# Basic definitions
# ============================================================
otimes_new = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

states = [-1, 0, 1]

def state_to_idx(val):
    return val + 1

def apply_otimes(a, b):
    return otimes_new[state_to_idx(a), state_to_idx(b)]

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

# ============================================================
# Perturbation propagation
# ============================================================

def evolve_config(f, config, L, steps):
    """Evolve a configuration for given steps, return trajectory"""
    trajectory = [config]
    current = config
    
    for _ in range(steps):
        sigmas = list(current[0::2])
        rs = list(current[1::2])
        
        sigmas_new = [0] * L
        rs_new = [0] * L
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i])
        
        new_config = []
        for i in range(L):
            new_config.append(sigmas_new[i])
            new_config.append(rs_new[i])
        current = tuple(new_config)
        trajectory.append(current)
    
    return trajectory

def hamming_distance(config1, config2, L):
    """Number of sites where config1 and config2 differ"""
    dist = 0
    for i in range(2*L):
        if config1[i] != config2[i]:
            dist += 1
    return dist

def analyze_perturbation(f, L, n_trials=20, max_steps=100):
    """Analyze how a local perturbation spreads"""
    
    all_distances = []
    
    for _ in range(n_trials):
        # Random initial configuration
        config1 = tuple(np.random.choice([-1, 0, 1], 2*L))
        
        # Create perturbed config: flip one site
        config2 = list(config1)
        perturb_site = np.random.randint(0, 2*L)
        config2[perturb_site] = np.random.choice([s for s in states if s != config2[perturb_site]])
        config2 = tuple(config2)
        
        # Evolve both
        traj1 = evolve_config(f, config1, L, max_steps)
        traj2 = evolve_config(f, config2, L, max_steps)
        
        # Track distance
        distances = [hamming_distance(traj1[t], traj2[t], L) for t in range(max_steps+1)]
        all_distances.append(distances)
    
    # Average over trials
    avg_distances = np.mean(all_distances, axis=0)
    
    return avg_distances

# ============================================================
# Time correlation function
# ============================================================

def compute_correlation(f, L, n_samples=50, max_steps=100):
    """Compute autocorrelation C(t) = <sigma_i(0) sigma_i(t)>"""
    
    all_correlations = []
    
    for _ in range(n_samples):
        config = tuple(np.random.choice([-1, 0, 1], 2*L))
        traj = evolve_config(f, config, L, max_steps)
        
        # Compute correlation for each site, average
        sigmas_0 = np.array([traj[0][2*i] for i in range(L)])
        
        correlations = []
        for t in range(max_steps+1):
            sigmas_t = np.array([traj[t][2*i] for i in range(L)])
            corr = np.mean(sigmas_0 * sigmas_t)
            correlations.append(corr)
        
        all_correlations.append(correlations)
    
    avg_correlations = np.mean(all_correlations, axis=0)
    return avg_correlations

# ============================================================
# Main analysis
# ============================================================

print("=" * 70)
print("PERTURBATION PROPAGATION & CORRELATION ANALYSIS")
print("d=9 vs d=27 RULES")
print("=" * 70)

# Find d=9 rules
np.random.seed(42)
d9_rules = []
attempts = 0
while len(d9_rules) < 3 and attempts < 500:
    f_vec = np.random.choice([-1, 0, 1], 27)
    f = f_to_array(f_vec)
    M = build_constraint_matrix_L(f, 3)
    rank = np.linalg.matrix_rank(M)
    dim_ker = 81 - rank
    
    if dim_ker == 9:
        d9_rules.append(f_vec.copy())
    
    attempts += 1

L = 6  # Larger system for better statistics
max_steps = 50

print(f"\nSystem size L={L}, max steps={max_steps}")

# Analyze d=9 rules
print("\n--- d=9 Rules ---")
for idx, f_vec in enumerate(d9_rules):
    print(f"\nRule #{idx+1}:")
    f = f_to_array(f_vec)
    
    # Perturbation propagation
    print("  Perturbation propagation...")
    avg_dist = analyze_perturbation(f, L, n_trials=30, max_steps=max_steps)
    
    # Fit exponential growth: d(t) ~ exp(lambda * t)
    # Use early times (before saturation)
    early_t = np.arange(1, min(15, max_steps))
    early_d = avg_dist[1:len(early_t)+1]
    
    if len(early_d) > 3:
        # Log-linear fit
        coeffs = np.polyfit(early_t, np.log(early_d + 0.01), 1)
        lambda_exp = coeffs[0]
        print(f"    Lyapunov exponent (early): {lambda_exp:.4f}")
    else:
        lambda_exp = 0
    
    print(f"    Initial distance: {avg_dist[0]:.1f}")
    print(f"    Distance at t=10: {avg_dist[10]:.2f}")
    print(f"    Distance at t=30: {avg_dist[30]:.2f}")
    print(f"    Distance at t=50: {avg_dist[50]:.2f}")
    print(f"    Saturation level: {np.mean(avg_dist[20:]):.2f}")

# Analyze d=27 rule
print(f"\n--- d=27 Rule (f ≡ 1) ---")
f_one = f_to_array(np.ones(27, dtype=int))
avg_dist_one = analyze_perturbation(f_one, L, n_trials=30, max_steps=max_steps)

print(f"    Initial distance: {avg_dist_one[0]:.1f}")
print(f"    Distance at t=10: {avg_dist_one[10]:.2f}")
print(f"    Distance at t=30: {avg_dist_one[30]:.2f}")
print(f"    Distance at t=50: {avg_dist_one[50]:.2f}")
print(f"    Saturation level: {np.mean(avg_dist_one[20:]):.2f}")

# ============================================================
# Time correlation
# ============================================================

print("\n" + "=" * 70)
print("TIME CORRELATION ANALYSIS")
print("=" * 70)

for idx, f_vec in enumerate(d9_rules):
    print(f"\nd=9 Rule #{idx+1}:")
    f = f_to_array(f_vec)
    correlations = compute_correlation(f, L, n_samples=30, max_steps=max_steps)
    
    print(f"    C(0) = {correlations[0]:.4f}")
    print(f"    C(5) = {correlations[5]:.4f}")
    print(f"    C(10) = {correlations[10]:.4f}")
    print(f"    C(20) = {correlations[20]:.4f}")
    print(f"    C(50) = {correlations[50]:.4f}")
    
    # Estimate decay rate
    decay_ratio = correlations[20] / correlations[5] if correlations[5] != 0 else 0
    print(f"    Decay ratio C(20)/C(5): {decay_ratio:.4f}")

print(f"\nd=27 Rule (f ≡ 1):")
correlations_one = compute_correlation(f_one, L, n_samples=30, max_steps=max_steps)
print(f"    C(0) = {correlations_one[0]:.4f}")
print(f"    C(5) = {correlations_one[5]:.4f}")
print(f"    C(10) = {correlations_one[10]:.4f}")
print(f"    C(20) = {correlations_one[20]:.4f}")
print(f"    C(50) = {correlations_one[50]:.4f}")