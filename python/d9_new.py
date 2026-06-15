import numpy as np
from itertools import product

# ============================================================
# Both otimes tables
# ============================================================
otimes_old = np.array([
    [ 1,  0, -1],
    [ 0, -1,  1],
    [-1,  1,  0]
])

otimes_new = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

def state_to_idx(val):
    return val + 1

def f_to_array(f_values):
    f = np.zeros((3, 3, 3), dtype=int)
    for idx, val in enumerate(f_values):
        i = idx // 9
        j = (idx % 9) // 3
        k = idx % 3
        f[i, j, k] = val
    return f

def apply_rule(f, s_left, s_self, s_right, r, otimes):
    f_out = f[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
    sigma_new = otimes[state_to_idx(f_out), state_to_idx(r)]
    r_new = s_self
    return sigma_new, r_new

def F_index(sigma_i, r_i, sigma_next, r_next):
    idx = 0
    for val in [sigma_i, r_i, sigma_next, r_next]:
        idx = idx * 3 + state_to_idx(val)
    return idx

def build_constraint_matrix_L(f, L, otimes):
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
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i], otimes)
        
        for i in range(L):
            i_next = (i+1) % L
            M[row, F_index(sigmas_new[i], rs_new[i], 
                          sigmas_new[i_next], rs_new[i_next])] += 1
            M[row, F_index(sigmas[i], rs[i], 
                          sigmas[i_next], rs[i_next])] -= 1
        
        row += 1
    
    return M

def get_d_value(f, L, otimes):
    M = build_constraint_matrix_L(f, L, otimes)
    rank = np.linalg.matrix_rank(M)
    return 81 - rank

# ============================================================
# Scan many random rules
# ============================================================

print("=" * 70)
print("SCANNING: Same f under DIFFERENT ⊗ tables")
print("=" * 70)

np.random.seed(12345)
n_samples = 500
results = {
    'both_9': 0,      # thermalizing in both
    'new9_old_gt9': 0, # thermalizing in new, non-thermalizing in old
    'new_gt9_old9': 0, # non-thermalizing in new, thermalizing in old
    'both_gt9': 0      # non-thermalizing in both
}

L = 3

for i in range(n_samples):
    f_vec = np.random.choice([-1, 0, 1], 27)
    f = f_to_array(f_vec)
    
    d_new = get_d_value(f, L, otimes_new)
    d_old = get_d_value(f, L, otimes_old)
    
    if d_new == 9 and d_old == 9:
        results['both_9'] += 1
    elif d_new == 9 and d_old > 9:
        results['new9_old_gt9'] += 1
    elif d_new > 9 and d_old == 9:
        results['new_gt9_old9'] += 1
    else:
        results['both_gt9'] += 1
    
    if (i+1) % 100 == 0:
        print(f"  Progress: {i+1}/{n_samples}")

print(f"\nResults ({n_samples} random rules, L={L}):")
print(f"  Both thermalizing (d=9 in both):        {results['both_9']} ({results['both_9']/n_samples*100:.1f}%)")
print(f"  New thermalizing, Old non-thermalizing: {results['new9_old_gt9']} ({results['new9_old_gt9']/n_samples*100:.1f}%)")
print(f"  New non-thermalizing, Old thermalizing: {results['new_gt9_old9']} ({results['new_gt9_old9']/n_samples*100:.1f}%)")
print(f"  Both non-thermalizing (d>9 in both):    {results['both_gt9']} ({results['both_gt9']/n_samples*100:.1f}%)")

if results['new9_old_gt9'] > 0 or results['new_gt9_old9'] > 0:
    print("\n  *** CROSSOVER EXISTS: ⊗ table matters for thermalization! ***")
else:
    print("\n  No crossover detected: ⊗ table does not change thermalization status")