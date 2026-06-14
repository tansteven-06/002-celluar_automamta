import numpy as np
from itertools import product
from collections import Counter

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

def config_to_idx(config, L):
    idx = 0
    for val in config:
        idx = idx * 3 + (val + 1)
    return idx

def idx_to_config(idx, L):
    config = []
    for _ in range(2*L):
        config.append((idx % 3) - 1)
        idx //= 3
    return tuple(reversed(config))

def evolve_one_step(f, config, L):
    """Evolve configuration by one step"""
    sigmas = list(config[0::2])
    rs = list(config[1::2])
    
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
    
    return tuple(new_config)

def compute_orbit_lengths(f, L, max_steps=10000):
    """Compute orbit lengths for all configurations"""
    n_states = 3**(2*L)
    visited = np.zeros(n_states, dtype=int)  # 0=unvisited, 1=visited
    orbit_lengths = []
    
    all_configs = list(product(states, repeat=2*L))
    
    for config in all_configs:
        idx = config_to_idx(config, L)
        if visited[idx]:
            continue
        
        # Follow orbit
        orbit = []
        current = config
        current_idx = idx
        
        while not visited[current_idx]:
            visited[current_idx] = 1
            orbit.append(current_idx)
            current = evolve_one_step(f, current, L)
            current_idx = config_to_idx(current, L)
            
            if len(orbit) > max_steps:
                break
        
        orbit_lengths.append(len(orbit))
    
    return orbit_lengths

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
# Main analysis
# ============================================================

print("=" * 70)
print("ORBIT LENGTH ANALYSIS: d=9 vs d=27 RULES (L=4)")
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

print(f"\nFound {len(d9_rules)} d=9 rules")

# Analyze orbit lengths for L=4 (3^8 = 6561 states)
L = 4
print(f"\nAnalyzing L={L} ({3**(2*L)} states)")

for idx, f_vec in enumerate(d9_rules):
    print(f"\n--- d=9 Rule #{idx+1} ---")
    f = f_to_array(f_vec)
    orbit_lengths = compute_orbit_lengths(f, L)
    
    lengths_count = Counter(orbit_lengths)
    total_orbits = len(orbit_lengths)
    
    print(f"  Total orbits: {total_orbits}")
    print(f"  Unique orbit lengths: {len(lengths_count)}")
    print(f"  Orbit length distribution (top 10):")
    for length, count in lengths_count.most_common(10):
        print(f"    Length {length}: {count} orbits ({count/total_orbits*100:.1f}%)")
    
    avg_length = np.mean(orbit_lengths)
    max_length = max(orbit_lengths)
    print(f"  Average orbit length: {avg_length:.1f}")
    print(f"  Max orbit length: {max_length}")

# d=27 rule
print(f"\n--- d=27 Rule (f ≡ 1) ---")
f_one = f_to_array(np.ones(27, dtype=int))
orbit_lengths_one = compute_orbit_lengths(f_one, L)
lengths_count_one = Counter(orbit_lengths_one)

print(f"  Total orbits: {len(orbit_lengths_one)}")
print(f"  Unique orbit lengths: {len(lengths_count_one)}")
print(f"  Orbit length distribution:")
for length, count in lengths_count_one.most_common():
    print(f"    Length {length}: {count} orbits ({count/len(orbit_lengths_one)*100:.1f}%)")
print(f"  Average orbit length: {np.mean(orbit_lengths_one):.1f}")