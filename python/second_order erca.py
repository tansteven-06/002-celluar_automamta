import numpy as np
from itertools import product

# ============================================================
# Binary ERCA setup
# ============================================================

def f_to_array(f_values):
    """8-element f → 2×2×2 array indexed by [s_{i-1}, s_i, s_{i+1}]"""
    f = np.zeros((2, 2, 2), dtype=int)
    for idx, val in enumerate(f_values):
        i = idx // 4
        j = (idx % 4) // 2
        k = idx % 2
        f[i, j, k] = val
    return f

def apply_rule(f, s_left, s_self, s_right, s_prev):
    f_out = f[s_left, s_self, s_right]
    return f_out ^ s_prev  # XOR

# ============================================================
# Subgroup analysis for binary ERCA
# ============================================================

# S_3 action on {0,1}^3
def permute(pt, x):
    a, b, c = x
    if pt == 'e': return (a, b, c)
    elif pt == '(12)': return (b, a, c)
    elif pt == '(13)': return (c, b, a)
    elif pt == '(23)': return (a, c, b)
    elif pt == '(123)': return (b, c, a)
    elif pt == '(132)': return (c, a, b)

# No inversion in binary case (no negative values)
def compute_orbits(generators):
    """Compute orbits of {0,1}^3 under subgroup"""
    if not generators:
        return 8  # trivial: 8 orbits
    
    elements = {('e',)}
    queue = ['e']
    while queue:
        cp = queue.pop(0)
        for gp in generators:
            new_p = compose_perms(cp, gp)
            if new_p not in elements:
                elements.add(new_p)
                queue.append(new_p)
    
    all_points = list(product([0, 1], repeat=3))
    assigned = set()
    orbit_count = 0
    
    for x in all_points:
        if x in assigned:
            continue
        orbit = set()
        for pt in elements:
            y = permute(pt, x)
            orbit.add(y)
        for y in orbit:
            assigned.add(y)
        orbit_count += 1
    
    return orbit_count

def compose_perms(p2, p1):
    test = (0, 1, 2)
    r1 = permute(p1, test)
    r2 = permute(p2, r1)
    for p in ['e', '(12)', '(13)', '(23)', '(123)', '(132)']:
        if permute(p, test) == r2:
            return p
    return 'e'

# ============================================================
# Orbit counting for thermalization test
# ============================================================

def config_to_idx(config, L):
    idx = 0
    for val in config:
        idx = idx * 2 + val
    return idx

def sample_orbit_length(f, L, n_samples=50, max_steps=50000):
    """Sample orbit lengths"""
    lengths = []
    for _ in range(n_samples):
        config = list(np.random.choice([0, 1], 2*L))
        visited = {}
        step = 0
        
        while tuple(config) not in visited:
            visited[tuple(config)] = step
            step += 1
            
            sigmas = config[0::2]
            prevs = config[1::2]
            new_config = [0] * (2*L)
            for i in range(L):
                s_left = sigmas[(i-1) % L]
                s_self = sigmas[i]
                s_right = sigmas[(i+1) % L]
                new_config[2*i] = apply_rule(f, s_left, s_self, s_right, prevs[i])
                new_config[2*i+1] = s_self
            config = new_config
            
            if step >= max_steps:
                break
        
        if tuple(config) in visited:
            lengths.append(step - visited[tuple(config)])
        else:
            lengths.append(step)
    
    return np.mean(lengths)

# ============================================================
# Generate rules with and without 3-cycle symmetry
# ============================================================

# Binary ERCA: f has 8 values (2^3 = 8 inputs)
# 3-cycle symmetry constrains f to be constant on orbits of (123)
# (123) orbits on {0,1}^3:
# Orbit 0: (0,0,0) - size 1
# Orbit 1: (1,1,1) - size 1
# Orbit 2: (1,0,0), (0,1,0), (0,0,1) - size 3
# Orbit 3: (0,1,1), (1,0,1), (1,1,0) - size 3
# Total: 4 orbits → 2^4 = 16 rules with 3-cycle symmetry

# Without symmetry: 2^8 = 256 rules

np.random.seed(42)
L = 6

print("=" * 70)
print("BINARY ERCA: 3-CYCLE SYMMETRY vs THERMALIZATION")
print("=" * 70)

# Test random rules WITHOUT 3-cycle symmetry
print("\n--- Without 3-cycle symmetry ---")
orbit_lengths_no_sym = []
for _ in range(50):
    f_vec = np.random.choice([0, 1], 8)
    f_arr = f_to_array(f_vec)
    avg_len = sample_orbit_length(f_arr, L, n_samples=10, max_steps=20000)
    orbit_lengths_no_sym.append(avg_len)

print(f"  Mean orbit length: {np.mean(orbit_lengths_no_sym):.1f}")
print(f"  Min: {np.min(orbit_lengths_no_sym):.1f}, Max: {np.max(orbit_lengths_no_sym):.1f}")

# Test ALL rules WITH 3-cycle symmetry (only 16 rules)
print("\n--- With 3-cycle symmetry (all 16 rules) ---")
orbit_lengths_sym = []

# Generate all 16 rules
for orbit_vals in product([0, 1], repeat=4):
    # Build f from orbit values
    f_vec = np.zeros(8, dtype=int)
    for idx, (a, b, c) in enumerate(product([0, 1], repeat=3)):
        # Determine which orbit (a,b,c) belongs to
        if a == b == c:
            orb = 0 if a == 0 else 1
        elif sum([a, b, c]) == 1:
            orb = 2
        else:
            orb = 3
        f_vec[idx] = orbit_vals[orb]
    
    f_arr = f_to_array(f_vec)
    avg_len = sample_orbit_length(f_arr, L, n_samples=10, max_steps=20000)
    orbit_lengths_sym.append(avg_len)
    
    # Print rule
    rule_str = ''.join(str(v) for v in f_vec)
    print(f"  Rule {rule_str}: avg orbit = {avg_len:.1f}")

print(f"\n  Mean orbit length: {np.mean(orbit_lengths_sym):.1f}")
print(f"  Min: {np.min(orbit_lengths_sym):.1f}, Max: {np.max(orbit_lengths_sym):.1f}")

# Compare
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"  Without 3-cycle symmetry: {np.mean(orbit_lengths_no_sym):.0f}")
print(f"  With 3-cycle symmetry:    {np.mean(orbit_lengths_sym):.0f}")

if np.mean(orbit_lengths_sym) < np.mean(orbit_lengths_no_sym) * 0.5:
    print(f"\n  → 3-cycle symmetry DESTROYS thermalization in binary ERCA!")
else:
    print(f"\n  → 3-cycle symmetry does NOT significantly affect thermalization.")