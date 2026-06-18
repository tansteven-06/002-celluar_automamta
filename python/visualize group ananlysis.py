import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time

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

def idx_to_state(idx):
    return idx - 1

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
# Permutation and group actions
# ============================================================

def permute(perm_type, x):
    a, b, c = x
    if perm_type == 'e':
        return (a, b, c)
    elif perm_type == '(12)':
        return (b, a, c)
    elif perm_type == '(13)':
        return (c, b, a)
    elif perm_type == '(23)':
        return (a, c, b)
    elif perm_type == '(123)':
        return (b, c, a)
    elif perm_type == '(132)':
        return (c, a, b)

def apply_S(x):
    return (-x[0], -x[1], -x[2])

def compose_perms(p2, p1):
    test = (1, 2, 3)
    result1 = permute(p1, test)
    result2 = permute(p2, result1)
    for p in ['e', '(12)', '(13)', '(23)', '(123)', '(132)']:
        if permute(p, test) == result2:
            return p
    return 'e'

def generate_group_elements(generators):
    if not generators:
        return {('e', 1)}
    
    elements = {('e', 1)}
    queue = [('e', 1)]
    
    while queue:
        current_perm, current_eps = queue.pop(0)
        for gen_perm, gen_eps in generators:
            new_perm = compose_perms(current_perm, gen_perm)
            new_eps = current_eps * gen_eps
            new_elem = (new_perm, new_eps)
            if new_elem not in elements:
                elements.add(new_elem)
                queue.append(new_elem)
    
    return elements

def compute_orbits(generators):
    elements = generate_group_elements(generators)
    all_points = list(product([-1, 0, 1], repeat=3))
    
    orbit_map = np.zeros((3, 3, 3), dtype=int)
    orbits = []
    assigned = set()
    
    for x in all_points:
        if x in assigned:
            continue
        
        orbit = set()
        for perm_type, eps in elements:
            y = permute(perm_type, x)
            if eps == -1:
                y = apply_S(y)
            orbit.add(y)
        
        orbit_idx = len(orbits)
        orbits.append(sorted(list(orbit)))
        for y in orbit:
            assigned.add(y)
            orbit_map[state_to_idx(y[0]), state_to_idx(y[1]), state_to_idx(y[2])] = orbit_idx
    
    return orbits, orbit_map

# ============================================================
# Subgroup definitions
# ============================================================

subgroup_classes = [
    ("H1", []),
    ("H2A", [(('(12)', 1))]),
    ("H2B", [(('e', -1))]),
    ("H2C", [(('(12)', -1))]),
    ("H3", [(('(123)', 1))]),
    ("H4", [(('(12)', 1)), (('e', -1))]),
    ("H6A", [(('(12)', 1)), (('(123)', 1))]),
    ("H6B", [(('(123)', 1)), (('e', -1))]),
    ("H6C", [(('(123)', 1)), (('(12)', -1))]),
    ("G", [(('(12)', 1)), (('(123)', 1)), (('e', -1))]),
]

# ============================================================
# Analysis function
# ============================================================

def analyze_class(name, generators, n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    orbits, orbit_map = compute_orbits(generators)
    n_orbits = len(orbits)
    
    thermal_count = 0
    
    for _ in range(n_samples):
        orbit_vals = np.random.choice([-1, 0, 1], n_orbits)
        f_vec = np.zeros(27, dtype=int)
        for idx in range(27):
            i = idx // 9
            j = (idx % 9) // 3
            k = idx % 3
            orb = orbit_map[i, j, k]
            f_vec[idx] = orbit_vals[orb]
        
        f = f_to_array(f_vec)
        M = build_constraint_matrix_L(f, 3)
        rank = np.linalg.matrix_rank(M)
        dim_ker = 81 - rank
        
        if dim_ker == 9:
            thermal_count += 1
    
    return thermal_count / n_samples * 100

# ============================================================
# Run analysis for different sample sizes
# ============================================================

sample_sizes = [30, 50, 100, 500, 1000]
all_results = {}

print("=" * 70)
print("Running analysis for different sample sizes...")
print("=" * 70)

for n_samples in sample_sizes:
    print(f"\nSample size: {n_samples}")
    results = {}
    for name, generators in subgroup_classes:
        start_time = time.time()
        thermal_pct = analyze_class(name, generators, n_samples, seed=42+n_samples)
        elapsed = time.time() - start_time
        results[name] = thermal_pct
        print(f"  {name:<6}: {thermal_pct:5.1f}% thermalizing ({elapsed:.1f}s)")
    all_results[n_samples] = results

# ============================================================
# Create visualization
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
markers = ['o', 's', 'D', '^', 'v']

x_labels = [name for name, _ in subgroup_classes]
x_pos = np.arange(len(x_labels))

for i, n_samples in enumerate(sample_sizes):
    results = all_results[n_samples]
    thermal_values = [results[name] for name, _ in subgroup_classes]
    
    ax.plot(x_pos, thermal_values, 
            color=colors[i], 
            marker=markers[i], 
            linewidth=2,
            markersize=8,
            label=f'n = {n_samples}',
            alpha=0.85)

ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Symmetry Subgroup', fontsize=13, fontweight='bold')
ax.set_ylabel('Thermalization Ratio (%)', fontsize=13, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylim(-5, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()

plt.savefig('thermalization_vs_symmetry.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as 'thermalization_vs_symmetry.png'")

plt.show()