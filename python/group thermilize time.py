import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

# ============================================================
# Setup
# ============================================================
OTIMES = np.array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]], dtype=int)

def state_to_idx(val):
    return val + 1

def apply_otimes(a, b):
    return OTIMES[state_to_idx(a), state_to_idx(b)]

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

def permute(pt, x):
    a, b, c = x
    if pt == 'e': return (a, b, c)
    elif pt == '(12)': return (b, a, c)
    elif pt == '(13)': return (c, b, a)
    elif pt == '(23)': return (a, c, b)
    elif pt == '(123)': return (b, c, a)
    elif pt == '(132)': return (c, a, b)

def apply_S(x):
    return (-x[0], -x[1], -x[2])

def compose_perms(p2, p1):
    test = (1, 2, 3)
    r1 = permute(p1, test)
    r2 = permute(p2, r1)
    for p in ['e', '(12)', '(13)', '(23)', '(123)', '(132)']:
        if permute(p, test) == r2: return p
    return 'e'

def generate_group_elements(generators):
    if not generators: return {('e', 1)}
    elements = {('e', 1)}
    queue = [('e', 1)]
    while queue:
        cp, ce = queue.pop(0)
        for gp, ge in generators:
            np_perm = compose_perms(cp, gp)
            np_eps = ce * ge
            ne = (np_perm, np_eps)
            if ne not in elements:
                elements.add(ne)
                queue.append(ne)
    return elements

def compute_orbits(generators):
    elements = generate_group_elements(generators)
    all_points = list(product([-1, 0, 1], repeat=3))
    orbit_map = np.zeros((3, 3, 3), dtype=int)
    assigned = set()
    orbit_idx = 0
    for x in all_points:
        if x in assigned: continue
        orbit = set()
        for pt, ep in elements:
            y = permute(pt, x)
            if ep == -1: y = apply_S(y)
            orbit.add(y)
        for y in orbit:
            assigned.add(y)
            orbit_map[state_to_idx(y[0]), state_to_idx(y[1]), state_to_idx(y[2])] = orbit_idx
        orbit_idx += 1
    return orbit_map, orbit_idx

def generate_symmetric_rule(orbit_map, n_orbits):
    orbit_vals = np.random.choice([-1, 0, 1], n_orbits)
    f_vec = np.zeros(27, dtype=int)
    for idx in range(27):
        i = idx // 9
        j = (idx % 9) // 3
        k = idx % 3
        orb = orbit_map[i, j, k]
        f_vec[idx] = orbit_vals[orb]
    return f_to_array(f_vec)

def get_d_value(f):
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

def compute_lyapunov(f, L, n_trials=3, n_steps=30):
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
            if d >= 2*L*0.8 or d == 0: break
            for cfg in [config1, config2]:
                sigmas = cfg[0::2]
                rs = cfg[1::2]
                new_cfg = [0]*(2*L)
                for i in range(L):
                    s_left = sigmas[(i-1)%L]
                    s_self = sigmas[i]
                    s_right = sigmas[(i+1)%L]
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

def compute_orbit_thermalization(f, L, n_samples=3, max_steps=50000):
    states = [-1, 0, 1]
    orbit_lengths = []
    for _ in range(n_samples):
        config = list(np.random.choice(states, 2*L))
        visited = {}
        step = 0
        while tuple(config) not in visited:
            visited[tuple(config)] = step
            step += 1
            sigmas = config[0::2]
            rs = config[1::2]
            new_config = [0]*(2*L)
            for i in range(L):
                s_left = sigmas[(i-1)%L]
                s_self = sigmas[i]
                s_right = sigmas[(i+1)%L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                new_config[2*i] = new_s
                new_config[2*i+1] = new_r
            config = new_config
            if step >= max_steps: break
        if tuple(config) in visited:
            orbit_lengths.append(step - visited[tuple(config)])
        else:
            orbit_lengths.append(step)
    return np.mean(orbit_lengths)

# ============================================================
# Subgroup classes
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
# Main computation: L=10, 100 rules per class
# ============================================================

np.random.seed(42)
L = 10
n_rules_per_class = 100

print("=" * 70)
print(f"TERNARY ERCA PHASE DIAGRAM (FRESH SAMPLING)")
print(f"L={L}, {n_rules_per_class} rules per class")
print(f"Method: Orbit tracking + Lyapunov exponent")
print("=" * 70)

all_results = {}

for class_name, generators in subgroup_classes:
    print(f"\n{class_name}...")
    start_time = time.time()
    
    orbit_map, n_orbits = compute_orbits(generators)
    
    d_list = []
    lyap_list = []
    orbit_list = []
    
    for i in range(n_rules_per_class):
        f_arr = generate_symmetric_rule(orbit_map, n_orbits)
        
        d = get_d_value(f_arr)
        d_list.append(d)
        
        lam = compute_lyapunov(f_arr, L, n_trials=2, n_steps=20)
        lyap_list.append(lam)
        
        orbit_len = compute_orbit_thermalization(f_arr, L, n_samples=1, max_steps=30000)
        orbit_list.append(orbit_len)
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_rules_per_class}")
    
    d_arr = np.array(d_list)
    lyap_arr = np.array(lyap_list)
    orbit_arr = np.array(orbit_list)
    
    all_results[class_name] = {
        'd_mean': np.mean(d_arr),
        'd_std': np.std(d_arr),
        'lyap_mean': np.mean(lyap_arr),
        'lyap_std': np.std(lyap_arr),
        'orbit_mean': np.mean(orbit_arr),
        'orbit_std': np.std(orbit_arr),
        'd_values': d_arr,
        'lyap_values': lyap_arr,
        'orbit_values': orbit_arr
    }
    
    thermal_frac = np.sum(d_arr == 9) / n_rules_per_class * 100
    elapsed = time.time() - start_time
    print(f"  d = {np.mean(d_arr):.1f} ± {np.std(d_arr):.1f} (thermalizing: {thermal_frac:.0f}%)")
    print(f"  λ = {np.mean(lyap_arr):.4f} ± {np.std(lyap_arr):.4f}")
    print(f"  Orbit length = {np.mean(orbit_arr):.1f} ± {np.std(orbit_arr):.1f}")
    print(f"  Time: {elapsed:.1f}s")

# ============================================================
# Generate 6 figures
# ============================================================

class_names = [c[0] for c in subgroup_classes]
x_pos = np.arange(len(class_names))

# Color based on actual data: green if d ≈ 9, red if d > 9
thermal_colors = []
for cn in class_names:
    if all_results[cn]['d_mean'] < 9.5:
        thermal_colors.append('#4CAF50')
    else:
        thermal_colors.append('#E91E63')

# FIGURE 1: d-value bar chart
fig1, ax = plt.subplots(figsize=(12, 7))
d_means = [all_results[cn]['d_mean'] for cn in class_names]
d_stds = [all_results[cn]['d_std'] for cn in class_names]

ax.bar(x_pos, d_means, yerr=d_stds, color=thermal_colors, alpha=0.85, 
       capsize=8, edgecolor='white', linewidth=1.5, width=0.7)
ax.axhline(y=9, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='d = 9 (threshold)')
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, fontsize=14, fontweight='bold')
ax.set_xlabel('Symmetry Class', fontsize=15, fontweight='bold')
ax.set_ylabel('d (nullspace dimension)', fontsize=15, fontweight='bold')
ax.set_title(f'Conservation Law Dimension (L={L}, n={n_rules_per_class})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig1_d_value.png', dpi=200, bbox_inches='tight')

# FIGURE 2: Lyapunov exponent
fig2, ax = plt.subplots(figsize=(12, 7))
lyap_means = [all_results[cn]['lyap_mean'] for cn in class_names]
lyap_stds = [all_results[cn]['lyap_std'] for cn in class_names]

ax.bar(x_pos, lyap_means, yerr=lyap_stds, color=thermal_colors, alpha=0.85, 
       capsize=8, edgecolor='white', linewidth=1.5, width=0.7)
ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.5, label='λ = 0')
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, fontsize=14, fontweight='bold')
ax.set_xlabel('Symmetry Class', fontsize=15, fontweight='bold')
ax.set_ylabel('Lyapunov Exponent λ', fontsize=15, fontweight='bold')
ax.set_title(f'Chaos Strength (L={L}, n={n_rules_per_class})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig2_lyapunov.png', dpi=200, bbox_inches='tight')

# FIGURE 3: Orbit length
fig3, ax = plt.subplots(figsize=(12, 7))
orbit_means = [all_results[cn]['orbit_mean'] for cn in class_names]
orbit_stds = [all_results[cn]['orbit_std'] for cn in class_names]

ax.bar(x_pos, orbit_means, yerr=orbit_stds, color=thermal_colors, alpha=0.85, 
       capsize=8, edgecolor='white', linewidth=1.5, width=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, fontsize=14, fontweight='bold')
ax.set_xlabel('Symmetry Class', fontsize=15, fontweight='bold')
ax.set_ylabel('Orbit Length', fontsize=15, fontweight='bold')
ax.set_title(f'Thermalization Proxy: Orbit Length (L={L}, n={n_rules_per_class})', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig3_orbit_length.png', dpi=200, bbox_inches='tight')

# FIGURE 4: λ vs d scatter
fig4, ax = plt.subplots(figsize=(12, 8))
for cn in class_names:
    d_vals = all_results[cn]['d_values']
    lyap_vals = all_results[cn]['lyap_values']
    is_thermal = d_vals <= 9
    
    if np.sum(is_thermal) > 0:
        ax.scatter(d_vals[is_thermal], lyap_vals[is_thermal], 
                   color='#4CAF50', alpha=0.5, s=30, marker='o', 
                   edgecolors='none', zorder=3)
    if np.sum(~is_thermal) > 0:
        ax.scatter(d_vals[~is_thermal], lyap_vals[~is_thermal], 
                   color='#E91E63', alpha=0.5, s=30, marker='x', 
                   linewidth=1, zorder=3)

ax.axvline(x=9.5, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Phase boundary')
ax.fill_between([8, 9.5], 0, 1, alpha=0.06, color='green')
ax.fill_between([9.5, 30], 0, 1, alpha=0.06, color='red')
ax.set_xlabel('d (nullspace dimension)', fontsize=15, fontweight='bold')
ax.set_ylabel('Lyapunov Exponent λ', fontsize=15, fontweight='bold')
ax.set_title(f'Phase Diagram: λ vs d (L={L}, n={n_rules_per_class})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_lambda_vs_d.png', dpi=200, bbox_inches='tight')

# FIGURE 5: Orbit length vs d scatter
fig5, ax = plt.subplots(figsize=(12, 8))
for cn in class_names:
    d_vals = all_results[cn]['d_values']
    orbit_vals = all_results[cn]['orbit_values']
    is_thermal = d_vals <= 9
    
    if np.sum(is_thermal) > 0:
        ax.scatter(d_vals[is_thermal], orbit_vals[is_thermal], 
                   color='#4CAF50', alpha=0.5, s=30, marker='o', 
                   edgecolors='none', zorder=3)
    if np.sum(~is_thermal) > 0:
        ax.scatter(d_vals[~is_thermal], orbit_vals[~is_thermal], 
                   color='#E91E63', alpha=0.5, s=30, marker='x', 
                   linewidth=1, zorder=3)

ax.axvline(x=9.5, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Phase boundary')
ax.fill_between([8, 9.5], 1e2, 1e6, alpha=0.06, color='green')
ax.fill_between([9.5, 30], 1e2, 1e6, alpha=0.06, color='red')
ax.set_xlabel('d (nullspace dimension)', fontsize=15, fontweight='bold')
ax.set_ylabel('Orbit Length', fontsize=15, fontweight='bold')
ax.set_title(f'Phase Diagram: Orbit Length vs d (L={L}, n={n_rules_per_class})', fontsize=16, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_orbit_vs_d.png', dpi=200, bbox_inches='tight')

# FIGURE 6: 3D Phase Diagram
fig6 = plt.figure(figsize=(14, 10))
ax = fig6.add_subplot(111, projection='3d')

for cn in class_names:
    d_vals = all_results[cn]['d_values']
    lyap_vals = all_results[cn]['lyap_values']
    orbit_vals = all_results[cn]['orbit_values']
    is_thermal = d_vals <= 9
    
    color = '#4CAF50' if all_results[cn]['d_mean'] < 9.5 else '#E91E63'
    marker = 'o' if all_results[cn]['d_mean'] < 9.5 else 'x'
    label = f'{cn} (Thermal)' if all_results[cn]['d_mean'] < 9.5 else f'{cn} (Non-thermal)'
    
    # Subsample for 3D plot clarity
    idx = np.random.choice(len(d_vals), min(100, len(d_vals)), replace=False)
    ax.scatter(d_vals[idx], lyap_vals[idx], orbit_vals[idx], 
               color=color, alpha=0.6, s=30, marker=marker, label=label)

ax.set_xlabel('d (nullspace dimension)', fontsize=12, fontweight='bold')
ax.set_ylabel('Lyapunov Exponent λ', fontsize=12, fontweight='bold')
ax.set_zlabel('Orbit Length', fontsize=12, fontweight='bold')
ax.set_title(f'3D Phase Diagram (L={L}, n={n_rules_per_class})', fontsize=15, fontweight='bold')
ax.legend(fontsize=7, loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('fig6_3d_phase_diagram.png', dpi=200, bbox_inches='tight')

# Summary table
print(f"\n{'='*70}")
print(f"SUMMARY TABLE")
print(f"{'='*70}")
print(f"{'Class':<8} {'d_mean':<8} {'λ_mean':<10} {'Orbit_mean':<12} {'Thermal%':<10}")
print(f"{'-'*48}")
for class_name, _ in subgroup_classes:
    res = all_results[class_name]
    thermal_pct = np.sum(res['d_values'] == 9) / n_rules_per_class * 100
    print(f"{class_name:<8} {res['d_mean']:<8.1f} {res['lyap_mean']:<10.4f} "
          f"{res['orbit_mean']:<12.1f} {thermal_pct:<10.0f}")

print("\nAll 6 figures saved:")
print("  fig1_d_value.png")
print("  fig2_lyapunov.png")
print("  fig3_orbit_length.png")
print("  fig4_lambda_vs_d.png")
print("  fig5_orbit_vs_d.png")
print("  fig6_3d_phase_diagram.png")
plt.show()