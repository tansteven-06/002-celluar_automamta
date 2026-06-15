import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Setup (same as before, compact)
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

# ============================================================
# Group actions (compact)
# ============================================================

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
        if permute(p, test) == r2:
            return p
    return 'e'

def generate_group_elements(generators):
    if not generators:
        return {('e', 1)}
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
        if x in assigned:
            continue
        orbit = set()
        for pt, ep in elements:
            y = permute(pt, x)
            if ep == -1:
                y = apply_S(y)
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

# ============================================================
# Compute correlation with saturation
# ============================================================

def compute_correlation_with_saturation(f, L, n_samples=15, max_steps=80):
    """Compute C(t) and extract both decay time and saturation value"""
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
        
        corr_t = [1.0]
        
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
# Analysis: focus on H1, H3, G (thermalizing vs non-thermalizing)
# ============================================================

np.random.seed(42)
L = 10
n_rules = 15

print("=" * 70)
print(f"CORRELATION SATURATION ANALYSIS (L={L})")
print("=" * 70)

for class_name, generators in subgroup_classes:
    print(f"\n{class_name}...")
    orbit_map, n_orbits = compute_orbits(generators)
    
    all_saturation = []
    all_decay_time = []
    
    for i in range(n_rules):
        f_arr = generate_symmetric_rule(orbit_map, n_orbits)
        corr = compute_correlation_with_saturation(f_arr, L, n_samples=10, max_steps=80)
        
        # Saturation value: average of last 20 points
        saturation = np.mean(np.abs(corr[-20:]))
        all_saturation.append(saturation)
        
        # Decay time: first time |C(t)| < 0.15
        decay_time = -1
        for t in range(len(corr)):
            if abs(corr[t]) < 0.15:
                decay_time = t
                break
        all_decay_time.append(decay_time if decay_time > 0 else 80)
    
    all_saturation = np.array(all_saturation)
    all_decay_time = np.array(all_decay_time)
    
    print(f"  C_sat = {np.mean(all_saturation):.4f} ± {np.std(all_saturation):.4f}")
    print(f"  τ_decay = {np.mean(all_decay_time):.1f} ± {np.std(all_decay_time):.1f}")
    print(f"  Thermalizing fraction (C_sat < 0.2): {np.sum(all_saturation < 0.2)/n_rules*100:.0f}%")

# ============================================================
# Focused plot: H1 vs G
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = {'H1': '#2196F3', 'H3': '#FF9800', 'G': '#E91E63', 
          'H6A': '#4CAF50', 'H6B': '#9C27B0', 'H6C': '#795548'}

# Plot correlation functions for a few representative rules
for class_name, generators in [("H1", []), ("G", subgroup_classes[-1][1])]:
    orbit_map, n_orbits = compute_orbits(generators)
    
    for i in range(3):  # 3 rules per class
        f_arr = generate_symmetric_rule(orbit_map, n_orbits)
        corr = compute_correlation_with_saturation(f_arr, L, n_samples=8, max_steps=80)
        
        label = class_name if i == 0 else None
        ax1.plot(np.abs(corr), color=colors.get(class_name, 'gray'), 
                 alpha=0.6, linewidth=1.5, label=label)

ax1.axhline(y=0.15, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold')
ax1.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax1.set_ylabel('|C(t)|', fontsize=12, fontweight='bold')
ax1.set_title(f'Correlation Functions: H1 vs G (L={L})', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Bar chart of saturation values
class_names_list = [c[0] for c in subgroup_classes]
sat_means = []
sat_stds = []

for class_name, generators in subgroup_classes:
    orbit_map, n_orbits = compute_orbits(generators)
    sats = []
    for i in range(n_rules):
        f_arr = generate_symmetric_rule(orbit_map, n_orbits)
        corr = compute_correlation_with_saturation(f_arr, L, n_samples=8, max_steps=60)
        sats.append(np.mean(np.abs(corr[-15:])))
    sat_means.append(np.mean(sats))
    sat_stds.append(np.std(sats))

x_pos = np.arange(len(class_names_list))
bars = ax2.bar(x_pos, sat_means, yerr=sat_stds, capsize=5, alpha=0.8, edgecolor='white')

# Color bars: green if sat < 0.2 (thermalizing), red if sat > 0.2 (non-thermalizing)
for i, (sat, bar) in enumerate(zip(sat_means, bars)):
    if sat < 0.2:
        bar.set_color('#4CAF50')
    else:
        bar.set_color('#E91E63')

ax2.axhline(y=0.2, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='C_sat = 0.2')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(class_names_list, fontsize=11)
ax2.set_xlabel('Symmetry Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Saturation Value C_sat', fontsize=12, fontweight='bold')
ax2.set_title(f'Correlation Saturation by Symmetry Class (L={L})', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Thermalization vs Non-Thermalization: Correlation Saturation', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_saturation.png', dpi=150, bbox_inches='tight')
print("\nSaved: correlation_saturation.png")
plt.show()