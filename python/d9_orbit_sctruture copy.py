import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# The specific d=9 rule
# ============================================================
rule_str = "111111111111111111111111111"

def char_to_val(c):
    if c == 'T':
        return -1
    elif c == '0':
        return 0
    else:
        return 1

f_values = [char_to_val(c) for c in rule_str]

otimes_new = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

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

f = f_to_array(f_values)

def config_to_idx(config, L):
    idx = 0
    for val in config:
        idx = idx * 3 + (val + 1)
    return idx

# ============================================================
# Compute orbit lengths
# ============================================================

L = 5 #x个数
states = [-1, 0, 1]
n_states = 3**(2*L)

print(f"Computing orbits for L={L} ({n_states} states)...")

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
        
        if length > 10000:
            break
    
    orbit_lengths.append(length)

orbit_lengths = np.array(orbit_lengths)

# ============================================================
# Plot - single histogram
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

# Log-spaced bins
bins = np.logspace(0, np.log10(max(orbit_lengths)+1), 40)
ax.hist(orbit_lengths, bins=bins, color='#4CAF50', edgecolor='white', alpha=0.85, linewidth=0.5)

ax.set_xscale('log')
ax.set_xlabel('Orbit Length (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Orbits', fontsize=13, fontweight='bold')
ax.set_title('Orbit Length Distribution\nd=27 Rule (111111111111111111111111111)', 
             fontsize=14, fontweight='bold')

# Mean line
mean_len = np.mean(orbit_lengths)
ax.axvline(x=mean_len, color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {mean_len:.1f}')

# Median line
median_len = np.median(orbit_lengths)
ax.axvline(x=median_len, color='blue', linestyle=':', linewidth=2.5, 
           label=f'Median: {median_len:.1f}')

# Shade regions
ax.axvspan(1, 10, alpha=0.1, color='red', label='Short orbits (<10)')
ax.axvspan(50, max(orbit_lengths)*1.1, alpha=0.1, color='green', label='Long orbits (≥50)')

ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle=':')
ax.set_axisbelow(True)

# Statistics box
n_short = np.sum(orbit_lengths < 10)
n_long = np.sum(orbit_lengths >= 50)
n_total = len(orbit_lengths)

stats_text = (
    f'Total states: {n_states}\n'
    f'Total orbits: {n_total}\n'
    f'Mean orbit length: {mean_len:.1f}\n'
    f'Median orbit length: {median_len:.1f}\n'
    f'Max orbit length: {max(orbit_lengths)}\n'
    f'Short orbits (<10): {n_short} ({n_short/n_total*100:.1f}%)\n'
    f'Long orbits (≥50): {n_long} ({n_long/n_total*100:.1f}%)'
)

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, family='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
plt.savefig('3_orbit_structure_new.png', dpi=200, bbox_inches='tight')
print("\nSaved: 3_orbit_structure.png")
plt.show()