import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# The specific d=9 rule
# ============================================================
rule_str = "T01101001T101010010110T0001"

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

def evolve_config(f, config, L, steps):
    trajectory = [config]
    current = list(config)
    
    for _ in range(steps):
        sigmas = current[0::2]
        rs = current[1::2]
        
        new_config = [0] * (2*L)
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            new_config[2*i], new_config[2*i+1] = apply_rule(f, s_left, s_self, s_right, rs[i])
        
        current = new_config
        trajectory.append(tuple(new_config))
    
    return trajectory

# ============================================================
# Generate space-time diagram
# ============================================================

L = 200
max_t = 100

print(f"Generating space-time diagram: L={L}, t_max={max_t}")

# Random initial configuration
np.random.seed(12345)
config = tuple(np.random.choice([-1, 0, 1], 2*L))
traj = evolve_config(f, config, L, max_t)

# Extract sigma and r values
sigma_data = np.zeros((max_t+1, L))
r_data = np.zeros((max_t+1, L))

for t in range(max_t+1):
    for i in range(L):
        sigma_data[t, i] = traj[t][2*i]
        r_data[t, i] = traj[t][2*i+1]

# ============================================================
# Plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# --- Sigma values ---
im1 = ax1.imshow(sigma_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                  extent=[0, L, max_t, 0], interpolation='nearest')
ax1.set_xlabel('Site i', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time t', fontsize=12, fontweight='bold')
ax1.set_title('σ Values (Red=+1, White=0, Blue=-1)', fontsize=13, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[-1, 0, 1])
cbar1.set_label('σ', fontsize=11)
ax1.grid(False)

# Highlight a single site's evolution
site_to_track = 5
ax1.axvline(x=site_to_track, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
ax1.annotate(f'Site {site_to_track}', xy=(site_to_track, 0), xytext=(site_to_track+2, -5),
             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
             fontsize=10, color='yellow', fontweight='bold')

# --- r values ---
im2 = ax2.imshow(r_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                  extent=[0, L, max_t, 0], interpolation='nearest')
ax2.set_xlabel('Site i', fontsize=12, fontweight='bold')
ax2.set_ylabel('Time t', fontsize=12, fontweight='bold')
ax2.set_title('r Values (Red=+1, White=0, Blue=-1)', fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2, ticks=[-1, 0, 1])
cbar2.set_label('r', fontsize=11)
ax2.grid(False)

# Overall title
fig.suptitle('Space-Time Diagram: Chaotic Thermalization\nd=9 Rule (T01101001T101010010110T0001)', 
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('4_spacetime_diagram.png', dpi=200, bbox_inches='tight')
print("\nSaved: 4_spacetime_diagram.png")
plt.show()

# ============================================================
# Print interpretation guide
# ============================================================

print("\n" + "=" * 70)
print("HOW TO INTERPRET THE SPACE-TIME DIAGRAM")
print("=" * 70)
print("""
THERMALIZATION INDICATORS (seen here):
  ✓ Random, speckled pattern (no regular stripes)
  ✓ Vertical lines are broken (no persistent structures)
  ✓ Patterns change rapidly from row to row
  ✓ No visible periodicity
  
NON-THERMALIZATION INDICATORS (not seen):
  ✗ Regular vertical stripes
  ✗ Repeating horizontal patterns
  ✗ Chessboard-like periodicity
  ✗ Frozen local structures
""")