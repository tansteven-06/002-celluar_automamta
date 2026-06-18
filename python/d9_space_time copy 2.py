import numpy as np
import matplotlib.pyplot as plt

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

L = 100
max_t = 200

print(f"Generating space-time diagram: L={L}, t_max={max_t}")

# Random initial configuration
np.random.seed(12345)
config = tuple(np.random.choice([-1, 0, 1], 2*L))
traj = evolve_config(f, config, L, max_t)

# Extract sigma values
sigma_data = np.zeros((max_t+1, L))
for t in range(max_t+1):
    for i in range(L):
        sigma_data[t, i] = traj[t][2*i]

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(sigma_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[0, L, max_t, 0], interpolation='nearest')
ax.set_xlabel('Site i', fontsize=14, fontweight='bold')
ax.set_ylabel('Time t', fontsize=14, fontweight='bold')
ax.set_title('Space-Time Diagram: Chaotic Thermalization\n'
             'd=27 Rule (111111111111111111111111111)', 
             fontsize=15, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
cbar.set_label(r'$\sigma$', fontsize=13)

plt.tight_layout()
plt.savefig('spacetime_diagram_sigma_new.png', dpi=200, bbox_inches='tight')
print("\nSaved: spacetime_diagram_sigma.png")
plt.show()