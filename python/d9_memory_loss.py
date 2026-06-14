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
# Compute time correlation
# ============================================================

L = 10
max_steps = 60
n_samples = 50

print(f"System size L={L}, max_steps={max_steps}, samples={n_samples}")
print("Computing time correlation...")

all_corr = []

for sample in range(n_samples):
    config = tuple(np.random.choice([-1, 0, 1], 2*L))
    traj = evolve_config(f, config, L, max_steps)
    
    sigmas_0 = np.array([traj[0][2*i] for i in range(L)])
    mean_0 = np.mean(sigmas_0)
    std_0 = np.std(sigmas_0)
    
    corr_t = []
    for t in range(max_steps+1):
        sigmas_t = np.array([traj[t][2*i] for i in range(L)])
        mean_t = np.mean(sigmas_t)
        std_t = np.std(sigmas_t)
        
        if std_0 > 0 and std_t > 0:
            corr = np.mean((sigmas_0 - mean_0) * (sigmas_t - mean_t)) / (std_0 * std_t)
        else:
            corr = 0
        corr_t.append(corr)
    all_corr.append(corr_t)

avg_corr = np.mean(all_corr, axis=0)
std_corr = np.std(all_corr, axis=0)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

t = np.arange(max_steps+1)

# Fill between for variance
ax.fill_between(t, avg_corr - std_corr, avg_corr + std_corr, 
                 alpha=0.25, color='#E91E63', label='±1 std')

# Main curve
ax.plot(t, avg_corr, 'r-', linewidth=2.5, label='C(t)')

# Zero reference
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Initial correlation
ax.scatter([0], [avg_corr[0]], color='blue', s=80, zorder=5)
ax.annotate(f'C(0) = {avg_corr[0]:.3f}', xy=(0, avg_corr[0]), 
            xytext=(5, avg_corr[0]+0.15),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=11, color='blue')

# Decay region
ax.axvspan(0, 8, alpha=0.08, color='orange')
ax.annotate('Rapid decay\nregion', xy=(3, 0.5), fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))

# Noise region
ax.axvspan(8, max_steps, alpha=0.08, color='green')
ax.annotate('Noise level\n(thermalized)', xy=(30, 0.12), fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3))

# Dashed lines for correlation bounds
ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=-0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Time t', fontsize=13, fontweight='bold')
ax.set_ylabel('Autocorrelation C(t)', fontsize=13, fontweight='bold')
ax.set_title('Memory Loss: Time Correlation Decay\nd=9 Rule (T01101001T101010010110T0001)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, max_steps)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)


plt.tight_layout()
plt.savefig('2_memory_loss.png', dpi=200, bbox_inches='tight')
print("\nSaved: 2_memory_loss.png")
plt.show()