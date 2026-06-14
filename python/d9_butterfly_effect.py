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

def hamming_distance(c1, c2, L):
    return sum(1 for i in range(2*L) if c1[i] != c2[i])

# ============================================================
# Compute perturbation propagation
# ============================================================

L = 8
max_steps = 100 #时间长度
n_trials = 30

print(f"System size L={L}, max_steps={max_steps}, trials={n_trials}")
print("Computing perturbation propagation...")

all_distances = []
for trial in range(n_trials):
    config1 = tuple(np.random.choice([-1, 0, 1], 2*L))
    
    config2 = list(config1)
    site = np.random.randint(0, 2*L)
    old_val = config2[site]
    new_vals = [v for v in [-1, 0, 1] if v != old_val]
    config2[site] = np.random.choice(new_vals)
    config2 = tuple(config2)
    
    traj1 = evolve_config(f, config1, L, max_steps)
    traj2 = evolve_config(f, config2, L, max_steps)
    
    distances = [hamming_distance(traj1[t], traj2[t], L) for t in range(max_steps+1)]
    all_distances.append(distances)

avg_dist = np.mean(all_distances, axis=0)
std_dist = np.std(all_distances, axis=0)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

t = np.arange(max_steps+1)

# Fill between for variance
ax.fill_between(t, avg_dist - std_dist, avg_dist + std_dist, 
                 alpha=0.25, color="#1B7BCA", label='±1 std')

# Main curve
ax.plot(t, avg_dist, 'b-', linewidth=2.5, label='Mean distance')

# Reference lines
max_possible = 2 * L
ax.axhline(y=max_possible * 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='50% system')
ax.axhline(y=max_possible * 0.67, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='67% system')

# Initial point highlight
ax.scatter([0], [avg_dist[0]], color='red', s=80, zorder=5)
ax.annotate(f'Initial: 1 site', xy=(0, avg_dist[0]), xytext=(5, 3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11, color='red')

# Saturation point
sat_val = np.mean(avg_dist[-20:])
ax.scatter([max_steps], [sat_val], color='green', s=80, zorder=5)
ax.annotate(f'Saturation: {sat_val:.1f} sites\n({sat_val/max_possible*100:.0f}% of system)', 
            xy=(max_steps, sat_val), xytext=(max_steps-25, sat_val+3),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=11, color='green')

# Early exponential growth fit
early_t = t[1:10]
early_d = avg_dist[1:10]
coeffs = np.polyfit(early_t, np.log(early_d + 0.01), 1)
lambda_val = coeffs[0]
fit_curve = np.exp(coeffs[1]) * np.exp(lambda_val * t)
ax.plot(t[:15], fit_curve[:15], 'r--', linewidth=1.5, alpha=0.7, 
        label=f'Exponential fit (λ={lambda_val:.3f})')

ax.set_xlabel('Time t', fontsize=13, fontweight='bold')
ax.set_ylabel('Hamming Distance', fontsize=13, fontweight='bold')
ax.set_title('Butterfly Effect: Perturbation Propagation\nd=9 Rule (T01101001T101010010110T0001)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_ylim(0, max_possible * 1.05)
ax.set_xlim(0, max_steps)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('1_butterfly_effect.png', dpi=200, bbox_inches='tight')
print("\nSaved: 1_butterfly_effect.png")
plt.show()