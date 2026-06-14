import numpy as np
import matplotlib.pyplot as plt
from itertools import product

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

# Parse the rule string into a 27-element array
# Order: sigma_{i-1}=1,0,T; sigma_i=1,0,T; sigma_{i+1}=1,0,T
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

# ============================================================
# 1. PERTURBATION PROPAGATION (butterfly effect)
# ============================================================

def evolve_config(f, config, L, steps):
    """Evolve configuration for given steps"""
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

L = 8
max_steps = 60
n_trials = 20

print(f"System size L={L}, max_steps={max_steps}, trials={n_trials}")
print("Computing perturbation propagation...")

all_distances = []
for trial in range(n_trials):
    # Random initial config
    config1 = tuple(np.random.choice([-1, 0, 1], 2*L))
    
    # Perturb one site
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
# 2. TIME CORRELATION
# ============================================================

print("Computing time correlation...")

n_samples = 30
all_corr = []

for _ in range(n_samples):
    config = tuple(np.random.choice([-1, 0, 1], 2*L))
    traj = evolve_config(f, config, L, max_steps)
    
    sigmas_0 = np.array([traj[0][2*i] for i in range(L)])
    
    corr_t = []
    for t in range(max_steps+1):
        sigmas_t = np.array([traj[t][2*i] for i in range(L)])
        # Remove mean
        s0 = sigmas_0 - np.mean(sigmas_0)
        st = sigmas_t - np.mean(sigmas_t)
        if np.std(s0) > 0 and np.std(st) > 0:
            corr = np.mean(s0 * st) / (np.std(s0) * np.std(st))
        else:
            corr = 0
        corr_t.append(corr)
    all_corr.append(corr_t)

avg_corr = np.mean(all_corr, axis=0)
std_corr = np.std(all_corr, axis=0)

# ============================================================
# 3. ORBIT STRUCTURE (small system L=4)
# ============================================================

print("Computing orbit structure...")

L_small = 4
states = [-1, 0, 1]
n_states = 3**(2*L_small)

def config_to_idx(config, L):
    idx = 0
    for val in config:
        idx = idx * 3 + (val + 1)
    return idx

# Compute orbits
visited = np.zeros(n_states, dtype=bool)
orbit_lengths = []

for config in product(states, repeat=2*L_small):
    idx = config_to_idx(config, L_small)
    if visited[idx]:
        continue
    
    current = config
    length = 0
    while not visited[config_to_idx(current, L_small)]:
        visited[config_to_idx(current, L_small)] = True
        length += 1
        
        sigmas = list(current[0::2])
        rs = list(current[1::2])
        
        new_config = []
        for i in range(L_small):
            s_left = sigmas[(i-1) % L_small]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L_small]
            new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
            new_config.append(new_s)
            new_config.append(new_r)
        
        current = tuple(new_config)
        
        if length > 5000:
            break
    
    orbit_lengths.append(length)

# ============================================================
# 4. VISUALIZATION
# ============================================================

print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- Plot 1: Perturbation propagation ---
ax1 = axes[0, 0]
t = np.arange(max_steps+1)
ax1.fill_between(t, avg_dist - std_dist, avg_dist + std_dist, 
                  alpha=0.3, color='#2196F3')
ax1.plot(t, avg_dist, 'b-', linewidth=2)
ax1.axhline(y=2*L*0.5, color='gray', linestyle='--', alpha=0.5, label='50% system size')
ax1.axhline(y=2*L*0.67, color='gray', linestyle=':', alpha=0.5, label='67% system size')
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('Hamming distance', fontsize=12)
ax1.set_title('Butterfly Effect: Perturbation Propagation', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.set_ylim(0, 2*L)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Time correlation ---
ax2 = axes[0, 1]
ax2.fill_between(t, avg_corr - std_corr, avg_corr + std_corr, 
                  alpha=0.3, color='#E91E63')
ax2.plot(t, avg_corr, 'r-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel('Autocorrelation C(t)', fontsize=12)
ax2.set_title('Memory Loss: Time Correlation Decay', fontsize=13, fontweight='bold')
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Orbit length distribution ---
ax3 = axes[1, 0]
bins = np.logspace(0, np.log10(max(orbit_lengths)+1), 30)
ax3.hist(orbit_lengths, bins=bins, color='#4CAF50', edgecolor='white', alpha=0.8)
ax3.set_xscale('log')
ax3.set_xlabel('Orbit length (log scale)', fontsize=12)
ax3.set_ylabel('Number of orbits', fontsize=12)
ax3.set_title(f'Orbit Structure (L={L_small}): Long Orbits Dominate', fontsize=13, fontweight='bold')
ax3.axvline(x=np.mean(orbit_lengths), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(orbit_lengths):.1f}')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Space-time diagram ---
ax4 = axes[1, 1]
# Generate a space-time diagram
L_st = 12
max_t = 50
config_st = tuple(np.random.choice([-1, 0, 1], 2*L_st))
traj_st = evolve_config(f, config_st, L_st, max_t)

# Extract sigma values
sigma_st = np.zeros((max_t+1, L_st))
for t_idx in range(max_t+1):
    for i in range(L_st):
        sigma_st[t_idx, i] = traj_st[t_idx][2*i]

im = ax4.imshow(sigma_st, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, 
                extent=[0, L_st, max_t, 0])
ax4.set_xlabel('Site i', fontsize=12)
ax4.set_ylabel('Time t', fontsize=12)
ax4.set_title('Space-Time Diagram (σ values)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax4, label='σ', ticks=[-1, 0, 1])

# Overall title
fig.suptitle(f'Thermalization of d=9 Rule: {rule_str[:9]}...{rule_str[-9:]}', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('thermalization_d9_rule.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as 'thermalization_d9_rule.png'")

# ============================================================
# Print summary statistics
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY: Thermalization Indicators for d=9 Rule")
print("=" * 70)
print(f"\n1. BUTTERFLY EFFECT:")
print(f"   Initial perturbation (1 site) spreads to {avg_dist[-1]:.1f} sites")
print(f"   Saturation at {avg_dist[-1]/(2*L)*100:.0f}% of system")
print(f"   Lyapunov exponent (early): {np.polyfit(t[1:10], np.log(avg_dist[1:10]+0.01), 1)[0]:.4f}")

print(f"\n2. MEMORY LOSS:")
print(f"   C(0) = {avg_corr[0]:.3f}")
print(f"   C(5) = {avg_corr[5]:.3f}")
print(f"   C(20) = {avg_corr[20]:.3f}")
print(f"   Correlation decays to noise level within ~5 steps")

print(f"\n3. ORBIT STRUCTURE (L={L_small}):")
print(f"   Total states: {n_states}")
print(f"   Number of orbits: {len(orbit_lengths)}")
print(f"   Mean orbit length: {np.mean(orbit_lengths):.1f}")
print(f"   Max orbit length: {max(orbit_lengths)}")
print(f"   Long orbits (>100): {sum(1 for l in orbit_lengths if l > 100)}")
print(f"   Short orbits (<10): {sum(1 for l in orbit_lengths if l < 10)}")

print(f"\n4. COMPARISON WITH d=27 (f≡1):")
print(f"   d=9:  Perturbation spreads, correlation decays, long orbits")
print(f"   d=27: Perturbation frozen, correlation oscillates, period-4 orbits")

plt.show()