import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Fixed otimes (Table #1)
# ============================================================
otimes = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

def state_to_idx(val):
    return val + 1

def apply_otimes(a, b):
    return otimes[state_to_idx(a), state_to_idx(b)]

# ============================================================
# G-invariant rule parameterization
# ============================================================

# Orbit representatives under G = S_3 × Z_2
# 6 orbits in the domain {-1,0,1}^3
orbit_repr = [
    (0, 0, 0),      # Orbit 0
    (1, 0, 0),      # Orbit 1
    (1, 1, 0),      # Orbit 2
    (1, -1, 0),     # Orbit 3
    (1, 1, 1),      # Orbit 4
    (1, 1, -1),     # Orbit 5
]

# Periodic rule: all orbits = 1
f_periodic_orbits = np.array([1, 1, 1, 1, 1, 1])

# Chaotic d=9 rule: find its orbit values
rule_str = "T01101001T101010010110T0001"
def char_to_val(c):
    if c == 'T': return -1
    elif c == '0': return 0
    else: return 1
f_chaotic_27 = np.array([char_to_val(c) for c in rule_str])

# Map 27 values to 6 orbit values
# Need the orbit map
from itertools import product

def permute(perm_type, x):
    a, b, c = x
    if perm_type == 'e': return (a, b, c)
    elif perm_type == '(12)': return (b, a, c)
    elif perm_type == '(13)': return (c, b, a)
    elif perm_type == '(23)': return (a, c, b)
    elif perm_type == '(123)': return (b, c, a)
    elif perm_type == '(132)': return (c, a, b)

def apply_S(x):
    return (-x[0], -x[1], -x[2])

# Generate orbit map
all_points = list(product([-1, 0, 1], repeat=3))
orbit_map_27 = np.zeros(27, dtype=int)

for idx, x in enumerate(all_points):
    # Check which orbit it belongs to
    for orb_idx, rep in enumerate(orbit_repr):
        # Check if x is in the same orbit as rep under G
        found = False
        for p in ['e', '(12)', '(13)', '(23)', '(123)', '(132)']:
            for eps in [1, -1]:
                y = permute(p, x)
                if eps == -1:
                    y = apply_S(y)
                if y == rep:
                    orbit_map_27[idx] = orb_idx
                    found = True
                    break
            if found:
                break
        if found:
            break

# Get orbit values for chaotic rule
f_chaotic_orbits = np.zeros(6, dtype=int)
for orb in range(6):
    # Find the first point in this orbit
    for idx in range(27):
        if orbit_map_27[idx] == orb:
            f_chaotic_orbits[orb] = f_chaotic_27[idx]
            break

print("Periodic rule orbit values:", f_periodic_orbits)
print("Chaotic rule orbit values:", f_chaotic_orbits)

# ============================================================
# Parameterize along one orbit direction
# ============================================================

def make_f_from_orbits(orbit_vals):
    """Create full 27-value f from 6 orbit values"""
    f_27 = np.zeros(27, dtype=int)
    for idx in range(27):
        orb = orbit_map_27[idx]
        f_27[idx] = orbit_vals[orb]
    return f_27

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

def evolve_single_site(f, L, n_transient, n_steps):
    states_list = [-1, 0, 1]
    config = list(np.random.choice(states_list, 2*L))
    
    for _ in range(n_transient):
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
    
    sigma_series = []
    for _ in range(n_steps):
        sigma_series.append(config[0])
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
    
    return np.array(sigma_series)

def find_peaks_manual(signal, height=0.2, distance=3):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > height and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if not peaks or i - peaks[-1] >= distance:
                peaks.append(i)
    return peaks

def detect_period(sigma_series, max_lag=300):
    sigma_centered = sigma_series - np.mean(sigma_series)
    corr = np.correlate(sigma_centered, sigma_centered, mode='full')
    corr = corr[len(corr)//2:]
    if abs(corr[0]) < 1e-10:
        return -1
    corr = corr / corr[0]
    peaks = find_peaks_manual(corr[1:max_lag], height=0.15, distance=3)
    if len(peaks) > 0:
        return peaks[0] + 1
    else:
        return 0  # chaotic

# ============================================================
# Scan: vary orbit 0 (the (0,0,0) orbit)
# ============================================================

L = 4
n_transient = 500
n_steps = 2000

# Parameterize: f_λ = (1-λ) * f_periodic + λ * f_chaotic
# But only for specific orbits
lambda_values = np.linspace(0, 1, 200)
periods_global = []

print(f"Scanning λ ∈ [0,1] (global interpolation)...")
for lam in lambda_values:
    f_cont = (1 - lam) * f_periodic_orbits + lam * f_chaotic_orbits
    f_disc = np.round(f_cont).astype(int)
    f_disc = np.clip(f_disc, -1, 1)
    f_27 = make_f_from_orbits(f_disc)
    f_arr = f_to_array(f_27)
    
    sigma_series = evolve_single_site(f_arr, L, n_transient, n_steps)
    period = detect_period(sigma_series)
    periods_global.append(period)
    
    if lam % 0.1 < 0.005:
        print(f"  λ = {lam:.2f}: period = {period}")

periods_global = np.array(periods_global)

# Find period doublings
unique_p = []
first_lam = []
for i, p in enumerate(periods_global):
    if p > 0 and p not in unique_p:
        unique_p.append(p)
        first_lam.append(lambda_values[i])

if len(unique_p) > 0:
    sorted_idx = np.argsort(unique_p)
    unique_p = np.array(unique_p)[sorted_idx]
    first_lam = np.array(first_lam)[sorted_idx]
    
    print(f"\nPeriod transitions:")
    for p, lam in zip(unique_p, first_lam):
        print(f"  Period {p}: λ = {lam:.6f}")

# Check for period doubling
doubling_seq = [1, 2, 4, 8, 16, 32, 64]
doubling_lams = []
for target in doubling_seq:
    found = False
    for p, lam in zip(unique_p, first_lam):
        if p == target:
            doubling_lams.append(lam)
            found = True
            break
    if not found:
        break

deltas = []
if len(doubling_lams) >= 3:
    print(f"\nPeriod doubling lambdas:")
    for p, lam in zip(doubling_seq[:len(doubling_lams)], doubling_lams):
        print(f"  Period {p}: λ = {lam:.6f}")
    
    for i in range(len(doubling_lams)-2):
        num = doubling_lams[i+1] - doubling_lams[i]
        den = doubling_lams[i+2] - doubling_lams[i+1]
        if abs(den) > 1e-15:
            deltas.append(num/den)
    
    if deltas:
        print(f"\nFeigenbaum δ estimates: {deltas}")
        print(f"Average δ = {np.mean(deltas):.6f}")
        print(f"True Feigenbaum δ = 4.669...")
else:
    print(f"\nOnly {len(doubling_lams)} period doublings found.")

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
for i, lam in enumerate(lambda_values):
    if periods_global[i] > 0:
        ax.plot(lam, periods_global[i], 'bo', markersize=3, alpha=0.5)

for p, lam in zip(doubling_seq[:len(doubling_lams)], doubling_lams):
    ax.axvline(x=lam, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

ax.set_xlabel('λ', fontsize=13, fontweight='bold')
ax.set_ylabel('Period', fontsize=13, fontweight='bold')
ax.set_title('Bifurcation Diagram: Global Interpolation', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bifurcation_global.png', dpi=150, bbox_inches='tight')
print("Saved: bifurcation_global.png")
plt.show()