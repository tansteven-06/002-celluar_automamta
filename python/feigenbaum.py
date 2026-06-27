import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Generate continuous family of otimes tables
# ============================================================

# Three commutative solutions (all isomorphic to Z_3 addition)
table1 = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

table3 = np.array([
    [ 1,  0, -1],
    [ 0, -1,  1],
    [-1,  1,  0]
])

def make_otimes(theta):
    """
    Create a continuous family of otimes tables.
    theta = 0 -> table1
    theta = pi/2 -> table3
    """
    w1 = np.cos(theta)**2
    w3 = np.sin(theta)**2
    
    table_cont = w1 * table1 + w3 * table3
    table_disc = np.round(table_cont).astype(int)
    table_disc = np.clip(table_disc, -1, 1)
    
    return table_disc

# ============================================================
# Fixed chaotic rule
# ============================================================
rule_str = "T01101001T101010010110T0001"

def char_to_val(c):
    if c == 'T': return -1
    elif c == '0': return 0
    else: return 1

f_chaotic_values = np.array([char_to_val(c) for c in rule_str])

def f_to_array(f_values):
    f = np.zeros((3, 3, 3), dtype=int)
    for idx, val in enumerate(f_values):
        i = idx // 9
        j = (idx % 9) // 3
        k = idx % 3
        f[i, j, k] = val
    return f

f = f_to_array(f_chaotic_values)

# ============================================================
# Evolution functions
# ============================================================

def state_to_idx(val):
    return val + 1

def apply_rule(f, s_left, s_self, s_right, r, otimes):
    f_out = f[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
    sigma_new = otimes[state_to_idx(f_out), state_to_idx(r)]
    r_new = s_self
    return sigma_new, r_new

def evolve_single_site(f, otimes, L, n_transient, n_steps):
    """Evolve and return time series of site 0"""
    states_list = [-1, 0, 1]
    config = list(np.random.choice(states_list, 2*L))
    
    # Transient
    for _ in range(n_transient):
        sigmas = config[0::2]
        rs = config[1::2]
        new_config = [0] * (2*L)
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i], otimes)
            new_config[2*i] = new_s
            new_config[2*i+1] = new_r
        config = new_config
    
    # Record
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
            new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i], otimes)
            new_config[2*i] = new_s
            new_config[2*i+1] = new_r
        config = new_config
    
    return np.array(sigma_series)

def find_peaks_manual(signal, height=0.2, distance=3):
    """Manual peak finding"""
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > height and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if not peaks or i - peaks[-1] >= distance:
                peaks.append(i)
    return peaks

def detect_period_from_autocorr(sigma_series, max_lag=300):
    """Detect dominant period using autocorrelation"""
    sigma_centered = sigma_series - np.mean(sigma_series)
    
    # Autocorrelation
    corr = np.correlate(sigma_centered, sigma_centered, mode='full')
    corr = corr[len(corr)//2:]
    
    if abs(corr[0]) < 1e-10:
        return -1
    
    corr = corr / corr[0]
    
    # Find first significant peak
    peaks = find_peaks_manual(corr[1:max_lag], height=0.15, distance=3)
    
    if len(peaks) > 0:
        return peaks[0] + 1
    else:
        # Check if signal is chaotic
        if np.max(np.abs(corr[20:])) < 0.2:
            return 0  # chaotic
        else:
            return -1  # unclear

# ============================================================
# Scan theta parameter
# ============================================================

L = 4
n_transient = 500
n_steps = 2000
theta_values = np.linspace(0, np.pi/2, 300)

periods = []

print(f"Scanning θ ∈ [0, π/2] with {len(theta_values)} points...")
print(f"(L={L}, transient={n_transient}, steps={n_steps})")
print()

for i, theta in enumerate(theta_values):
    otimes = make_otimes(theta)
    sigma_series = evolve_single_site(f, otimes, L, n_transient, n_steps)
    period = detect_period_from_autocorr(sigma_series)
    periods.append(period)
    
    if i % 30 == 0:
        lam_eq = np.sin(theta)**2
        print(f"  θ = {theta:.4f} (λ_eq ≈ {lam_eq:.3f}): period = {period}")

periods = np.array(periods)

# ============================================================
# Analyze period doubling sequence
# ============================================================

print("\n" + "=" * 60)
print("PERIOD DOUBLING ANALYSIS")
print("=" * 60)

# Find unique periods and their first appearance
unique_periods = []
first_appearance = []

for i, p in enumerate(periods):
    if p > 0 and p not in unique_periods:
        unique_periods.append(p)
        first_appearance.append(theta_values[i])

# Sort by period
if len(unique_periods) > 0:
    sorted_idx = np.argsort(unique_periods)
    unique_periods = np.array(unique_periods)[sorted_idx]
    first_appearance = np.array(first_appearance)[sorted_idx]

    print("\nFirst appearance of each period:")
    for p, th in zip(unique_periods, first_appearance):
        lam_eq = np.sin(th)**2
        print(f"  Period {p}: θ = {th:.6f} (λ_eq = {lam_eq:.6f})")

# Check for period doubling
doubling_sequence = [1, 2, 4, 8, 16, 32, 64]
doubling_thetas = []

for target in doubling_sequence:
    found = False
    for p, th in zip(unique_periods, first_appearance):
        if p == target:
            doubling_thetas.append(th)
            found = True
            break
    if not found:
        break

deltas = []

if len(doubling_thetas) >= 3:
    print(f"\nPeriod doubling sequence:")
    for p, th in zip(doubling_sequence[:len(doubling_thetas)], doubling_thetas):
        print(f"  Period {p}: θ_{p} = {th:.6f}")
    
    # Estimate Feigenbaum delta
    for i in range(len(doubling_thetas)-2):
        num = doubling_thetas[i+1] - doubling_thetas[i]
        den = doubling_thetas[i+2] - doubling_thetas[i+1]
        if abs(den) > 1e-15:
            d = num / den
            deltas.append(d)
    
    if deltas:
        print(f"\n{'='*60}")
        print(f"FEIGENBAUM DELTA ESTIMATES")
        print(f"{'='*60}")
        for i, d in enumerate(deltas):
            print(f"  δ_{i+1} = {d:.6f}")
        print(f"  Average δ = {np.mean(deltas):.6f}")
        print(f"  True Feigenbaum δ = 4.6692016...")
else:
    print(f"\nFound only {len(doubling_thetas)} period doublings, need ≥3 for Feigenbaum δ")

# ============================================================
# Bifurcation diagram
# ============================================================

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Plot bifurcation diagram
for i, theta in enumerate(theta_values):
    if periods[i] > 0:
        ax1.plot(theta, periods[i], 'bo', markersize=3, alpha=0.5)

# Highlight period doublings
for p, th in zip(doubling_sequence[:len(doubling_thetas)], doubling_thetas):
    ax1.axvline(x=th, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(th, p*1.3, f'θ={th:.4f}', rotation=90, fontsize=9, color='red', fontweight='bold')

ax1.set_xlabel('θ (rotation angle)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Period', fontsize=13, fontweight='bold')
ax1.set_title('Bifurcation Diagram: θ-parameterized ⊗ table\nPeriod Doubling Route to Chaos', 
              fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feigenbaum_bifurcation.png', dpi=150, bbox_inches='tight')
print("\nSaved: feigenbaum_bifurcation.png")

# If we have deltas, also plot convergence
if len(deltas) >= 2:
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(range(1, len(deltas)+1), deltas, 'ro-', linewidth=2, markersize=10)
    ax2.axhline(y=4.669, color='blue', linestyle='--', linewidth=2, label='True δ = 4.669')
    ax2.set_xlabel('Bifurcation step n', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Estimated δ_n', fontsize=13, fontweight='bold')
    ax2.set_title('Convergence to Feigenbaum δ', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feigenbaum_delta_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: feigenbaum_delta_convergence.png")

plt.show()