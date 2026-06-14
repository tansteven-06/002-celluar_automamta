import numpy as np
from itertools import product

rule_str = "T01101001T101010010110T0001"

def char_to_val(c):
    if c == 'T': return -1
    elif c == '0': return 0
    else: return 1

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

def sample_orbit_lengths(f, L, n_samples=500):
    """Sample orbit lengths by random initial conditions"""
    states = [-1, 0, 1]
    lengths = []
    
    for _ in range(n_samples):
        config = tuple(np.random.choice(states, 2*L))
        
        # Follow orbit
        visited = {}
        current = config
        step = 0
        
        while current not in visited:
            visited[current] = step
            step += 1
            
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
            
            if step > 100000:
                break
        
        orbit_len = step - visited.get(current, 0)
        lengths.append(orbit_len)
    
    return lengths

# Test different L
print("=" * 60)
print("Orbit length scaling with system size")
print("=" * 60)
print(f"{'L':<6} {'States':<10} {'Mean orbit':<12} {'Max orbit':<12}")
print("-" * 50)

for L in [3, 4, 5, 6]:
    n_states = 3**(2*L)
    lengths = sample_orbit_lengths(f, L, n_samples=200)
    print(f"{L:<6} {n_states:<10} {np.mean(lengths):<12.1f} {max(lengths):<12}")