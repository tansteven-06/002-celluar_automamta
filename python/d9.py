import numpy as np
from itertools import product

# ============================================================
# Basic definitions
# ============================================================
otimes_new = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

states = [-1, 0, 1]

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

def F_index(sigma_i, r_i, sigma_next, r_next):
    idx = 0
    for val in [sigma_i, r_i, sigma_next, r_next]:
        idx = idx * 3 + state_to_idx(val)
    return idx

def build_constraint_matrix_L(f, L):
    n_configs = 3**(2*L)
    M = np.zeros((n_configs, 81), dtype=float)
    row = 0
    
    for config in product(states, repeat=2*L):
        sigmas = list(config[0::2])
        rs = list(config[1::2])
        
        sigmas_new = [0] * L
        rs_new = [0] * L
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i])
        
        for i in range(L):
            i_next = (i+1) % L
            M[row, F_index(sigmas_new[i], rs_new[i], 
                          sigmas_new[i_next], rs_new[i_next])] += 1
            M[row, F_index(sigmas[i], rs[i], 
                          sigmas[i_next], rs[i_next])] -= 1
        
        row += 1
    
    return M

# ============================================================
# Find a d=9 rule
# ============================================================

print("Finding a d=9 rule...")
np.random.seed(12345)

d9_rule = None
attempts = 0
while d9_rule is None and attempts < 1000:
    f_vec = np.random.choice([-1, 0, 1], 27)
    f = f_to_array(f_vec)
    M = build_constraint_matrix_L(f, 3)
    rank = np.linalg.matrix_rank(M)
    dim_ker = 81 - rank
    
    if dim_ker == 9:
        d9_rule = f_vec.copy()
        print(f"Found at attempt {attempts+1}")
    
    attempts += 1

# ============================================================
# Output the rule
# ============================================================

def val_to_char(v):
    """Convert -1,0,1 to T,0,1"""
    if v == -1:
        return 'T'
    elif v == 0:
        return '0'
    else:
        return '1'

# The 27 inputs, ordered from (1,1,1) to (-1,-1,-1)
# Standard ordering: iterate from (1,1,1) down to (-1,-1,-1)
# Each coordinate: 1, 0, -1

print("\n" + "=" * 70)
print("d=9 RULE")
print("=" * 70)

# Build the rule string in order from (1,1,1) to (-1,-1,-1)
rule_str = ""
f_array_3d = f_to_array(d9_rule)

# Order: sigma_{i-1}, sigma_i, sigma_{i+1} each from 1 to -1
for s_left in [1, 0, -1]:
    for s_self in [1, 0, -1]:
        for s_right in [1, 0, -1]:
            val = f_array_3d[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
            rule_str += val_to_char(val)

print(f"\nRule string (from 111 to TTT):")
print(rule_str)

# Also show in structured format
print("\nStructured format (by σ_{i-1}):")
for s_left in [1, 0, -1]:
    left_char = val_to_char(s_left)
    block = ""
    for s_self in [1, 0, -1]:
        for s_right in [1, 0, -1]:
            val = f_array_3d[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
            block += val_to_char(val)
    print(f"  σ_{{i-1}}={left_char}: {block}")

# Verify d=9
f_check = f_to_array(d9_rule)
M_check = build_constraint_matrix_L(f_check, 3)
rank_check = np.linalg.matrix_rank(M_check)
dim_ker_check = 81 - rank_check
print(f"\nVerification: rank={rank_check}, dim ker={dim_ker_check}, d=9: {dim_ker_check==9}")