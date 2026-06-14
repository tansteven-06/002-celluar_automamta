import numpy as np
from itertools import product

# ============================================================
# 基本定义
# ============================================================
otimes_new = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
])

states = [-1, 0, 1]

def state_to_idx(val):
    return val + 1

def idx_to_state(idx):
    return idx - 1

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
# 置换和群作用
# ============================================================

def permute(perm_type, x):
    a, b, c = x
    if perm_type == 'e':
        return (a, b, c)
    elif perm_type == '(12)':
        return (b, a, c)
    elif perm_type == '(13)':
        return (c, b, a)
    elif perm_type == '(23)':
        return (a, c, b)
    elif perm_type == '(123)':
        return (b, c, a)
    elif perm_type == '(132)':
        return (c, a, b)

def apply_S(x):
    return (-x[0], -x[1], -x[2])

def compose_perms(p2, p1):
    test = (1, 2, 3)
    result1 = permute(p1, test)
    result2 = permute(p2, result1)
    for p in ['e', '(12)', '(13)', '(23)', '(123)', '(132)']:
        if permute(p, test) == result2:
            return p
    return 'e'

def generate_group_elements(generators):
    if not generators:
        return {('e', 1)}
    
    elements = {('e', 1)}
    queue = [('e', 1)]
    
    while queue:
        current_perm, current_eps = queue.pop(0)
        for gen_perm, gen_eps in generators:
            new_perm = compose_perms(current_perm, gen_perm)
            new_eps = current_eps * gen_eps
            new_elem = (new_perm, new_eps)
            if new_elem not in elements:
                elements.add(new_elem)
                queue.append(new_elem)
    
    return elements

def compute_orbits(generators):
    elements = generate_group_elements(generators)
    all_points = list(product([-1, 0, 1], repeat=3))
    
    orbit_map = np.zeros((3, 3, 3), dtype=int)
    orbits = []
    assigned = set()
    
    for x in all_points:
        if x in assigned:
            continue
        
        orbit = set()
        for perm_type, eps in elements:
            y = permute(perm_type, x)
            if eps == -1:
                y = apply_S(y)
            orbit.add(y)
        
        orbit_idx = len(orbits)
        orbits.append(sorted(list(orbit)))
        for y in orbit:
            assigned.add(y)
            orbit_map[state_to_idx(y[0]), state_to_idx(y[1]), state_to_idx(y[2])] = orbit_idx
    
    return orbits, orbit_map

# ============================================================
# 快速分析
# ============================================================

subgroup_classes = [
    ("H1 无对称", []),
    ("H2A 纯对换", [(('(12)', 1))]),
    ("H2B 纯反演", [(('e', -1))]),
    ("H2C 对换+反演", [(('(12)', -1))]),
    ("H3 纯轮换", [(('(123)', 1))]),
    ("H4 V4", [(('(12)', 1)), (('e', -1))]),
    ("H6A 纯S3", [(('(12)', 1)), (('(123)', 1))]),
    ("H6B A3xZ2", [(('(123)', 1)), (('e', -1))]),
    ("H6C 对角线S3", [(('(123)', 1)), (('(12)', -1))]),
    ("G 全对称", [(('(12)', 1)), (('(123)', 1)), (('e', -1))]),
]

print("=" * 70)
print("所有对称类的守恒量分析 (每类采样100个)")
print("=" * 70)

np.random.seed(42)

for name, generators in subgroup_classes:
    orbits, orbit_map = compute_orbits(generators)
    n_orbits = len(orbits)
    group_order = len(generate_group_elements(generators))
    
    # 采样
    results = {}
    for _ in range(600):
        orbit_vals = np.random.choice([-1, 0, 1], n_orbits)
        f_vec = np.zeros(27, dtype=int)
        for idx in range(27):
            i = idx // 9
            j = (idx % 9) // 3
            k = idx % 3
            orb = orbit_map[i, j, k]
            f_vec[idx] = orbit_vals[orb]
        
        f = f_to_array(f_vec)
        M = build_constraint_matrix_L(f, 3)
        rank = np.linalg.matrix_rank(M)
        dim_ker = 81 - rank
        results[dim_ker] = results.get(dim_ker, 0) + 1
    
    # 输出
    thermal = results.get(9, 0)
    non_thermal = 600 - thermal
    print(f"\n{name} (|H|={group_order}, 轨道数={n_orbits}):")
    print(f"  分布: {dict(sorted(results.items()))}")
    print(f"  热化(d=9): {thermal}%, 非热化(d>9): {non_thermal}%")