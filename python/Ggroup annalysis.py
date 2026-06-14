import numpy as np
from itertools import product

# ============================================================
# ⊗ 表（新表）
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
# G-不变 f 的参数化
# ============================================================

# 定义域中 27 个点到其轨道的映射
# 先构建 3×3×3 数组，每个位置存储其轨道编号 (0-5)

def get_orbit_map():
    """
    返回 (3,3,3) 数组，值 ∈ {0,1,2,3,4,5} 表示轨道编号
    以及每个轨道的一个代表点
    """
    orbit_map = np.zeros((3, 3, 3), dtype=int)
    
    # 轨道定义（在 S_3 × Z_2 下）
    # 用绝对值模式和符号模式分类
    
    orbit_repr = {}  # 轨道编号 -> 代表点
    
    for a, b, c in product(range(3), repeat=3):
        x = (idx_to_state(a), idx_to_state(b), idx_to_state(c))
        # 计算该点的轨道特征
        abs_vals = sorted([abs(x[0]), abs(x[1]), abs(x[2])])
        n_zeros = abs_vals.count(0)
        n_pos = sum(1 for v in x if v == 1)
        n_neg = sum(1 for v in x if v == -1)
        
        # 根据 (n_zeros, n_pos, n_neg) 分类
        if n_zeros == 3:
            orb = 0  # (0,0,0)
        elif n_zeros == 2:
            orb = 1  # 一个非零
        elif n_zeros == 1:
            if n_pos == 2 or n_neg == 2:
                orb = 2  # 两个同号
            else:
                orb = 3  # 两个异号
        else:  # n_zeros == 0
            if n_pos == 3 or n_neg == 3:
                orb = 4  # 全同号
            else:
                orb = 5  # 两同一异
        
        orbit_map[a, b, c] = orb
        if orb not in orbit_repr:
            orbit_repr[orb] = x
    
    return orbit_map, orbit_repr

def idx_to_state(idx):
    return idx - 1

# ============================================================
# 主分析
# ============================================================

print("=" * 70)
print("G-不变规则的可加性守恒量分析")
print("=" * 70)

# 获取轨道映射
orbit_map, orbit_repr = get_orbit_map()

print("轨道信息:")
for orb in range(6):
    x = orbit_repr[orb]
    print(f"  轨道 {orb}: 代表点 {x}")

# 遍历所有 3^6 = 729 个 G-不变规则
total = 0
thermalizing = 0  # 零空间维 = 9
non_thermalizing = 0  # 零空间维 > 9
results = {}  # 零空间维 -> 计数

# 先创建 f 的查找表
# orbit_assignments: 长度为 6 的数组，每个值 ∈ {-1,0,1}
for orbit_vals in product([-1, 0, 1], repeat=6):
    # 构建 f 的 27 维向量
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
    total += 1
    
    if dim_ker == 9:
        thermalizing += 1
    else:
        non_thermalizing += 1
    
    # 进度显示
    if total % 100 == 0:
        print(f"  进度: {total}/729")

print(f"\n总计: {total} 个 G-不变规则")
print(f"\n零空间维度分布:")
for d in sorted(results.keys()):
    print(f"  d = {d}: {results[d]} 个规则 ({results[d]/total*100:.1f}%)")

print(f"\n热化判断:")
print(f"  可能热化 (d=9): {thermalizing} 个 ({thermalizing/total*100:.1f}%)")
print(f"  非热化 (d>9): {non_thermalizing} 个 ({non_thermalizing/total*100:.1f}%)")

# 分析非热化规则的特征
if non_thermalizing > 0:
    print(f"\n非热化规则的零空间维分布:")
    for d in sorted(results.keys()):
        if d > 9:
            print(f"  d = {d}: {results[d]} 个")

# ============================================================
# 可选：深入分析非热化规则
# ============================================================

if non_thermalizing > 0:
    print("\n" + "=" * 70)
    print("非热化规则特征分析")
    print("=" * 70)
    
    # 找出哪些轨道赋值导致非热化
    # 特别检查常值规则
    const_rules = {
        "全 -1": (-1, -1, -1, -1, -1, -1),
        "全 0": (0, 0, 0, 0, 0, 0),
        "全 1": (1, 1, 1, 1, 1, 1),
    }
    
    for name, vals in const_rules.items():
        f_vec = np.zeros(27, dtype=int)
        for idx in range(27):
            i = idx // 9
            j = (idx % 9) // 3
            k = idx % 3
            orb = orbit_map[i, j, k]
            f_vec[idx] = vals[orb]
        f = f_to_array(f_vec)
        M = build_constraint_matrix_L(f, 3)
        rank = np.linalg.matrix_rank(M)
        dim_ker = 81 - rank
        print(f"  {name}: 零空间维 = {dim_ker}")