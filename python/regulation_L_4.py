import numpy as np
from itertools import product

# ============================================================
# 基本定义（同前）
# ============================================================
states = [-1, 0, 1]
otimes = np.array([
    [ 1,  0, -1],
    [ 0, -1,  1],
    [-1,  1,  0]
])

def state_to_idx(val):
    return val + 1

def apply_otimes(a, b):
    return otimes[state_to_idx(a), state_to_idx(b)]

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

# ============================================================
# L=4 的约束矩阵构建
# ============================================================

def build_constraint_matrix_L(f, L):
    """
    对长度 L 的环构建约束矩阵
    f: 3×3×3 数组
    L: 环长度
    返回: M (3^(2L) × 81)
    """
    n_configs = 3**(2*L)
    M = np.zeros((n_configs, 81), dtype=float)
    row = 0
    
    # 枚举所有初始构型
    for config in product(states, repeat=2*L):
        # 解析 σ 和 r
        sigmas = list(config[0::2])  # 偶数索引
        rs = list(config[1::2])      # 奇数索引
        
        # 一步更新
        sigmas_new = [0] * L
        rs_new = [0] * L
        for i in range(L):
            s_left = sigmas[(i-1) % L]
            s_self = sigmas[i]
            s_right = sigmas[(i+1) % L]
            sigmas_new[i], rs_new[i] = apply_rule(f, s_left, s_self, s_right, rs[i])
        
        # Φ(t+1) - Φ(t) = 0
        for i in range(L):
            i_next = (i+1) % L
            # 新构型贡献
            M[row, F_index(sigmas_new[i], rs_new[i], 
                          sigmas_new[i_next], rs_new[i_next])] += 1
            # 旧构型贡献
            M[row, F_index(sigmas[i], rs[i], 
                          sigmas[i_next], rs[i_next])] -= 1
        
        row += 1
    
    return M

def analyze_rule_L(f_values, L_values=[3, 4, 5]):
    """分析不同 L 下的守恒量"""
    f = f_to_array(f_values)
    
    print(f"{'L':<6} {'配置数':<12} {'秩':<8} {'零空间维':<10} {'非平凡守恒量':<12}")
    print("-" * 50)
    
    for L in L_values:
        M = build_constraint_matrix_L(f, L)
        rank = np.linalg.matrix_rank(M)
        dim_ker = 81 - rank
        nontrivial = dim_ker - 1  # 减去常数守恒量
        
        print(f"{L:<6} {3**(2*L):<12} {rank:<8} {dim_ker:<10} {nontrivial:<12}")
    
    return

# ============================================================
# 测试: f ≡ 1
# ============================================================

print("=" * 60)
print("f ≡ 1 在不同环长度下的守恒量分析")
print("=" * 60)

f_one = np.ones(27, dtype=int)
analyze_rule_L(f_one, L_values=[3, 4])

# 如果 L=4 的 3^8 = 6561 太大，可能需要优化