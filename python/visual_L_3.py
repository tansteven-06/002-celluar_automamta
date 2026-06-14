import numpy as np
from itertools import product

# ============================================================
# 基本定义
# ============================================================
states = [-1, 0, 1]
otimes = np.array([
    [ 1,  0, -1],
    [ 0, -1,  1],
    [-1,  1,  0]
])

def state_to_idx(val):
    return val + 1

def idx_to_state(idx):
    return idx - 1

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

def F_decode(idx):
    """将索引解码为 (σ_i, r_i, σ_{i+1}, r_{i+1})"""
    vals = []
    for _ in range(4):
        vals.append(idx_to_state(idx % 3))
        idx //= 3
    return tuple(reversed(vals))

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
# 零空间详细分析
# ============================================================

def analyze_nullspace_detailed(f_values, L=3):
    """深度分析零空间结构"""
    f = f_to_array(f_values)
    M = build_constraint_matrix_L(f, L)
    
    # SVD
    u, s, vt = np.linalg.svd(M)
    
    # 数值秩
    tol = 1e-10
    rank = np.sum(s > tol)
    dim_ker = 81 - rank
    
    print(f"矩阵 M ({3**(2*L)} × 81):")
    print(f"  数值秩 = {rank}")
    print(f"  零空间维 = {dim_ker}")
    print(f"  奇异值范围: {s[0]:.2e} 到 {s[-1]:.2e}")
    
    # 提取零空间
    null_basis = vt[rank:]  # (dim_ker, 81)
    
    # 找出常数基向量
    const_vec = np.ones(81) / np.sqrt(81)
    overlaps = np.abs(null_basis @ const_vec)
    const_idx = np.argmax(overlaps)
    
    print(f"\n常数基向量: 索引 {const_idx}, 重叠 = {overlaps[const_idx]:.6f}")
    
    # 分析非平凡基向量
    print("\n" + "=" * 70)
    print("非平凡守恒量分析")
    print("=" * 70)
    
    for i, vec in enumerate(null_basis):
        if i == const_idx:
            continue
        
        F_reshaped = vec.reshape(3, 3, 3, 3)
        # F[σ_i, r_i, σ_{i+1}, r_{i+1}]
        
        print(f"\n--- 基向量 {i+1} ---")
        
        # 检查对称性
        # 1. 是否只依赖于 σ_i？
        dep_sigma_i = np.var([np.mean(F_reshaped[si, :, :, :]) for si in range(3)])
        dep_r_i = np.var([np.mean(F_reshaped[:, ri, :, :]) for ri in range(3)])
        dep_sigma_next = np.var([np.mean(F_reshaped[:, :, sj, :]) for sj in range(3)])
        dep_r_next = np.var([np.mean(F_reshaped[:, :, :, rj]) for rj in range(3)])
        
        print(f"  对 σ_i 依赖性: {dep_sigma_i:.6f}")
        print(f"  对 r_i 依赖性: {dep_r_i:.6f}")
        print(f"  对 σ_{i+1} 依赖性: {dep_sigma_next:.6f}")
        print(f"  对 r_{i+1} 依赖性: {dep_r_next:.6f}")
        
        # 2. 是否可分解为 F(σ_i, r_i) + G(σ_{i+1}, r_{i+1})？
        # 检查: F(a,b,c,d) - F(a,b,c',d') - F(a',b',c,d) + F(a',b',c',d') ≈ 0
        # 对所有 a,b,c,d, a',b',c',d' 的均值
        cross_term = 0
        count = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        for ap in range(3):
                            for bp in range(3):
                                for cp in range(3):
                                    for dp in range(3):
                                        val = (F_reshaped[a,b,c,d] - F_reshaped[a,b,cp,dp] 
                                               - F_reshaped[ap,bp,c,d] + F_reshaped[ap,bp,cp,dp])
                                        cross_term += val**2
                                        count += 1
        cross_term = np.sqrt(cross_term / count)
        print(f"  可分解性检验 (越小越可分解): {cross_term:.6f}")
        
        # 3. 均值
        print(f"  均值: {np.mean(F_reshaped):.6f}")
        
        # 4. 最小值和最大值
        print(f"  值域: [{np.min(F_reshaped):.4f}, {np.max(F_reshaped):.4f}]")

# ============================================================
# 运行分析
# ============================================================

print("=" * 70)
print("f ≡ 1 零空间深度分析")
print("=" * 70)

f_one = np.ones(27, dtype=int)
analyze_nullspace_detailed(f_one, L=3)