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
# 构建边界项空间
# ============================================================

# 9 个 J 基函数: J(σ_i, r_i) = δ(σ_i=a, r_i=b)
J_basis = np.zeros((9, 81))
for idx_J, (a, b) in enumerate(product([-1, 0, 1], repeat=2)):
    for sn in [-1, 0, 1]:
        for rn in [-1, 0, 1]:
            idx1 = F_index(a, b, sn, rn)
            J_basis[idx_J, idx1] += 1  # F = J(σ_i, r_i)
            idx2 = F_index(sn, rn, a, b)
            J_basis[idx_J, idx2] -= 1  # F = -J(σ_{i+1}, r_{i+1})

# 加上常数向量
const_vec = np.ones(81).reshape(1, -1)
boundary_space = np.vstack([const_vec, J_basis])  # (10, 81)
u_b, s_b, vt_b = np.linalg.svd(boundary_space)
rank_b = np.sum(s_b > 1e-10)
print(f"边界项+常数空间的秩: {rank_b}")

# ============================================================
# 比较随机规则零空间与边界项空间
# ============================================================

print("\n" + "=" * 70)
print("随机规则零空间 vs 边界项空间")
print("=" * 70)

for seed in [42, 123, 456, 789]:
    rng = np.random.RandomState(seed)
    f_rand = rng.choice([-1, 0, 1], 27)
    f = f_to_array(f_rand)
    M = build_constraint_matrix_L(f, 3)
    u, s, vt = np.linalg.svd(M)
    rank = np.sum(s > 1e-10)
    null = vt[rank:]  # (dim_ker, 81)
    
    # 检查 null 是否包含在 boundary_space 中
    combined = np.vstack([null, boundary_space])
    u_c, s_c, vt_c = np.linalg.svd(combined)
    rank_combined = np.sum(s_c > 1e-10)
    
    # 如果 rank_combined == rank_b，则 null ⊆ boundary_space
    print(f"\n随机规则 (seed={seed}):")
    print(f"  零空间维 = {null.shape[0]}")
    print(f"  null ∪ boundary 的秩 = {rank_combined}")
    print(f"  边界空间秩 = {rank_b}")
    if rank_combined == rank_b:
        print(f"  ✓ 零空间完全包含于边界项空间!")
    else:
        print(f"  ✗ 零空间超出边界项空间 {rank_combined - rank_b} 维")