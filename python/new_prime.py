import numpy as np
from itertools import product

# ============================================================
# 新 ⊗ 表
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
# 分析哪些 J 基函数在随机规则的零空间中
# ============================================================

print("=" * 70)
print("边界项分析：哪些 J 基函数是守恒量？")
print("=" * 70)

# 构建 9 个 J 基函数
J_basis = np.zeros((9, 81))
J_labels = []
for idx_J, (a, b) in enumerate(product([-1, 0, 1], repeat=2)):
    J_labels.append(f"J({a},{b})")
    for sn in [-1, 0, 1]:
        for rn in [-1, 0, 1]:
            idx1 = F_index(a, b, sn, rn)
            J_basis[idx_J, idx1] += 1
            idx2 = F_index(sn, rn, a, b)
            J_basis[idx_J, idx2] -= 1

const_vec = np.ones(81).reshape(1, -1)

# 对每个随机规则，检查 J 基函数是否在零空间中
for seed in [42, 123]:
    rng = np.random.RandomState(seed)
    f_rand = rng.choice([-1, 0, 1], 27)
    f = f_to_array(f_rand)
    M = build_constraint_matrix_L(f, 3)
    u, s, vt = np.linalg.svd(M)
    rank = np.sum(s > 1e-10)
    null = vt[rank:]
    
    print(f"\n随机规则 (seed={seed}): 零空间维 = {null.shape[0]}")
    
    # 检查每个 J 基向量
    for i in range(9):
        proj = J_basis[i] @ null.T @ null
        residual = np.linalg.norm(J_basis[i] - proj)
        in_null = residual < 1e-10
        print(f"  {J_labels[i]}: 在零空间中? {'✓' if in_null else '✗'} (残差={residual:.2e})")
    
    # 检查常数向量
    proj_const = const_vec @ null.T @ null
    residual_const = np.linalg.norm(const_vec - proj_const)
    print(f"  常数: 在零空间中? {'✓' if residual_const < 1e-10 else '✗'} (残差={residual_const:.2e})")

# ============================================================
# 寻找哪些 J 的组合在零空间中
# ============================================================

print("\n" + "=" * 70)
print("寻找零空间中 J 基函数的线性组合")
print("=" * 70)

# 用4个随机规则的公共零空间
M_joint = []
for seed in [42, 123, 456, 789]:
    rng = np.random.RandomState(seed)
    f_rand = rng.choice([-1, 0, 1], 27)
    f = f_to_array(f_rand)
    M = build_constraint_matrix_L(f, 3)
    M_joint.append(M)

M_combined = np.vstack(M_joint)
u_comb, s_comb, vt_comb = np.linalg.svd(M_combined)
rank_comb = np.sum(s_comb > 1e-10)
null_common = vt_comb[rank_comb:]

print(f"公共零空间维: {null_common.shape[0]}")

# 将 J_basis 和 const_vec 投影到公共零空间
combined = np.vstack([null_common, const_vec, J_basis])
u_c, s_c, vt_c = np.linalg.svd(combined)
rank_c = np.sum(s_c > 1e-10)
print(f"null_common + [const, J1..J9] 的秩: {rank_c}")
print(f"null_common 的秩: {null_common.shape[0]}")
print(f"因此在 null_common 中，const+J_basis 贡献了 {rank_c - null_common.shape[0]} 维新方向")

# 找出哪些 J 基的线性组合在公共零空间中
# 将每个 J 基投影到 null_common
print("\n各 J 基在公共零空间中的投影:")
for i in range(9):
    proj = J_basis[i] @ null_common.T @ null_common
    residual = np.linalg.norm(J_basis[i] - proj)
    in_null = residual < 1e-10
    print(f"  {J_labels[i]}: {'✓' if in_null else '✗'} (残差={residual:.2e}, 投影范数={np.linalg.norm(proj):.4f})")

# 检查常数
proj_const = const_vec @ null_common.T @ null_common
print(f"  常数: {'✓' if np.linalg.norm(const_vec - proj_const) < 1e-10 else '✗'} (残差={np.linalg.norm(const_vec - proj_const):.2e})")