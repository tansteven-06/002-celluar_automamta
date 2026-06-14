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
# 寻找自然基：通过对称性约化
# ============================================================

print("=" * 70)
print("寻找物理上自然的守恒量")
print("=" * 70)

# 使用 f ≡ 1
f_one = f_to_array(np.ones(27, dtype=int))
M = build_constraint_matrix_L(f_one, 3)

# 计算零空间
u, s, vt = np.linalg.svd(M)
rank = np.sum(s > 1e-10)
null_basis = vt[rank:]  # (27, 81)

# 方法1：寻找"最稀疏"的基（通过最小化 L1 范数）
# 使用旋转：对 null_basis 做线性组合，使每个基向量尽可能稀疏

print("\n方法1: 寻找稀疏基")
print("-" * 50)

# 简单方法：对于每个 81 维的标准基向量 e_i，投影到零空间
# 如果投影非零，这就是一个"局域化"的守恒量

# 实际上更好的方法：寻找那些只在少数 F 值上非零的向量
# 尝试所有 81 个坐标方向
for idx in range(81):
    e_i = np.zeros(81)
    e_i[idx] = 1.0
    # 投影到零空间
    proj = e_i @ null_basis.T @ null_basis
    norm = np.linalg.norm(proj)
    if norm > 0.5:  # 投影显著
        nnz = np.sum(np.abs(proj) > 0.01)
        vals = F_decode(idx)
        print(f"F{vals} 投影: 范数={norm:.4f}, 非零元={nnz}")

# 方法2：寻找"局域守恒量"形式
# 尝试形式 F(σ_i, r_i, σ_{i+1}, r_{i+1}) = δ(σ_i=a, r_i=b) 
# 即只在特定输入上为1，其余为0

print("\n方法2: 寻找指示函数型守恒量")
print("-" * 50)

# 构建 9 个试探向量，每个对应一对 (σ_i, r_i)
for si in [-1, 0, 1]:
    for ri in [-1, 0, 1]:
        # 构建向量: F(σ_i, r_i, σ_{i+1}, r_{i+1}) = 1 对所有 σ_{i+1}, r_{i+1}
        # 即 F 只依赖于前两个变量
        test_vec = np.zeros(81)
        for sn in [-1, 0, 1]:
            for rn in [-1, 0, 1]:
                idx = F_index(si, ri, sn, rn)
                test_vec[idx] = 1.0
        
        # 投影到零空间
        proj = test_vec @ null_basis.T @ null_basis
        residual = np.linalg.norm(test_vec - proj)
        if residual < 0.1:
            print(f"F(σ_i={si}, r_i={ri}, *, *) 形式的守恒量存在! 残差={residual:.6f}")

# 方法3：寻找差值形式的守恒量
# 形式 F(σ_i, r_i, σ_{i+1}, r_{i+1}) = φ(σ_i, r_i) - φ(σ_{i+1}, r_{i+1})
# 这种 F 自动满足 ∑ F = 0（边界项相消）

print("\n方法3: 寻找差值形式 φ(σ_i, r_i) - φ(σ_{i+1}, r_{i+1})")
print("-" * 50)

# 构建 9 个基函数
phi_basis = np.zeros((9, 81))
for idx_phi, (s, r) in enumerate(product([-1, 0, 1], repeat=2)):
    for sn in [-1, 0, 1]:
        for rn in [-1, 0, 1]:
            idx = F_index(s, r, sn, rn)
            phi_basis[idx_phi, idx] += 1
            idx2 = F_index(sn, rn, s, r)
            phi_basis[idx_phi, idx2] -= 1

# 检查这些基向量是否在零空间中
for idx_phi in range(9):
    proj = phi_basis[idx_phi] @ null_basis.T @ null_basis
    residual = np.linalg.norm(phi_basis[idx_phi] - proj)
    s, r = list(product([-1, 0, 1], repeat=2))[idx_phi]
    if residual < 0.1:
        print(f"φ({s},{r}) - φ(σ_{{i+1}}, r_{{i+1}}) 形式: 残差={residual:.6f}")
    else:
        print(f"φ({s},{r}): 不在零空间中 (残差={residual:.4f})")

# 方法4：寻找"局部守恒量"满足 J_i - J_{i+1} 形式
# 如果 F(σ_i, r_i, σ_{i+1}, r_{i+1}) = J(σ_i, r_i) - J(σ_{i+1}, r_{i+1})
# 则 ∑ F = 0 自动成立（周期性边界）

print("\n方法4: 寻找一般形式 J(σ_i, r_i) - J(σ_{i+1}, r_{i+1})")
print("-" * 50)

# 构建 9 个 J 基函数
J_basis = np.zeros((9, 81))
for idx_J, (s, r) in enumerate(product([-1, 0, 1], repeat=2)):
    for sn in [-1, 0, 1]:
        for rn in [-1, 0, 1]:
            idx1 = F_index(s, r, sn, rn)
            J_basis[idx_J, idx1] += 1
            idx2 = F_index(sn, rn, s, r)
            J_basis[idx_J, idx2] -= 1

# 检查零空间是否由这些张成
combined = np.vstack([null_basis, J_basis])  # (27+9, 81)
u_c, s_c, vt_c = np.linalg.svd(combined)
rank_null = np.sum(s_c > 1e-10)
print(f"null_basis 的秩: {null_basis.shape[0]}")
print(f"null_basis + J_basis 的秩: {rank_null}")
print(f"J_basis 在零空间中的维数: {null_basis.shape[0] + 9 - rank_null}")