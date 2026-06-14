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
# 1. 完整测试所有规则类型
# ============================================================
print("=" * 70)
print("新 ⊗ 表：完整守恒量分析")
print("=" * 70)

# 常值规则
print("\n--- 常值规则 ---")
for name, f_vals in [("f ≡ 1", np.ones(27, dtype=int)),
                      ("f ≡ 0", np.zeros(27, dtype=int)),
                      ("f ≡ -1", -np.ones(27, dtype=int))]:
    f = f_to_array(f_vals)
    for L in [3, 4]:
        M = build_constraint_matrix_L(f, L)
        rank = np.linalg.matrix_rank(M)
        dim_ker = 81 - rank
        print(f"  {name} (L={L}): 秩={rank}, 零空间维={dim_ker}, 非平凡={dim_ker-1}")

# 随机规则
print("\n--- 随机规则 ---")
random_results = []
for seed in [42, 123, 456, 789, 111, 222, 333, 444]:
    rng = np.random.RandomState(seed)
    f_rand = rng.choice([-1, 0, 1], 27)
    f = f_to_array(f_rand)
    M3 = build_constraint_matrix_L(f, 3)
    M4 = build_constraint_matrix_L(f, 4)
    rank3 = np.linalg.matrix_rank(M3)
    rank4 = np.linalg.matrix_rank(M4)
    dim3 = 81 - rank3
    dim4 = 81 - rank4
    random_results.append((seed, dim3, dim4))
    print(f"  seed={seed}: L=3: {dim3}维, L=4: {dim4}维")

# 统计随机规则的零空间维度分布
from collections import Counter
dim_counter = Counter([d3 for _, d3, _ in random_results])
print(f"\n  零空间维度分布 (L=3): {dict(dim_counter)}")

# ============================================================
# 2. 寻找所有规则的公共守恒量
# ============================================================
print("\n" + "=" * 70)
print("公共守恒量分析（4个随机规则的交集）")
print("=" * 70)

# 构建4个随机规则的联合约束矩阵
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
dim_ker_comb = 81 - rank_comb

print(f"联合约束矩阵: {M_combined.shape[0]} × 81")
print(f"公共零空间维: {dim_ker_comb}")

# ============================================================
# 3. 分析公共零空间的物理结构
# ============================================================
if dim_ker_comb > 0:
    null_common = vt_comb[rank_comb:]
    
    print(f"\n公共零空间基向量分析:")
    
    # 检查每个基向量是否可写成 J(σ_i, r_i) - J(σ_{i+1}, r_{i+1})
    J_basis = np.zeros((9, 81))
    for idx_J, (a, b) in enumerate(product([-1, 0, 1], repeat=2)):
        for sn in [-1, 0, 1]:
            for rn in [-1, 0, 1]:
                idx1 = F_index(a, b, sn, rn)
                J_basis[idx_J, idx1] += 1
                idx2 = F_index(sn, rn, a, b)
                J_basis[idx_J, idx2] -= 1
    
    const_vec = np.ones(81).reshape(1, -1)
    boundary_space = np.vstack([const_vec, J_basis])
    
    # 检查公共零空间是否包含于边界项空间
    combined = np.vstack([null_common, boundary_space])
    u_c, s_c, vt_c = np.linalg.svd(combined)
    rank_c = np.sum(s_c > 1e-10)
    
    print(f"  公共零空间维 = {null_common.shape[0]}")
    print(f"  边界项空间维 = 10")
    print(f"  联合秩 = {rank_c}")
    
    if rank_c == 10:
        print(f"  ✓ 公共零空间 = 边界项空间（无非平凡动力学守恒量）")
    else:
        extra = null_common.shape[0] + 10 - rank_c
        print(f"  → 存在 {extra} 维额外公共守恒量!")
        
        # 提取额外守恒量
        # 从 null_common 中移除 boundary_space 的分量
        # 使用正交投影
        u_b, s_b, vt_b = np.linalg.svd(boundary_space)
        boundary_ortho = vt_b[10:]  # 正交补空间
        
        # 将 null_common 投影到 boundary_space 的正交补上
        extra_conserved = null_common @ boundary_ortho.T @ boundary_ortho
        
        print(f"\n  额外守恒量的物理结构:")
        for i in range(extra_conserved.shape[0]):
            vec = extra_conserved[i]
            if np.linalg.norm(vec) < 1e-10:
                continue
            F_reshaped = vec.reshape(3, 3, 3, 3)
            print(f"\n  额外守恒量 {i+1}:")
            print(f"    值域: [{np.min(F_reshaped):.4f}, {np.max(F_reshaped):.4f}]")
            print(f"    均值: {np.mean(F_reshaped):.6f}")
            
            # 分析依赖关系
            dep_si = np.std([np.mean(F_reshaped[si,:,:,:]) for si in range(3)])
            dep_ri = np.std([np.mean(F_reshaped[:,ri,:,:]) for ri in range(3)])
            dep_sn = np.std([np.mean(F_reshaped[:,:,sn,:]) for sn in range(3)])
            dep_rn = np.std([np.mean(F_reshaped[:,:,:,rn]) for rn in range(3)])
            print(f"    依赖: σ_i={dep_si:.4f}, r_i={dep_ri:.4f}, σ_i+1={dep_sn:.4f}, r_i+1={dep_rn:.4f}")

# ============================================================
# 4. f ≡ 1 的额外守恒量深度分析
# ============================================================
print("\n" + "=" * 70)
print("f ≡ 1 额外守恒量分析")
print("=" * 70)

f_one = f_to_array(np.ones(27, dtype=int))
M = build_constraint_matrix_L(f_one, 3)
u, s, vt = np.linalg.svd(M)
rank = np.sum(s > 1e-10)
null_one = vt[rank:]

# 从零空间中移除边界项空间
combined = np.vstack([null_one, boundary_space])
u_c, s_c, vt_c = np.linalg.svd(combined)
rank_c = np.sum(s_c > 1e-10)
extra_dim = null_one.shape[0] + 10 - rank_c

print(f"f ≡ 1 零空间维: {null_one.shape[0]}")
print(f"边界项空间维: 10")
print(f"联合秩: {rank_c}")
print(f"额外守恒量维数: {extra_dim}")

# 提取额外守恒量
if extra_dim > 0:
    # 使用正交投影
    u_b, s_b, vt_b = np.linalg.svd(boundary_space)
    boundary_ortho = vt_b[10:]  # 正交补的基
    
    # 将 null_one 的基投影到正交补上
    null_proj = null_one @ boundary_ortho.T @ boundary_ortho
    
    # SVD 找到非零的额外方向
    u_e, s_e, vt_e = np.linalg.svd(null_proj)
    extra_rank = np.sum(s_e > 1e-10)
    extra_basis = vt_e[:extra_rank]
    
    print(f"\n额外守恒量基向量数: {extra_rank}")
    
    # 尝试寻找简单形式
    print("\n寻找简单形式的额外守恒量:")
    
    # 试探 1: F(σ_i, r_i, σ_{i+1}, r_{i+1}) = φ(σ_i, r_i)（只依赖前两个变量）
    print("\n试探1: 只依赖 (σ_i, r_i) 的守恒量:")
    for a, b in product([-1,0,1], repeat=2):
        test_vec = np.zeros(81)
        for sn, rn in product([-1,0,1], repeat=2):
            test_vec[F_index(a, b, sn, rn)] = 1.0
        # 投影到额外空间
        proj = test_vec @ extra_basis.T @ extra_basis
        residual = np.linalg.norm(test_vec - proj)
        if residual < 0.1 and np.linalg.norm(proj) > 0.1:
            print(f"  φ({a},{b}) 形式: 残差={residual:.6f} ✓")
    
    # 试探 2: 是否由 9 个局域守恒量张成（每个格点的某种局域量守恒）
    print("\n试探2: 局域守恒量 φ(σ_i, r_i) 是否在演化下守恒?")
    for a, b in product([-1,0,1], repeat=2):
        # 演化: (σ,r) → (1⊗r, σ)
        sigma_new = apply_otimes(1, b)
        r_new = a
        if (sigma_new, r_new) == (a, b):
            print(f"  ({a},{b}) 是不动点!")
    
    # 试探 3: 周期分析
    print("\n试探3: f ≡ 1 的局域轨道分析:")
    for a, b in product([-1,0,1], repeat=2):
        visited = []
        current = (a, b)
        for step in range(10):
            if current in visited:
                cycle_start = visited.index(current)
                cycle_len = len(visited) - cycle_start
                if cycle_len > 1:
                    print(f"  初态 ({a},{b}): 周期={cycle_len}, 轨道={visited[cycle_start:]}")
                break
            visited.append(current)
            s, r = current
            sigma_new = apply_otimes(1, r)
            r_new = s
            current = (sigma_new, r_new)