import numpy as np
from itertools import product

# ============================================================
# 1. 基本定义
# ============================================================

states = [-1, 0, 1]          # 状态值
n_states = 3

# ⊗ 运算表: 索引 0→-1, 1→0, 2→1
otimes = np.array([
    [ 1,  0, -1],   # a = -1
    [ 0, -1,  1],   # a =  0
    [-1,  1,  0]    # a =  1
])

def state_to_idx(val):
    """将状态值 -1,0,1 映射到索引 0,1,2"""
    return val + 1

def idx_to_state(idx):
    """将索引 0,1,2 映射到状态值 -1,0,1"""
    return idx - 1

def apply_otimes(a, b):
    """计算 a ⊗ b"""
    return otimes[state_to_idx(a), state_to_idx(b)]

# ============================================================
# 2. f 的索引: f[σ_{i-1}, σ_i, σ_{i+1}] 取值 -1,0,1
#    f 用三维数组表示，索引 0→-1, 1→0, 2→1
# ============================================================

def f_to_array(f_values):
    """
    将 27 个 f 值 (按字典序: (-1,-1,-1), (-1,-1,0), ..., (1,1,1))
    转换为 3×3×3 数组，索引为 [σ_{i-1}+1, σ_i+1, σ_{i+1}+1]
    """
    f = np.zeros((3, 3, 3), dtype=int)
    for idx, val in enumerate(f_values):
        i = idx // 9
        j = (idx % 9) // 3
        k = idx % 3
        f[i, j, k] = val
    return f

def apply_rule(f, s_left, s_self, s_right, r):
    """
    更新一个格点
    f: 3×3×3 数组
    s_left, s_self, s_right, r: -1, 0, 1
    返回: (σ_new, r_new)
    """
    f_out = f[state_to_idx(s_left), state_to_idx(s_self), state_to_idx(s_right)]
    sigma_new = apply_otimes(f_out, r)
    r_new = s_self
    return sigma_new, r_new

# ============================================================
# 3. F 的索引: F[σ_i, r_i, σ_{i+1}, r_{i+1}]
#    共 3^4 = 81 个值
# ============================================================

def F_index(sigma_i, r_i, sigma_next, r_next):
    """
    将 (σ_i, r_i, σ_{i+1}, r_{i+1}) 映射到 0..80 的索引
    顺序: σ_i (最显著), r_i, σ_{i+1}, r_{i+1} (最不显著)
    """
    idx = 0
    for val in [sigma_i, r_i, sigma_next, r_next]:
        idx = idx * 3 + state_to_idx(val)
    return idx

# ============================================================
# 4. 构建约束矩阵 M (729 × 81)
# ============================================================

def build_constraint_matrix(f):
    """
    f: 3×3×3 数组
    返回: M (729 × 81 矩阵), M @ F_vec = 0
    """
    M = np.zeros((729, 81), dtype=float)
    row = 0
    
    # 枚举所有 L=3 初始构型
    for s0, r0, s1, r1, s2, r2 in product(states, repeat=6):
        # 一步更新
        s0_new, r0_new = apply_rule(f, s2, s0, s1, r0)
        s1_new, r1_new = apply_rule(f, s0, s1, s2, r1)
        s2_new, r2_new = apply_rule(f, s1, s2, s0, r2)
        
        # Φ(t+1) - Φ(t) = 0
        # Σ_i F(σ_i', r_i', σ_{i+1}', r_{i+1}') - Σ_i F(σ_i, r_i, σ_{i+1}, r_{i+1}) = 0
        
        # 新构型的正贡献
        M[row, F_index(s0_new, r0_new, s1_new, r1_new)] += 1
        M[row, F_index(s1_new, r1_new, s2_new, r2_new)] += 1
        M[row, F_index(s2_new, r2_new, s0_new, r0_new)] += 1
        
        # 旧构型的负贡献
        M[row, F_index(s0, r0, s1, r1)] -= 1
        M[row, F_index(s1, r1, s2, r2)] -= 1
        M[row, F_index(s2, r2, s0, r0)] -= 1
        
        row += 1
    
    return M

# ============================================================
# 5. 分析给定规则
# ============================================================

def analyze_rule(f_values):
    """分析给定 f 的守恒量零空间"""
    f = f_to_array(f_values)
    M = build_constraint_matrix(f)
    
    # 计算秩和零空间维度
    rank = np.linalg.matrix_rank(M)
    dim_ker = 81 - rank
    
    print(f"矩阵 M: {M.shape[0]} × {M.shape[1]}")
    print(f"秩 = {rank}")
    print(f"零空间维度 = {dim_ker}")
    
    if dim_ker > 1:
        print(f"→ 存在 {dim_ker - 1} 个非平凡可加性守恒量")
    else:
        print("→ 只有平凡（常数）守恒量")
    
    # 计算零空间基
    if dim_ker > 0:
        u, s, vt = np.linalg.svd(M)
        null_space = vt[-dim_ker:]  # 最后 dim_ker 行
        print(f"\n零空间基向量（每个长度 81）:")
        for i, vec in enumerate(null_space):
            print(f"  基向量 {i+1}: 非零元素数 = {np.sum(np.abs(vec) > 1e-10)}")
    
    return dim_ker, rank

# ============================================================
# 6. 示例: 完全对称规则 (G 不变)
# ============================================================

def generate_G_symmetric_f():
    """
    生成在 G = S_3 × Z_2 下完全对称的 f
    f 在 G-轨道上取常值。
    
    G 作用在 {-1,0,1}^3 上有 6 个轨道 (之前算的)。
    但 f 是函数值，对纯置换不变，且对 S 变号。
    """
    # G 在定义域上的 6 个轨道代表:
    # A: (0,0,0)          → f 值任意
    # B: (1,0,0)          → 轨道大小 6
    # C: (1,1,0)          → 轨道大小 6
    # D: (1,-1,0)         → 轨道大小 6
    # E: (1,1,1)          → 轨道大小 2
    # F: (1,1,-1)         → 轨道大小 6
    
    # 全对称约束: f(g·x) = sign(g)·f(x)? 
    # 纯 S_3 部分: f(σ·x) = f(x) (不变)
    # S 部分: f(-x) = -f(x) (变号)
    # 但 f 的输出还要再被 ⊗ 处理，这里先不深入。
    
    # 简化: 让 f 在所有 27 个输入上随机取值
    # （完全随机的 f 属于 H_1 类）
    f_values = np.random.choice([-1, 0, 1], size=27)
    return f_values

# ============================================================
# 7. 运行示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("三元 ERCA 可加性守恒量分析")
    print("=" * 60)
    
    # 测试 1: 完全随机规则 (H1 类)
    print("\n--- 测试 1: 随机规则 (H1 类) ---")
    np.random.seed(42)
    f_random = np.random.choice([-1, 0, 1], size=27)
    analyze_rule(f_random)
    
    # 测试 2: 全零规则
    print("\n--- 测试 2: f ≡ 0 ---")
    f_zero = np.zeros(27, dtype=int)
    analyze_rule(f_zero)
    
    # 测试 3: f ≡ 1
    print("\n--- 测试 3: f ≡ 1 ---")
    f_one = np.ones(27, dtype=int)
    analyze_rule(f_one)