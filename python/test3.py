import numpy as np
from itertools import product
from collections import Counter
import time

# ============================================================
# 三元 ERCA 核心引擎
# ============================================================

# ⊗ 运算表 (Table #1, 新表)
# 索引 0→-1, 1→0, 2→1
OTIMES = np.array([
    [-1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  1]
], dtype=int)

# d=9 混沌规则: T01101001T101010010110T0001
RULE_D9 = np.array([
    -1, 0, 1, 1, 0, 1, 0, 0, 1,   # σ_{i-1}=1
    -1, 1, 0, 1, 0, 1, 0, 0, 1,   # σ_{i-1}=0
     0, 1, 1, 0,-1, 0, 0, 0, 1    # σ_{i-1}=-1
], dtype=int)

# d=27 周期规则: f ≡ 1
RULE_D27 = np.ones(27, dtype=int)

def state_to_idx(val):
    """-1,0,1 → 0,1,2"""
    return val + 1

def apply_otimes(a, b):
    return OTIMES[state_to_idx(a), state_to_idx(b)]

def f_to_array(f_values):
    """27-element f → 3×3×3 array indexed by [σ_{i-1}+1, σ_i+1, σ_{i+1}+1]"""
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

def config_to_idx(config, L):
    """Configuration → unique integer ID (base-3 encoding)"""
    idx = 0
    for val in config:
        idx = idx * 3 + (val + 1)
    return idx

# ============================================================
# 轨道分析引擎：遍历所有状态，追踪轨道长度
# ============================================================

def compute_all_orbits(f_rule, L):
    """
    遍历相空间中所有 3^{2L} 个状态，
    计算每条轨道的长度，返回轨道长度列表。
    
    使用 visited 数组裁剪已访问状态。
    """
    f = f_to_array(f_rule)
    states = [-1, 0, 1]
    n_states = 3 ** (2 * L)
    
    visited = np.zeros(n_states, dtype=bool)
    orbit_lengths = []
    
    # 遍历所有初始构型
    for config in product(states, repeat=2*L):
        start_idx = config_to_idx(config, L)
        if visited[start_idx]:
            continue
        
        # 追踪轨道
        current = list(config)
        length = 0
        
        while True:
            current_idx = config_to_idx(tuple(current), L)
            if visited[current_idx]:
                break
            
            visited[current_idx] = True
            length += 1
            
            # 一步演化
            sigmas = current[0::2]  # 偶数位置: σ_i
            rs = current[1::2]      # 奇数位置: r_i
            
            new_config = [0] * (2 * L)
            for i in range(L):
                s_left = sigmas[(i - 1) % L]
                s_self = sigmas[i]
                s_right = sigmas[(i + 1) % L]
                new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                new_config[2 * i] = new_s
                new_config[2 * i + 1] = new_r
            
            current = new_config
            
            # 安全检查
            if length > 1000000:
                print(f"  Warning: orbit exceeded 1M steps at L={L}")
                break
        
        orbit_lengths.append(length)
    
    return orbit_lengths

# ============================================================
# 批量分析：不同 L 下的轨道统计
# ============================================================

def analyze_rule(rule_values, rule_name, L_max_dense=5, L_max_sparse=10):
    """
    分析给定规则的轨道结构。
    L_max_dense: 完整遍历的最大 L
    L_max_sparse: 采样分析的最大 L
    """
    print(f"\n{'='*70}")
    print(f"Rule: {rule_name}")
    print(f"{'='*70}")
    print(f"{'L':<6} {'States':<12} {'Orbits':<10} {'Max T':<10} {'Mean T':<10}")
    print(f"{'-'*48}")
    
    for L in range(3, L_max_dense + 1):
        n_states = 3 ** (2 * L)
        start = time.time()
        orbit_lengths = compute_all_orbits(rule_values, L)
        elapsed = time.time() - start
        
        orbits_arr = np.array(orbit_lengths)
        print(f"{L:<6} {n_states:<12} {len(orbits_arr):<10} "
              f"{np.max(orbits_arr):<10} {np.mean(orbits_arr):<10.1f} "
              f"({elapsed:.1f}s)")
    
    # 大 L 采样
    if L_max_sparse > L_max_dense:
        print(f"\n{'='*70}")
        print(f"Large-L sampling (random initial states):")
        print(f"{'='*70}")
        print(f"{'L':<6} {'States':<14} {'Samples':<10} {'Mean T (sample)':<15}")
        print(f"{'-'*48}")
        
        for L in range(L_max_dense + 1, L_max_sparse + 1):
            n_states = 3 ** (2 * L)
            n_samples = min(500, n_states)
            
            # 采样轨道
            sampled_lengths = []
            states_list = [-1, 0, 1]
            f = f_to_array(rule_values)
            
            for _ in range(n_samples):
                config = list(np.random.choice(states_list, 2 * L))
                start_config = tuple(config)
                visited = {}
                step = 0
                
                while tuple(config) not in visited:
                    visited[tuple(config)] = step
                    step += 1
                    
                    sigmas = config[0::2]
                    rs = config[1::2]
                    new_config = [0] * (2 * L)
                    for i in range(L):
                        s_left = sigmas[(i - 1) % L]
                        s_self = sigmas[i]
                        s_right = sigmas[(i + 1) % L]
                        new_s, new_r = apply_rule(f, s_left, s_self, s_right, rs[i])
                        new_config[2 * i] = new_s
                        new_config[2 * i + 1] = new_r
                    config = new_config
                    
                    if step > 50000:
                        break
                
                if tuple(config) in visited:
                    sampled_lengths.append(step - visited[tuple(config)])
                else:
                    sampled_lengths.append(step)
            
            sampled_arr = np.array(sampled_lengths)
            print(f"{L:<6} {n_states:<14} {n_samples:<10} {np.mean(sampled_arr):<15.1f}")
    
    return

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TERNARY ERCA - ORBIT STRUCTURE ANALYSIS")
    print("=" * 70)
    print(f"⊗ table (Table #1, Z₃ addition):")
    for a in [-1, 0, 1]:
        row = [apply_otimes(a, b) for b in [-1, 0, 1]]
        print(f"  {a} ⊗ * = {row}")
    
    # 分析 d=9 混沌规则
    analyze_rule(RULE_D9, "d=9 (CHAOTIC)", L_max_dense=5, L_max_sparse=8)
    
    # 分析 d=27 周期规则
    analyze_rule(RULE_D27, "d=27 (PERIODIC, f≡1)", L_max_dense=5, L_max_sparse=8)
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")