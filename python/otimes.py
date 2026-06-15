import numpy as np
from itertools import product

states = [-1, 0, 1]
n = 3

def is_latin_square(table):
    """检查每行每列是否是排列"""
    for i in range(n):
        row = table[i, :]
        col = table[:, i]
        if set(row) != set(states) or set(col) != set(states):
            return False
    return True

def is_symmetric(table):
    """检查是否对称: a⊗b = b⊗a"""
    for i in range(n):
        for j in range(n):
            if table[i, j] != table[j, i]:
                return False
    return True

def satisfies_property(table):
    """检查: (a⊗b)⊗a = b 对所有 a,b"""
    for a_idx, a in enumerate(states):
        for b_idx, b in enumerate(states):
            c = table[a_idx, b_idx]
            c_idx = states.index(c)
            result = table[c_idx, a_idx]
            if result != b:
                return False
    return True

# ============================================================
# 搜索所有 3×3 表格
# ============================================================

print("Searching for all ⊗ tables satisfying:")
print("  1. Latin square (rows and columns are permutations)")
print("  2. Commutative: a⊗b = b⊗a")
print("  3. (a⊗b)⊗a = b for all a,b")
print()

# 穷举所有 3^9 = 19683 种表格
all_tables_comm = []
all_tables_noncomm = []

for flat in product(states, repeat=9):
    table = np.array(flat).reshape(3, 3)
    
    if not is_latin_square(table):
        continue
    
    if satisfies_property(table):
        if is_symmetric(table):
            all_tables_comm.append(table.copy())
        else:
            all_tables_noncomm.append(table.copy())

print(f"Commutative tables satisfying all conditions: {len(all_tables_comm)}")
print(f"Non-commutative tables satisfying (a⊗b)⊗a = b: {len(all_tables_noncomm)}")
print()

# 显示所有交换解
if len(all_tables_comm) > 0:
    print("=" * 60)
    print("COMMUTATIVE SOLUTIONS")
    print("=" * 60)
    for idx, table in enumerate(all_tables_comm):
        print(f"\nTable #{idx+1}:")
        for a in states:
            row = [table[states.index(a), states.index(b)] for b in states]
            print(f"  {a} ⊗ * = {row}")
        print(f"  Matrix form:\n{table}")

# 显示非交换解（如果有）
if len(all_tables_noncomm) > 0:
    print("\n" + "=" * 60)
    print("NON-COMMUTATIVE SOLUTIONS")
    print("=" * 60)
    for idx, table in enumerate(all_tables_noncomm[:5]):  # 只显示前5个
        print(f"\nTable #{idx+1}:")
        for a in states:
            row = [table[states.index(a), states.index(b)] for b in states]
            print(f"  {a} ⊗ * = {row}")
        print(f"  Matrix form:\n{table}")
    if len(all_tables_noncomm) > 5:
        print(f"\n  ... and {len(all_tables_noncomm) - 5} more")

# 如果没有解，说明原因
if len(all_tables_comm) == 0 and len(all_tables_noncomm) == 0:
    print("No tables satisfy (a⊗b)⊗a = b with Latin square property.")
    print("\nReasoning:")
    print("  For a finite quasigroup of order 3, the condition (a⊗b)⊗a = b")
    print("  implies a = (a⊗b) \ b (right division) and a = b / (a⊗b) (left division)")
    print("  This forces the quasigroup to be medial or isotopic to an abelian group.")
    print("  Let's check which group it corresponds to...")
    
    # Check if it corresponds to Z_3
    z3_table = np.array([
        [0, 1, 2],  # a=-1?
        [1, 2, 0],
        [2, 0, 1]
    ])  # This is addition mod 3
    print(f"\n  Z_3 addition table (after mapping to {-1,0,1}):")
    # Map 0->0, 1->1, 2->-1?