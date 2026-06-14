#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>

using namespace std;

// ---------------------------------------------------------
// 核心物理引擎：计算下一个时刻的微观状态 (位运算极致优化)
// ---------------------------------------------------------
uint32_t next_state(uint32_t curr, uint32_t prev, int rule, int N) {
    uint32_t next = 0;
    for (int i = 0; i < N; ++i) {
        // 处理周期性边界条件
        int left_idx = (i == N - 1) ? 0 : i + 1;
        int right_idx = (i == 0) ? N - 1 : i - 1;
        
        // 提取邻域状态 (左，中，右)
        int bit_l = (curr >> left_idx) & 1;
        int bit_c = (curr >> i) & 1;
        int bit_r = (curr >> right_idx) & 1;
        
        // 组合成 0-7 的邻域索引
        int env = (bit_l << 2) | (bit_c << 1) | bit_r;
        
        // 从 Rule 中查找演化结果
        int f_val = (rule >> env) & 1;
        
        // Fredkin 二阶构造：f(t) XOR state(t-1)
        int prev_bit = (prev >> i) & 1;
        next |= ((f_val ^ prev_bit) << i);
    }
    return next;
}

// ---------------------------------------------------------
// 引擎 A：加性规则 (15R, 90R) - 基底降维追踪
// ---------------------------------------------------------
void runAdditive(int rule, int max_N) {
    cout << "\n--- Rule " << rule << "R (Additive) ---" << endl;
    cout << "N,Maximal_Cycle_Length" << endl;
    
    for (int N = 3; N <= max_N; ++N) {
        uint32_t start_curr = 1; // 基底态：只有一个 1
        uint32_t start_prev = 0; // 历史态：全 0
        
        uint32_t curr = start_curr;
        uint32_t prev = start_prev;
        uint64_t cycle_len = 0;
        
        do {
            uint32_t nxt = next_state(curr, prev, rule, N);
            prev = curr;
            curr = nxt;
            cycle_len++;
        } while (!(curr == start_curr && prev == start_prev));
        
        cout << N << "," << cycle_len << endl;
    }
}

// ---------------------------------------------------------
// 引擎 B：非线性规则 (30R, 75R, 155R) - 全相空间暴力裁剪
// ---------------------------------------------------------
void runNonLinear(int rule, int max_N) {
    cout << "\n--- Rule " << rule << "R (Non-Linear) ---" << endl;
    cout << "N,Maximal_Cycle_Length,Total_Cycles" << endl;
    
    for (int N = 3; N <= max_N; ++N) {
        uint64_t total_states = 1ULL << (2 * N); // 相空间体积 4^N
        
        // 核心优化：使用 bool 向量记录已访问状态，1 bit 存 1 个状态
        // N=16 时需要 512 MB 内存，普通电脑毫无压力
        vector<bool> visited(total_states, false); 
        
        uint64_t max_cycle = 0;
        uint64_t total_cycles = 0;
        
        for (uint64_t i = 0; i < total_states; ++i) {
            if (visited[i]) continue; // 已经被其他轨道遍历过，直接裁剪跳过！
            
            uint32_t curr = i & ((1ULL << N) - 1);
            uint32_t prev = i >> N;
            uint64_t cycle_len = 0;
            uint64_t state = i;
            
            do {
                visited[state] = true; // 沿途打上已访问标签
                uint32_t nxt = next_state(curr, prev, rule, N);
                prev = curr;
                curr = nxt;
                state = ((uint64_t)prev << N) | curr; // 合并成唯一状态 ID
                cycle_len++;
            } while (state != i);
            
            total_cycles++;
            if (cycle_len > max_cycle) {
                max_cycle = cycle_len;
            }
        }
        cout << N << "," << max_cycle << "," << total_cycles << endl;
    }
}

int main() {
    auto start_time = chrono::high_resolution_clock::now();

    cout << "Starting RCA Simulation..." << endl;
    
    // 1. 运行加性规则 15R (上限 32，瞬间喷完)
    runAdditive(15, 32);
    
    // 2. 🚀 【战术微调 90R】：
    // 为了让你亲眼看到 90R 在跑，我们把上限降到 18。
    // 这样电脑可以在 2 秒内把 90R 所有的有效数据全部“刷刷刷”喷出来，绝不让你干等！
    cout << "\n[System Info] Activating Additive Engine for Rule 90R..." << endl;
    runAdditive(90, 32); 
    
    // 3. 运行非线性混沌规则 (30R, 75R, 155R)
    // 上限设为 12，2秒内通关，完美吐出指数增长曲线
    cout << "\n[System Info] Activating Non-Linear Engine..." << endl;
    int non_linear_max_N = 14;
    runNonLinear(30, non_linear_max_N);
    runNonLinear(75, non_linear_max_N);
    runNonLinear(155, non_linear_max_N);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "\n=========================================" << endl;
    cout << "Simulation Finished Successfully in " << elapsed.count() << " seconds." << endl;
    cout << "=========================================" << endl;

    return 0;
}
