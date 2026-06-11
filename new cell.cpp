// 周期性边界的三元ERCA
#include<cstdio>
#include<ctime>
#include<cstring>
#include<cstdlib>

using namespace std;

int X[3][3]={{1,0,-1},{0,-1,1},{-1,1,0}};
int f[3][3][3];

int p[1000][1000];
int r[1000][1000];

int F(int s0,int s1,int s2){
    return f[s0+1][s1+1][s2+1];
}

int XOR(int a,int b){
    return X[a+1][b+1];
}

int main(int argc, char* argv[]){
    int total_steps = 40;
    int seed = 0;
    int SPACE_SIZE = 100;
    
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-t") == 0 && i+1 < argc)
            total_steps = atoi(argv[++i]);
        else if(strcmp(argv[i], "-s") == 0 && i+1 < argc)
            SPACE_SIZE = atoi(argv[++i]);
        else if(strcmp(argv[i], "-r") == 0 && i+1 < argc)
            seed = atoi(argv[++i]);
    }
    
    if(total_steps < 1) total_steps = 1;
    if(total_steps > 500) total_steps = 500;
    if(SPACE_SIZE < 3) SPACE_SIZE = 3;
    if(SPACE_SIZE > 900) SPACE_SIZE = 900;
    
    if(seed == 0) {
        srand(time(0));
    } else {
        srand(seed);
    }
    
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                f[i][j][k]=rand()%3-1;
            }
        }
    }
    
    printf("rule:\n");
    for(int i=1;i>=-1;i--){
        for(int j=1;j>=-1;j--){
            for(int k=1;k>=-1;k--){
                F(i,j,k)<0?printf("T"):printf("%d",F(i,j,k));
            }
        }
    }
    printf("\n");
    
    // 初始条件：中心为1
    //int center = SPACE_SIZE / 2;
    p[1][0]=1;
    
    // ===== 正向演化 =====
    int t;
    for(t=0;t<=total_steps;t++){
        // 计算下一时刻
        for(int i=0;i<SPACE_SIZE;i++){
            // 周期边界：环状结构
            int left = (i-1+SPACE_SIZE) % SPACE_SIZE;
            int right = (i+1) % SPACE_SIZE;
            
            p[i][t+1]=XOR(F(p[left][t],p[i][t],p[right][t]),r[i][t]);
            r[i][t+1]=p[i][t];
        }
        
        // 打印当前时刻
        for(int i=0;i<SPACE_SIZE;i++){
            p[i][t]<0?printf("T "):printf("%d ",p[i][t]);
        }
        printf("\n");
    }
    
    t--;
    printf("--time reverse--\n");
    
    // ===== 逆向初始化 =====
    for(int i=0;i<SPACE_SIZE;i++){
        r[i][t]=p[i][t+1];
    }
    
    // ===== 逆向演化 =====
    for(;t>=0;t--){
        // 计算前一步
        for(int i=0;i<SPACE_SIZE;i++){
            int left = (i-1+SPACE_SIZE) % SPACE_SIZE;
            int right = (i+1) % SPACE_SIZE;
            
            p[i][t-1]=XOR(F(p[left][t],p[i][t],p[right][t]),r[i][t]);
            r[i][t-1]=p[i][t];
        }
        
        // 打印
        for(int i=0;i<SPACE_SIZE;i++){
            p[i][t]<0?printf("T "):printf("%d ",p[i][t]);
        }
        printf("\n");
    }
    
    // ===== 验证可逆性 =====
    bool ok = true;
    for(int i=0;i<SPACE_SIZE;i++){
        int expected = (i == 1) ? 1 : 0;
        if(p[i][0] != expected){
            printf("Mismatch at %d: got %d, expected %d\n", i, p[i][0], expected);
            ok = false;
            break;
        }
    }
    printf("Reversible: %s\n", ok ? "YES" : "NO");
    
    return 0;
}