#include<cstdio>
#include<ctime>
#include<algorithm>

using namespace std;

int X[3][3]={{-1,1,0},{1,0,-1},{0,-1,1}};
int f[3][3][3];

int p[100][100],r[100][100];

int F(int s0,int s1,int s2){
    return f[s0+1][s1+1][s2+1];

}

int XOR(int a,int b){
    return X[a+1][b+1];
}

int main(){
    srand(time(0));
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                f[i][j][k]=rand()%3-1;
            }

        }
    }
    printf("rule:\n");
    int R=0;
    for(int i=1;i>=-1;i--){
        for(int j=1;j>=-1;j--){
            for(int k=1;k>-1;k--){
                F(i,j,k)<0?printf("T"):printf("%d",F(i,j,k));
            }

        }
    }
    printf("\n");
    p[50][0]=1;
    int t;
    for(t=0;t<=40;t++){
       for(int i=1;i<99;i++){
            p[i][t+1]=XOR(F(p[i-1][t],p[i][t],p[i+1][t]),r[i][t]);
            r[i][t+1]=p[i][t];
            p[i][t]<0?printf("T "):printf("%d ",p[i][t]);
       }
        printf("\n");
    }
    t--;
    printf("--time reverse--\n");
    for(int i=1;i<=100;i++){
        r[i][t]=p[i][t+1];
    }
    for(;t>=0;t--){
        
        for(int i=1;i<99;i++){
            p[i][t-1]=XOR(F(p[i-1][t],p[i][t],p[i+1][t]),r[i][t]);
            r[i][t-1]=p[i][t];
            p[i][t]<0?printf("T "):printf("%d ",p[i][t]);
       }
        printf("\n");

    }
    return 0;
}
