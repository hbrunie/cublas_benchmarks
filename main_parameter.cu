#include <stdlib>
#include <helper_functions.h>
#include "gemm_cublas_functions.h"

int main(int argc, char * argv[]){
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -size=matrixSquaredRootSize\n");
        exit(EXIT_SUCCESS);
    }
    //Matrix size (width == height)
    int N;
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        N = std::max(getCmdLineArgumentInt(argc, (const char **)argv, "size"), 1);
    }
    MatrixMultiplyReal(N);
    return 0;
}
