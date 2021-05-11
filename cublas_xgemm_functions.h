// One dim block size
#define BSIZE 512 
// Number of same multiplication iteration: result is mean 
#define NITER 300 
// TODO: test some matrix value, compared to CPU computing in double
#define RANDCORRECTION 10

void MatrixMultiplyComplex(int N, const int block_size);
void MatrixMultiplyReal(const int N, const int block_size);
