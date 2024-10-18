#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define CHECK(call)                                                                  \
    {                                                                                \
        const cudaError_t cuda_ret = call;                                           \
        if (cuda_ret != cudaSuccess)                                                 \
        {                                                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
            printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
            exit(-1);                                                                \
        }                                                                            \
    }

double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec / 1.0e6);
}

// CPU matrix multiplication (basicSgemm_h)
void basicSgemm_h(float *A_h, float *B_h, float *C_h, unsigned int m, unsigned int k, unsigned int n)
{
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (unsigned int l = 0; l < k; l++)
            {
                sum += A_h[i * k + l] * B_h[l * n + j];
            }
            C_h[i * n + j] = sum;
        }
    }
}

// Kernel 1: 1 thread per element (matrixMulKernel_1thread1element)
__global__ void matrixMulKernel_1thread1element(float *A_d, float *B_d, float *C_d, unsigned int m, unsigned int k, unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < k; i++)
        {
            sum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = sum;
    }
}

// Kernel 2: 1 thread per row (matrixMulKernel_1thread1row)
__global__ void matrixMulKernel_1thread1row(float *A_d, float *B_d, float *C_d, unsigned int m, unsigned int k, unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (unsigned int l = 0; l < k; l++)
            {
                sum += A_d[row * k + l] * B_d[l * n + j];
            }
            C_d[row * n + j] = sum;
        }
    }
}

// Kernel 3: 1 thread per column (matrixMulKernel_1thread1column)
__global__ void matrixMulKernel_1thread1column(float *A_d, float *B_d, float *C_d, unsigned int m, unsigned int k, unsigned int n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n)
    {
        for (unsigned int i = 0; i < m; i++)
        {
            float sum = 0.0f;
            for (unsigned int l = 0; l < k; l++)
            {
                sum += A_d[i * k + l] * B_d[l * n + col];
            }
            C_d[i * n + col] = sum;
        }
    }
}

// Verification function
bool verify(float *CPU_Answer, float *GPU_Answer, unsigned int nRows, unsigned int nCols)
{
    for (unsigned int i = 0; i < nRows * nCols; i++)
    {
        if (fabs(CPU_Answer[i] - GPU_Answer[i]) > 1e-3)
        {
            return false;
        }
    }
    return true;
}

void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float *C_h)
{
    double startTime, endTime;

    float *A_d, *B_d, *C_d;

    startTime = myCPUTimer();
    CHECK(cudaMalloc((void **)&A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void **)&B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void **)&C_d, sizeof(float) * m * n));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMalloc:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    dim3 block(16, 16, 1); // 16x16 threads per block
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<grid, block>>>(A_d, B_d, C_d, m, k, n);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tMatrix multiplication<<<(%d, %d, %d), (%d, %d, %d)>>>:\t%f s\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float *C_h)
{
    double startTime, endTime;

    float *A_d, *B_d, *C_d;

    startTime = myCPUTimer();
    CHECK(cudaMalloc((void **)&A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void **)&B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void **)&C_d, sizeof(float) * m * n));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMalloc:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    // Define block and grid sizes: grid.y must cover rows
    dim3 block(1, 16);                         // 16 threads per block in the y direction
    dim3 grid(1, (m + block.y - 1) / block.y); // Grid spans rows

    startTime = myCPUTimer();
    matrixMulKernel_1thread1row<<<grid, block>>>(A_d, B_d, C_d, m, k, n);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tMatrix multiplication<<<(%d, %d, %d), (%d, %d, %d)>>>:\t%f s\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

void basicSgemm_d_1thread1column(int m, int k, int n, const float *A_h, const float *B_h, float *C_h)
{
    double startTime, endTime;

    float *A_d, *B_d, *C_d;

    startTime = myCPUTimer();
    CHECK(cudaMalloc((void **)&A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void **)&B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void **)&C_d, sizeof(float) * m * n));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMalloc:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    // Define block and grid sizes: grid.x must cover columns
    dim3 block(16, 1);                         // 16 threads per block in the x direction
    dim3 grid((n + block.y - 1) / block.y, 1); // Grid spans columns

    startTime = myCPUTimer();
    matrixMulKernel_1thread1column<<<grid, block>>>(A_d, B_d, C_d, m, k, n);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tMatrix multiplication<<<(%d, %d, %d), (%d, %d, %d)>>>:\t%f s\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, endTime - startTime);
    fflush(stdout);

    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();
    printf("\tcudaMemcpy:\t\t%f s\n", endTime - startTime);
    fflush(stdout);

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

int main(int argc, char **argv)
{
    CHECK(cudaDeviceSynchronize());
    double startTime, endTime;

    int m, k, n;
    if (argc < 4)
    {
        printf("Error. Usage: ./sgemm <m> <k> <n>\n");
        exit(1);
    }
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);

    // Allocate host memory for matrices A_h, B_h, and C_h
    float *A_h = (float *)malloc(m * k * sizeof(float));
    for (unsigned int i = 0; i < m * k; i++)
        A_h[i] = (float)(rand() % 100) / 100.0f;
    float *B_h = (float *)malloc(k * n * sizeof(float));
    for (unsigned int i = 0; i < k * n; i++)
        B_h[i] = (float)(rand() % 100) / 100.0f;
    float *C_h_cpu = (float *)calloc(m * n, sizeof(float));
    float *C_h_gpu = (float *)calloc(m * n, sizeof(float));

    // CPU Computation
    startTime = myCPUTimer();
    basicSgemm_h(A_h, B_h, C_h_cpu, m, k, n);
    endTime = myCPUTimer();
    printf("matrix multiplication on CPU:\t\t%f s\n\n", endTime - startTime);
    fflush(stdout);

    // 1thread1element
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h_gpu);
    endTime = myCPUTimer();

    printf("\tVerification:\t\t%s\n", verify(C_h_cpu, C_h_gpu, m, n) ? "Success" : "Failed");
    fflush(stdout);

    printf("matrix multiplication (1thread1element) on GPU:\t\t%f s\n\n", endTime - startTime);
    fflush(stdout);

    // 1thread1row
    startTime = myCPUTimer();
    basicSgemm_d_1thread1row(m, k, n, A_h, B_h, C_h_gpu);
    endTime = myCPUTimer();

    printf("\tVerification:\t\t%s\n", verify(C_h_cpu, C_h_gpu, m, n) ? "Success" : "Failed");
    fflush(stdout);

    printf("matrix multiplication (1thread1row) on GPU:\t\t%f s\n\n", endTime - startTime);
    fflush(stdout);

    // 1thread1column
    startTime = myCPUTimer();
    basicSgemm_d_1thread1column(m, k, n, A_h, B_h, C_h_gpu);
    endTime = myCPUTimer();

    printf("\tVerification:\t\t%s\n", verify(C_h_cpu, C_h_gpu, m, n) ? "Success" : "Failed");
    fflush(stdout);

    printf("matrix multiplication (1thread1column) on GPU:\t\t%f s\n\n", endTime - startTime);
    fflush(stdout);

    // Free host memory of arrays A_h, B_h, and C_h
    free(A_h);
    free(B_h);
    free(C_h_cpu);
    free(C_h_gpu);

    return 0;
}
