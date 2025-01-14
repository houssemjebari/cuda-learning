#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_runtime.h>


#define M 256 // Number of Lines in the Matrix
#define N 512 // Number of columns in the matrix
#define K 256
#define BLOCK_SIZE 32 // Number of threads within each block


void matmul_cpu(float *a, float *b, float* c, int m, int k, int n) {
    for (int i=0; i< m; i++) {
        for (int j=0; j<n; j++) {
            for (int l=0; l<k; l++){
                c[i * n + j] += a[i * k + l] * b[l * n + j] ;
            }
        }
    }
}

__global__ void matmul_gpu(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.f;
        for (int l=0; l<k; l++) {
            sum+= a[row * k + l] * b[col + l * k];
        }
        c[row * n + col] = sum;
    }   
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}



int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;

    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate Host Memory
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C_cpu = (float *)malloc(size_C);
    h_C_gpu = (float *)malloc(size_C);

    // Initiate the matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B,K, N);

    // Allocate Device Memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1)/ BLOCK_SIZE, 
                 (M + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    
    // Benchmarking the CPU Implementation 
    printf("Benchmarking the CPU Implementation...\n");
    double cpu_total_time = 0.;
    for (int i=0;i<20;i++) {
        double start_time = get_time();
        matmul_cpu(h_A,h_B,h_C_cpu,M,K,N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmarking GPU Implementation 
    printf("Benchmarking the GPU Implementation...\n");
    double gpu_total_time = 0.;
    for (int i=0;i<20;i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C,M,K,N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}