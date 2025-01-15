#include <cuda_runtime.h> 
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMultiplyOptimized(float *A, float *B, float *C, int M, int N, int K) {
    
    // Declare Shared Arrays 
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // Compute Indicies 
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    // Loop over all the tiles 
    float sum = 0.f;
    for (int tile=0; tile < (K + TILE_SIZE -1) / TILE_SIZE; ++tile) {
       
        // Fill the shared arrays 
        if (row < M && (tx + tile * TILE_SIZE) < K)
            sharedA[ty][tx] = A[row * M + tx + tile*TILE_SIZE];
        else 
            sharedA[ty][tx] = 0.f;
        if (col < N && (ty + tile*TILE_SIZE) <  K)
            sharedB[ty][tx] = B[col + N*(ty + tile*TILE_SIZE)];
        else 
            sharedB[ty][tx] = 0.f;
        __syncthreads();
        
        // Compute the sums
        for (int k=0; k<TILE_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }
    // Update the output matrix
    if (row < M && col < N)
        C[row * N + col] = sum;    
}

int main() {

    // Define Matrix Dimensions
    const int M = 1024; // Number of rows in A and C
    const int N = 1024; // Number of columns in B and C
    const int K = 1024; // Number of columns in A and rows in B

    // Calculate the matrix sizes in bytes 
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate Device Memory 
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Kernel Launch 
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1)/ TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Synchronize Device
    cudaDeviceSynchronize();

    // Free device memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any CUDA errors 
    cudaError_t error= cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    return 0;


}