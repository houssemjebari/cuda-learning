#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000
#define BLOCK_SIZE_1D 1024 
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8


void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


__global__ void vector_add_gpu_3d(float* a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    int j = blockIdx.y * blockDim.y  + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float) rand() / RAND_MAX;
    }
}

double get_time() { 
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*) malloc(size);
    h_b = (float*) malloc(size);
    h_c_cpu = (float*) malloc(size);
    h_c_gpu = (float*) malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);


    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Benchmark CPU Implementation 
    printf("Benchmarking CPU Implementation ...\n");
    double cpu_total_time = 0.;
    for (int i=0; i<20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        cpu_total_time += get_time() - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.;

    // Benchmark GPU Implementation 1D
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    printf("Benchmarking GPU Implementation 1D...\n");
    double gpu_1d_total_time = 0.;
    for (int i=0; i<20; i++) {
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        gpu_1d_total_time += get_time() - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 20.;


    // Benchmark GPU Implementation 3D
    int nx = 100, ny = 100, nz = 1000;
    dim3 block_dims(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks((nx + BLOCK_SIZE_3D_X - 1) / BLOCK_SIZE_3D_X,
                    (ny + BLOCK_SIZE_3D_Y - 1) / BLOCK_SIZE_3D_Y,
                    (nz + BLOCK_SIZE_3D_Z - 1) / BLOCK_SIZE_3D_Z);

    printf("Benchmarking GPU Implementation 3D...\n");
    double gpu_total_time = 0.;
    for (int i=0; i<20; i++) {
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks, block_dims>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
        gpu_total_time += get_time() - start_time;
    }
    double gpu_3d_avg_time = gpu_total_time / 20.;    


    // Print Results 
    printf("CPU Average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D Average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D Average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("speedup 1D: %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("speedup 3D: %fx\n", cpu_avg_time / gpu_3d_avg_time);


    // verify results 
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i=0; i<N; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]){
            correct = false;
            break;
        }
    }
    printf("results are %s\n", correct? "correct" : "incorrect");

    // Free the memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}