#include <cuda_runtime.h>
#include <stdio.h>

// Our mutex structure
struct Mutex {
    int * lock;
};

// Initialize the mutex 
__host__ void initMutex(Mutex * m) {
    cudaMalloc((void **) &m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// Acquire the mutex 
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0 , 1) != 0) {
    }
}

// Release the mutex 
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// Kernel function to demonstrate the mutex usage
__global__ void mutexKernel(int* counter, Mutex *m) {
    lock(m);
    // Critical section
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);

    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel with multiple threads 
    mutexKernel<<<1, 1000>>>(d_counter, &m);

    // Copy the results back from the GPU
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Counter value: %d\n", result);

    // Free-up memory
    cudaFree(m.lock);
    cudaFree(d_counter);
}