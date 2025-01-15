#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        printf("Device %d: \"%s\"\n", device, prop.name);
        printf("  CUDA Capability Major/Minor version number: %d.%d\n", 
               prop.major, prop.minor);
        printf("  Total Global Memory: %zu bytes (%.2f GB)\n",
               prop.totalGlobalMem, (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max dimension size of block: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimension size of grid: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock rate: %d kHz\n", prop.clockRate);
        printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("  Multi-processor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per multi-processor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Device overlap (can overlap kernel & memcopies): %s\n", 
               (prop.deviceOverlap ? "Yes" : "No"));
        printf("  Concurrent kernels: %s\n\n", 
               (prop.concurrentKernels ? "Yes" : "No"));
    }

    return 0;
}
