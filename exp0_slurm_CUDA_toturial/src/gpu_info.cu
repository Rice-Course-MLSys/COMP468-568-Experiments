#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Display detailed GPU information
 */
void printGPUInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    printf("========================================\n");
    printf("CUDA GPU Information\n");
    printf("========================================\n");
    printf("Number of GPUs: %d\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Total Constant Memory: %zu KB\n", prop.totalConstMem / 1024);
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Size: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Can Map Host Memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("\n");
    }
    printf("========================================\n");
}

int main() {
    printGPUInfo();
    return 0;
}
