#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1000000  // Size of vectors
#define BLOCK_SIZE 256

/**
 * CUDA kernel for vector addition
 * Performs element-wise addition: C[i] = A[i] + B[i]
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * CPU function to verify CUDA results
 */
void verifyCUDAResults(const float *A, const float *B, const float *C, int n) {
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        float actual = C[i];
        if (fabs(expected - actual) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n", i, expected, actual);
            return;
        }
    }
    printf("✓ Verification passed! All values match.\n");
}

int main(int argc, char *argv[]) {
    int size = N;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    
    size_t bytes = size * sizeof(float);
    
    printf("========================================\n");
    printf("CUDA Vector Addition Experiment\n");
    printf("========================================\n");
    printf("Vector size: %d elements\n", size);
    printf("Memory per vector: %.2f MB\n", bytes / (1024.0 * 1024.0));
    printf("Block size: %d threads\n", BLOCK_SIZE);
    printf("Grid size: %d blocks\n", (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("========================================\n\n");
    
    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }
    
    // Initialize host arrays
    printf("Initializing vectors...\n");
    for (int i = 0; i < size; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copy data to device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Execute kernel
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching kernel...\n");
    cudaEventRecord(start);
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back to host
    printf("Copying results back to host...\n");
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("Verifying results...\n");
    verifyCUDAResults(h_A, h_B, h_C, size);
    
    // Print performance metrics
    printf("\n========================================\n");
    printf("Performance Metrics:\n");
    printf("========================================\n");
    printf("Kernel execution time: %.4f ms\n", milliseconds);
    printf("Throughput: %.2f GB/s\n", (3.0 * bytes / (1e6 * milliseconds)));
    printf("========================================\n");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n✓ Program completed successfully!\n");
    return 0;
}
