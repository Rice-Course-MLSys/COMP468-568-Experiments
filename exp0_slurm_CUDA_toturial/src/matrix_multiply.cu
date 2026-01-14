#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_SIZE 1024
#define TILE_SIZE 32

/**
 * CUDA kernel for matrix multiplication with tiling optimization
 * Computes C = A * B where A, B, C are MATRIX_SIZE x MATRIX_SIZE matrices
 */
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, 
                                     int size) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over all tiles in the K dimension
    for (int tile = 0; tile < (size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        int tile_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < size && tile_col < size) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * size + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tile_row < size && col < size) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_row * size + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}

/**
 * Initialize matrix with random values
 */
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

/**
 * Verify matrix multiplication result (sample-based for large matrices)
 */
void verifyMatrixMultiply(const float *A, const float *B, const float *C, 
                          int size, int samples) {
    for (int s = 0; s < samples; s++) {
        int i = rand() % size;
        int j = rand() % size;
        
        float expected = 0.0f;
        for (int k = 0; k < size; k++) {
            expected += A[i * size + k] * B[k * size + j];
        }
        
        float actual = C[i * size + j];
        if (fabs(expected - actual) > 0.1f) {
            printf("Verification failed at [%d][%d]: expected %f, got %f\n", 
                   i, j, expected, actual);
            return;
        }
    }
    printf("✓ Verification passed! (sampled %d elements)\n", samples);
}

int main(int argc, char *argv[]) {
    int size = MATRIX_SIZE;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    
    // Ensure size is multiple of TILE_SIZE
    if (size % TILE_SIZE != 0) {
        size = ((size + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        printf("Adjusted matrix size to %d to be multiple of TILE_SIZE\n", size);
    }
    
    size_t matrix_bytes = size * size * sizeof(float);
    
    printf("========================================\n");
    printf("CUDA Matrix Multiplication Experiment\n");
    printf("========================================\n");
    printf("Matrix size: %d x %d\n", size, size);
    printf("Memory per matrix: %.2f MB\n", matrix_bytes / (1024.0 * 1024.0));
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("========================================\n\n");
    
    // Allocate host memory
    float *h_A = (float *)malloc(matrix_bytes);
    float *h_B = (float *)malloc(matrix_bytes);
    float *h_C = (float *)malloc(matrix_bytes);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrix_bytes);
    cudaMalloc(&d_B, matrix_bytes);
    cudaMalloc(&d_C, matrix_bytes);
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copy data to device
    printf("Copying matrices to GPU...\n");
    cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + TILE_SIZE - 1) / TILE_SIZE, 
                 (size + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Launching kernel with grid (%d, %d) and block (%d, %d)...\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    cudaEventRecord(start);
    matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back
    printf("Copying result back to host...\n");
    cudaMemcpy(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("Verifying results (sampled verification)...\n");
    verifyMatrixMultiply(h_A, h_B, h_C, size, 100);
    
    // Print performance metrics
    double operations = 2.0 * size * size * size;  // FLOPs
    double gflops = (operations / (milliseconds * 1e6));
    
    printf("\n========================================\n");
    printf("Performance Metrics:\n");
    printf("========================================\n");
    printf("Kernel execution time: %.4f ms\n", milliseconds);
    printf("Total operations: %.2e FLOP\n", operations);
    printf("Performance: %.2f GFLOP/s\n", gflops);
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
