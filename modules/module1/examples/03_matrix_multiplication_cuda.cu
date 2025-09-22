#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple matrix multiplication kernel (naive implementation)
__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication using shared memory (tiled)
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // Tile size (matches block size)
    const int TILE_SIZE = 16;
    
    // Shared memory for tiles of A and B
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && (t * TILE_SIZE + threadIdx.y) < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Matrix size (N x N)
    const int N = 512;  // Start with smaller size for demonstration
    const int size = N * N * sizeof(float);
    
    printf("Matrix Multiplication: %dx%d matrices\n", N, N);
    
    // Host matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);  // CPU reference
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // GPU computation using naive kernel
    printf("Running naive GPU kernel...\n");
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Time naive implementation
    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naiveTime;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // GPU computation using tiled kernel
    printf("Running tiled GPU kernel...\n");
    
    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float tiledTime;
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    
    // Copy tiled result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // CPU computation for verification (small subset)
    printf("Running CPU verification...\n");
    matrixMultiplyCPU(h_A, h_B, h_C_ref, min(N, 64));  // Only verify 64x64 subset for speed
    
    // Verify result (check small subset)
    bool correct = true;
    int checkSize = min(N, 64);
    for (int i = 0; i < checkSize; i++) {
        for (int j = 0; j < checkSize; j++) {
            if (fabs(h_C[i * N + j] - h_C_ref[i * checkSize + j]) > 1e-3) {
                correct = false;
                printf("Mismatch at (%d,%d): GPU=%.6f, CPU=%.6f\n", 
                       i, j, h_C[i * N + j], h_C_ref[i * checkSize + j]);
                break;
            }
        }
        if (!correct) break;
    }
    
    // Performance analysis
    printf("\n=== Performance Results ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Naive GPU time: %.3f ms\n", naiveTime);
    printf("Tiled GPU time: %.3f ms\n", tiledTime);
    printf("Tiled speedup: %.2fx\n", naiveTime / tiledTime);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / 1e9;  // 2N^3 operations
    printf("Naive performance: %.2f GFLOPS\n", gflops / (naiveTime / 1000.0));
    printf("Tiled performance: %.2f GFLOPS\n", gflops / (tiledTime / 1000.0));
    
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Memory bandwidth analysis
    double bytesTransferred = 3.0 * size;  // Read A, B and write C
    printf("Memory bandwidth (tiled): %.2f GB/s\n", 
           (bytesTransferred / (1024.0 * 1024.0 * 1024.0)) / (tiledTime / 1000.0));
    
    // Theoretical analysis
    printf("\n=== Analysis ===\n");
    printf("Total operations: %.2f billion\n", gflops);
    printf("Memory accesses (naive): %ld (inefficient)\n", (long)N * N * (2L * N + 1));
    printf("Memory accesses (tiled): %ld (optimized)\n", (long)3L * N * N);
    printf("Shared memory used: %d bytes per block\n", 2 * 16 * 16 * (int)sizeof(float));
    
    // Get device properties for context
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    // CUDA 13: use cudaDeviceGetAttribute for memory metrics
    int memClockKHz = 0;
    int busWidthBits = 0;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peakGBs = 2.0 * (memClockKHz / 1e6) * (busWidthBits / 8.0);
    printf("Peak memory bandwidth: %.1f GB/s\n", peakGBs);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    printf("\nMatrix multiplication completed successfully!\n");
    return 0;
}