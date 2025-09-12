#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32

// Naive matrix transpose (inefficient)
__global__ void matrixTransposeNaive(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Optimized transpose using shared memory with bank conflict avoidance
__global__ void matrixTransposeShared(float *input, float *output, int width, int height) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Global indices for input
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Global indices for output (transposed)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write transposed data to output
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Demonstration of bank conflicts
__global__ void bankConflictDemo(float *data, int n) {
    __shared__ float shared_data[32][32];
    __shared__ float shared_padded[32][33];  // Padded to avoid conflicts
    
    int tid = threadIdx.x;
    
    if (blockIdx.x == 0 && tid < 32) {
        // BAD: All threads access the same bank (bank conflict)
        shared_data[tid][0] = data[tid];
        __syncthreads();
        
        // GOOD: Each thread accesses a different bank
        shared_padded[0][tid] = data[tid + 32];
        __syncthreads();
        
        // Write back results
        data[tid] = shared_data[tid][0] + shared_padded[0][tid];
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

// CPU matrix transpose for verification
void matrixTransposeCPU(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

int main() {
    printf("CUDA Shared Memory Matrix Transpose Example\n");
    printf("==========================================\n");
    
    // Matrix dimensions
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    // Host matrices
    float *h_input = (float*)malloc(size);
    float *h_output_naive = (float*)malloc(size);
    float *h_output_shared = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    
    // Initialize input matrix
    printf("Initializing %dx%d matrix...\n", width, height);
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(i % 100);
    }
    
    // Device matrices
    float *d_input, *d_output_naive, *d_output_shared;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output_naive, size));
    CUDA_CHECK(cudaMalloc(&d_output_shared, size));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, 
                  (height + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid size: (%d, %d), Block size: (%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test 1: Naive transpose
    printf("\nTesting naive transpose...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixTransposeNaive<<<gridSize, blockSize>>>(d_input, d_output_naive, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naiveTime;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    
    // Test 2: Shared memory transpose
    printf("Testing shared memory transpose...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixTransposeShared<<<gridSize, blockSize>>>(d_input, d_output_shared, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float sharedTime;
    CUDA_CHECK(cudaEventElapsedTime(&sharedTime, start, stop));
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output_shared, size, cudaMemcpyDeviceToHost));
    
    // CPU reference for verification
    printf("Computing CPU reference...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixTransposeCPU(h_input, h_output_cpu, width, height);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Verify results
    bool naive_correct = true, shared_correct = true;
    for (int i = 0; i < width * height; i++) {
        if (fabs(h_output_naive[i] - h_output_cpu[i]) > 1e-5) {
            naive_correct = false;
        }
        if (fabs(h_output_shared[i] - h_output_cpu[i]) > 1e-5) {
            shared_correct = false;
        }
    }
    
    // Performance results
    printf("\n=== Performance Results ===\n");
    printf("Matrix size: %dx%d\n", width, height);
    printf("CPU time: %.3f ms\n", cpuTime);
    printf("Naive GPU time: %.3f ms\n", naiveTime);
    printf("Shared memory time: %.3f ms\n", sharedTime);
    printf("CPU vs Naive speedup: %.2fx\n", cpuTime / naiveTime);
    printf("CPU vs Shared speedup: %.2fx\n", cpuTime / sharedTime);
    printf("Naive vs Shared speedup: %.2fx\n", naiveTime / sharedTime);
    
    // Bandwidth analysis
    double bytes_transferred = 2.0 * size;  // Read input + write output
    printf("\nBandwidth Analysis:\n");
    printf("Naive bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (naiveTime / 1000.0));
    printf("Shared bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (sharedTime / 1000.0));
    
    printf("\nVerification:\n");
    printf("Naive transpose: %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("Shared transpose: %s\n", shared_correct ? "PASSED" : "FAILED");
    
    // Memory access pattern analysis
    printf("\n=== Memory Access Pattern Analysis ===\n");
    printf("Naive transpose:\n");
    printf("  - Input: coalesced reads (good)\n");
    printf("  - Output: strided writes (bad)\n");
    printf("  - No reuse of loaded data\n");
    printf("\nShared memory transpose:\n");
    printf("  - Input: coalesced reads (good)\n");
    printf("  - Shared memory: efficient reuse\n");
    printf("  - Output: coalesced writes (good)\n");
    printf("  - Bank conflict avoidance with padding\n");
    
    // Bank conflict demonstration
    printf("\nTesting bank conflict demonstration...\n");
    float *d_bank_data;
    CUDA_CHECK(cudaMalloc(&d_bank_data, 64 * sizeof(float)));
    
    dim3 bankBlock(32, 1);
    dim3 bankGrid(1, 1);
    bankConflictDemo<<<bankGrid, bankBlock>>>(d_bank_data, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Bank conflict demo completed (check with profiler)\n");
    
    // Cleanup
    free(h_input);
    free(h_output_naive);
    free(h_output_shared);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_shared);
    cudaFree(d_bank_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nShared memory transpose example completed successfully!\n");
    return 0;
}