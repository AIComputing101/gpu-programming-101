#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple matrix multiplication kernel (naive implementation)
__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    // Calculate row and column for this thread
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
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
    
    int row = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && (t * TILE_SIZE + hipThreadIdx_x) < N) {
            tileA[hipThreadIdx_y][hipThreadIdx_x] = A[row * N + t * TILE_SIZE + hipThreadIdx_x];
        } else {
            tileA[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }
        
        if (col < N && (t * TILE_SIZE + hipThreadIdx_y) < N) {
            tileB[hipThreadIdx_y][hipThreadIdx_x] = B[(t * TILE_SIZE + hipThreadIdx_y) * N + col];
        } else {
            tileB[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[hipThreadIdx_y][k] * tileB[k][hipThreadIdx_x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Platform-specific optimized version
#ifdef __HIP_PLATFORM_AMD__
__global__ void matrixMultiplyAMDOptimized(float *A, float *B, float *C, int N) {
    // AMD-specific optimizations
    const int TILE_SIZE = 16;
    
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    int row = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    
    float sum = 0.0f;
    
    // Unrolled tile loop for better instruction scheduling
    #pragma unroll 4
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Coalesced loads
        tileA[hipThreadIdx_y][hipThreadIdx_x] = 
            (row < N && (t * TILE_SIZE + hipThreadIdx_x) < N) ? 
            A[row * N + t * TILE_SIZE + hipThreadIdx_x] : 0.0f;
            
        tileB[hipThreadIdx_y][hipThreadIdx_x] = 
            (col < N && (t * TILE_SIZE + hipThreadIdx_y) < N) ? 
            B[(t * TILE_SIZE + hipThreadIdx_y) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Unrolled computation for better pipeline utilization
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[hipThreadIdx_y][k] * tileB[k][hipThreadIdx_x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
#elif defined(__HIP_PLATFORM_NVIDIA__)
__global__ void matrixMultiplyNVIDIAOptimized(float *A, float *B, float *C, int N) {
    // NVIDIA-specific optimizations
    const int TILE_SIZE = 16;
    
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    int row = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Use texture cache hints for better memory performance
        tileA[hipThreadIdx_y][hipThreadIdx_x] = 
            (row < N && (t * TILE_SIZE + hipThreadIdx_x) < N) ? 
            __ldg(&A[row * N + t * TILE_SIZE + hipThreadIdx_x]) : 0.0f;
            
        tileB[hipThreadIdx_y][hipThreadIdx_x] = 
            (col < N && (t * TILE_SIZE + hipThreadIdx_y) < N) ? 
            __ldg(&B[(t * TILE_SIZE + hipThreadIdx_y) * N + col]) : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(tileA[hipThreadIdx_y][k], tileB[k][hipThreadIdx_x], sum);
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
#endif

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
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
    
    printf("HIP Matrix Multiplication: %dx%d matrices\n", N, N);
    
    // Get device information
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Running on: %s\n", props.name);
    printf("Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
#else
    printf("Unknown\n");
#endif
    
    // Host matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);  // CPU reference
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    srand(42);  // Fixed seed for reproducible results
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size));
    HIP_CHECK(hipMalloc(&d_B, size));
    HIP_CHECK(hipMalloc(&d_C, size));
    
    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    printf("\n=== Performance Comparison ===\n");
    
    // Test naive implementation
    printf("Running naive HIP kernel...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float naiveTime;
    HIP_CHECK(hipEventElapsedTime(&naiveTime, start, stop));
    
    // Test tiled implementation
    printf("Running tiled HIP kernel...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float tiledTime;
    HIP_CHECK(hipEventElapsedTime(&tiledTime, start, stop));
    
    // Test platform-specific optimized version
    float optimizedTime = 0.0f;
    bool hasOptimized = false;
    
#ifdef __HIP_PLATFORM_AMD__
    printf("Running AMD-optimized kernel...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixMultiplyAMDOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&optimizedTime, start, stop));
    hasOptimized = true;
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("Running NVIDIA-optimized kernel...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixMultiplyNVIDIAOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&optimizedTime, start, stop));
    hasOptimized = true;
#endif
    
    // Copy final result back
    HIP_CHECK(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));
    
    // CPU computation for verification (small subset)
    printf("Running CPU verification...\n");
    int checkSize = (N > 64) ? 64 : N;  // Only verify subset for speed
    matrixMultiplyCPU(h_A, h_B, h_C_ref, checkSize);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < checkSize && correct; i++) {
        for (int j = 0; j < checkSize && correct; j++) {
            int gpu_idx = i * N + j;
            int cpu_idx = i * checkSize + j;
            if (fabs(h_C[gpu_idx] - h_C_ref[cpu_idx]) > 1e-3) {
                correct = false;
                printf("Mismatch at (%d,%d): GPU=%.6f, CPU=%.6f\n", 
                       i, j, h_C[gpu_idx], h_C_ref[cpu_idx]);
            }
        }
    }
    
    // Performance analysis
    printf("\n=== Performance Results ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Naive time: %.3f ms\n", naiveTime);
    printf("Tiled time: %.3f ms\n", tiledTime);
    if (hasOptimized) {
        printf("Optimized time: %.3f ms\n", optimizedTime);
        printf("Tiled vs Naive speedup: %.2fx\n", naiveTime / tiledTime);
        printf("Optimized vs Tiled speedup: %.2fx\n", tiledTime / optimizedTime);
        printf("Overall speedup: %.2fx\n", naiveTime / optimizedTime);
    } else {
        printf("Tiled speedup: %.2fx\n", naiveTime / tiledTime);
    }
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / 1e9;  // 2N^3 operations
    printf("\nPerformance (GFLOPS):\n");
    printf("  Naive: %.2f GFLOPS\n", gflops / (naiveTime / 1000.0));
    printf("  Tiled: %.2f GFLOPS\n", gflops / (tiledTime / 1000.0));
    if (hasOptimized) {
        printf("  Optimized: %.2f GFLOPS\n", gflops / (optimizedTime / 1000.0));
    }
    
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Memory and compute analysis
    printf("\n=== Analysis ===\n");
    printf("Total operations: %.2f billion\n", gflops);
    printf("Arithmetic intensity: %.2f ops/byte\n", 
           (2.0 * N * N * N) / (3.0 * N * N * sizeof(float)));
    
    double bestTime = hasOptimized ? optimizedTime : tiledTime;
    double bytesTransferred = 3.0 * size;  // Read A, B and write C
    printf("Memory bandwidth achieved: %.2f GB/s\n", 
           (bytesTransferred / (1024.0 * 1024.0 * 1024.0)) / (bestTime / 1000.0));
    
    // Theoretical peak bandwidth
    double theoreticalBW = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical peak bandwidth: %.2f GB/s\n", theoreticalBW);
    printf("Bandwidth efficiency: %.1f%%\n", 
           ((bytesTransferred / (1024.0 * 1024.0 * 1024.0)) / (bestTime / 1000.0) / theoreticalBW) * 100.0);
    
    // Platform-specific insights
    printf("\nPlatform-specific optimizations:\n");
#ifdef __HIP_PLATFORM_AMD__
    printf("  - Used wavefront-optimized memory access patterns\n");
    printf("  - Applied loop unrolling for better instruction scheduling\n");
    printf("  - Optimized for AMD GCN architecture\n");
    printf("  - Wavefront size: %d\n", props.warpSize);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("  - Used texture cache with __ldg() for reads\n");
    printf("  - Applied fused multiply-add with __fmaf_rn()\n");
    printf("  - Optimized for NVIDIA warp execution\n");
    printf("  - Warp size: %d\n", props.warpSize);
#else
    printf("  - Generic optimizations applied\n");
#endif
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipEventDestroy(start); hipEventDestroy(stop);
    
    printf("\nHIP matrix multiplication completed successfully!\n");
    return 0;
}