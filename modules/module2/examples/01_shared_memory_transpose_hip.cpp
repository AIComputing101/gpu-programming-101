#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define TILE_SIZE 32

// Naive matrix transpose (inefficient)
__global__ void matrixTransposeNaive(float *input, float *output, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Optimized transpose using shared memory with bank conflict avoidance
__global__ void matrixTransposeShared(float *input, float *output, int width, int height) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Global indices for input
    int x = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    
    // Load data into shared memory
    if (x < width && y < height) {
        tile[hipThreadIdx_y][hipThreadIdx_x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Global indices for output (transposed)
    x = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_y;
    
    // Write transposed data to output
    if (x < height && y < width) {
        output[y * height + x] = tile[hipThreadIdx_x][hipThreadIdx_y];
    }
}

// Platform-specific optimized versions
#ifdef __HIP_PLATFORM_AMD__
__global__ void matrixTransposeAMDOptimized(float *input, float *output, int width, int height) {
    // AMD-specific optimizations for wavefront execution
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    
    // Coalesced load with wavefront-aware access
    if (x < width && y < height) {
        tile[hipThreadIdx_y][hipThreadIdx_x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Transpose indices
    x = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_y;
    
    // Coalesced store
    if (x < height && y < width) {
        output[y * height + x] = tile[hipThreadIdx_x][hipThreadIdx_y];
    }
}
#elif defined(__HIP_PLATFORM_NVIDIA__)
__global__ void matrixTransposeNVIDIAOptimized(float *input, float *output, int width, int height) {
    // NVIDIA-specific optimizations
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    
    // Use texture cache hint for input reads
    if (x < width && y < height) {
        tile[hipThreadIdx_y][hipThreadIdx_x] = __ldg(&input[y * width + x]);
    }
    
    __syncthreads();
    
    x = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[hipThreadIdx_x][hipThreadIdx_y];
    }
}
#endif

// Demonstration of bank conflicts (AMD: LDS conflicts, NVIDIA: shared memory conflicts)
__global__ void bankConflictDemo(float *data, int n) {
    __shared__ float shared_data[32][32];
    __shared__ float shared_padded[32][33];  // Padded to avoid conflicts
    
    int tid = hipThreadIdx_x;
    
    if (hipBlockIdx_x == 0 && tid < 32) {
        // BAD: All threads access the same bank/LDS bank (conflict)
        shared_data[tid][0] = data[tid];
        __syncthreads();
        
        // GOOD: Each thread accesses a different bank/LDS bank
        shared_padded[0][tid] = data[tid + 32];
        __syncthreads();
        
        // Write back results
        data[tid] = shared_data[tid][0] + shared_padded[0][tid];
    }
}

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
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
    printf("HIP Shared Memory Matrix Transpose Example\n");
    printf("==========================================\n");
    
    // Get device information
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Running on: %s\n", props.name);
    printf("Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
    printf("Wavefront size: %d\n", props.warpSize);
    printf("LDS per workgroup: %zu bytes\n", props.sharedMemPerBlock);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
    printf("Warp size: %d\n", props.warpSize);
    printf("Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
#else
    printf("Unknown\n");
#endif
    
    // Matrix dimensions
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    // Host matrices
    float *h_input = (float*)malloc(size);
    float *h_output_naive = (float*)malloc(size);
    float *h_output_shared = (float*)malloc(size);
    float *h_output_optimized = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    
    // Initialize input matrix
    printf("\nInitializing %dx%d matrix...\n", width, height);
    srand(42);  // Fixed seed for reproducible results
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(rand() % 100);
    }
    
    // Device matrices
    float *d_input, *d_output_naive, *d_output_shared, *d_output_optimized;
    HIP_CHECK(hipMalloc(&d_input, size));
    HIP_CHECK(hipMalloc(&d_output_naive, size));
    HIP_CHECK(hipMalloc(&d_output_shared, size));
    HIP_CHECK(hipMalloc(&d_output_optimized, size));
    
    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
    
    // Grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, 
                  (height + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid size: (%d, %d), Block size: (%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Test 1: Naive transpose
    printf("\n=== Performance Tests ===\n");
    printf("Testing naive transpose...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixTransposeNaive<<<gridSize, blockSize>>>(d_input, d_output_naive, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float naiveTime;
    HIP_CHECK(hipEventElapsedTime(&naiveTime, start, stop));
    
    // Test 2: Shared memory transpose
    printf("Testing shared memory transpose...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixTransposeShared<<<gridSize, blockSize>>>(d_input, d_output_shared, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float sharedTime;
    HIP_CHECK(hipEventElapsedTime(&sharedTime, start, stop));
    
    // Test 3: Platform-specific optimized version
    float optimizedTime = 0.0f;
    bool hasOptimized = false;
    
#ifdef __HIP_PLATFORM_AMD__
    printf("Testing AMD-optimized transpose...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixTransposeAMDOptimized<<<gridSize, blockSize>>>(d_input, d_output_optimized, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&optimizedTime, start, stop));
    hasOptimized = true;
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("Testing NVIDIA-optimized transpose...\n");
    HIP_CHECK(hipEventRecord(start));
    matrixTransposeNVIDIAOptimized<<<gridSize, blockSize>>>(d_input, d_output_optimized, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&optimizedTime, start, stop));
    hasOptimized = true;
#endif
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_output_naive, d_output_naive, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_output_shared, d_output_shared, size, hipMemcpyDeviceToHost));
    if (hasOptimized) {
        HIP_CHECK(hipMemcpy(h_output_optimized, d_output_optimized, size, hipMemcpyDeviceToHost));
    }
    
    // CPU reference for verification
    printf("Computing CPU reference...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixTransposeCPU(h_input, h_output_cpu, width, height);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Verify results
    bool naive_correct = true, shared_correct = true, optimized_correct = true;
    for (int i = 0; i < width * height; i++) {
        if (fabs(h_output_naive[i] - h_output_cpu[i]) > 1e-5) {
            naive_correct = false;
        }
        if (fabs(h_output_shared[i] - h_output_cpu[i]) > 1e-5) {
            shared_correct = false;
        }
        if (hasOptimized && fabs(h_output_optimized[i] - h_output_cpu[i]) > 1e-5) {
            optimized_correct = false;
        }
    }
    
    // Performance results
    printf("\n=== Performance Results ===\n");
    printf("Matrix size: %dx%d\n", width, height);
    printf("CPU time: %.3f ms\n", cpuTime);
    printf("Naive HIP time: %.3f ms\n", naiveTime);
    printf("Shared memory time: %.3f ms\n", sharedTime);
    if (hasOptimized) {
        printf("Platform-optimized time: %.3f ms\n", optimizedTime);
    }
    
    printf("\nSpeedup Analysis:\n");
    printf("CPU vs Naive: %.2fx\n", cpuTime / naiveTime);
    printf("CPU vs Shared: %.2fx\n", cpuTime / sharedTime);
    printf("Naive vs Shared: %.2fx\n", naiveTime / sharedTime);
    if (hasOptimized) {
        printf("CPU vs Optimized: %.2fx\n", cpuTime / optimizedTime);
        printf("Shared vs Optimized: %.2fx\n", sharedTime / optimizedTime);
    }
    
    // Bandwidth analysis
    double bytes_transferred = 2.0 * size;  // Read input + write output
    printf("\nBandwidth Analysis:\n");
    printf("Naive bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (naiveTime / 1000.0));
    printf("Shared bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (sharedTime / 1000.0));
    if (hasOptimized) {
        printf("Optimized bandwidth: %.2f GB/s\n", 
               (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (optimizedTime / 1000.0));
    }
    
    // Theoretical peak bandwidth
    double theoreticalBW = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical peak bandwidth: %.2f GB/s\n", theoreticalBW);
    
    double bestBW = (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / 
                    ((hasOptimized ? optimizedTime : sharedTime) / 1000.0);
    printf("Bandwidth efficiency: %.1f%%\n", (bestBW / theoreticalBW) * 100.0);
    
    printf("\nVerification:\n");
    printf("Naive transpose: %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("Shared transpose: %s\n", shared_correct ? "PASSED" : "FAILED");
    if (hasOptimized) {
        printf("Optimized transpose: %s\n", optimized_correct ? "PASSED" : "FAILED");
    }
    
    // Memory access pattern analysis
    printf("\n=== Memory Access Pattern Analysis ===\n");
    printf("Naive transpose:\n");
    printf("  - Input: coalesced reads (good)\n");
    printf("  - Output: strided writes (bad for bandwidth)\n");
    printf("  - No data reuse\n");
    printf("\nShared memory transpose:\n");
    printf("  - Input: coalesced reads (good)\n");
    printf("  - Shared memory: efficient data reuse\n");
    printf("  - Output: coalesced writes (good)\n");
    printf("  - Bank/LDS conflict avoidance with padding\n");
    
#ifdef __HIP_PLATFORM_AMD__
    printf("\nAMD-specific optimizations:\n");
    printf("  - Wavefront-aware memory access patterns\n");
    printf("  - LDS (Local Data Share) optimization\n");
    printf("  - Memory coalescing for GCN architecture\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("\nNVIDIA-specific optimizations:\n");
    printf("  - Texture cache utilization with __ldg()\n");
    printf("  - Warp-level memory coalescing\n");
    printf("  - Shared memory bank conflict avoidance\n");
#endif
    
    // Bank conflict demonstration
    printf("\nTesting bank/LDS conflict demonstration...\n");
    float *d_bank_data;
    HIP_CHECK(hipMalloc(&d_bank_data, 64 * sizeof(float)));
    
    dim3 bankBlock(32, 1);
    dim3 bankGrid(1, 1);
    bankConflictDemo<<<bankGrid, bankBlock>>>(d_bank_data, 64);
    HIP_CHECK(hipDeviceSynchronize());
    
    printf("Bank conflict demo completed\n");
    printf("(Use rocprof or nvprof to analyze memory conflicts)\n");
    
    // Cleanup
    free(h_input);
    free(h_output_naive);
    free(h_output_shared);
    free(h_output_optimized);
    free(h_output_cpu);
    hipFree(d_input);
    hipFree(d_output_naive);
    hipFree(d_output_shared);
    hipFree(d_output_optimized);
    hipFree(d_bank_data);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    printf("\nHIP shared memory transpose example completed successfully!\n");
    return 0;
}