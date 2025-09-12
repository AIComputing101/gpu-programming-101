#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Example kernel with potential issues for debugging practice
__global__ void debugKernel(float *data, int n) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    // Debug print (only from first few threads to avoid spam)
    if (tid < 5) {
        printf("HIP Thread %d: processing element %d\n", tid, tid);
    }
    
    // Boundary check - this is important!
    if (tid < n) {
        // Demonstrate thread divergence
        if (tid % 2 == 0) {
            data[tid] = tid * 2.0f;
        } else {
            data[tid] = tid * 3.0f;
        }
    }
    
    // Example of potential out-of-bounds access (commented out for safety)
    // data[tid] = tid;  // This would cause issues if tid >= n
}

// Kernel to demonstrate shared memory usage
__global__ void sharedMemoryExample(float *input, float *output, int n) {
    // Allocate shared memory
    __shared__ float shared_data[256];
    
    int tid = hipThreadIdx_x;
    int gid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    // Load data to shared memory
    if (gid < n) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;  // Handle out-of-bounds
    }
    
    // Synchronize all threads in the block
    __syncthreads();
    
    // Simple processing in shared memory (multiply by 2)
    shared_data[tid] *= 2.0f;
    
    // Synchronize again before writing back
    __syncthreads();
    
    // Write back to global memory
    if (gid < n) {
        output[gid] = shared_data[tid];
    }
}

// HIP-specific warp-level operations example
__global__ void warpLevelExample(float *input, float *output, int n) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    if (tid < n) {
        float value = input[tid];
        
        // Warp-level primitive operations (HIP/CUDA compatible)
        // Note: These require newer GPU architectures
        #if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
        // Simple warp reduction example
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down(value, offset);
        }
        #endif
        
        // Only the first thread in each warp writes the result
        if ((hipThreadIdx_x % 32) == 0) {
            output[tid / 32] = value;
        }
    }
}

// Comprehensive error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to check device properties and limits
void checkDeviceLimits() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
    printf("Device Information:\n");
    printf("  Name: %s\n", props.name);
    printf("  Architecture: %s\n", props.gcnArchName);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Max block dimensions: (%d, %d, %d)\n", 
           props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("  Max grid dimensions: (%d, %d, %d)\n",
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("  Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
    printf("  Warp size: %d\n", props.warpSize);
    
    // Check memory availability
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("  Memory: %zu MB free of %zu MB total\n", 
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));
    
    // Platform-specific information
    printf("  Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
    printf("  ISA: %s\n", props.gcnArchName);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
#else
    printf("Unknown\n");
#endif
    
    printf("\n");
}

int main() {
    printf("HIP GPU Debugging Examples\n");
    printf("==========================\n");
    
    // Check device capabilities
    checkDeviceLimits();
    
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    // Device arrays
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    
    // Test 1: Basic kernel with debug output
    printf("Test 1: Debug kernel with thread divergence\n");
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Launching kernel with %d blocks of %d threads\n", gridSize, blockSize);
    
    debugKernel<<<gridSize, blockSize>>>(d_input, N);
    
    // Check for kernel errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel launch error: %s\n", hipGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to complete
    HIP_CHECK(hipDeviceSynchronize());
    printf("Debug kernel completed successfully\n\n");
    
    // Test 2: Shared memory example
    printf("Test 2: Shared memory example\n");
    
    sharedMemoryExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    // Check for errors again
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    // Verify results (first few elements)
    printf("Verification (input -> output for first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %.1f -> %.1f\n", i, h_input[i], h_output[i]);
    }
    
    // Test 3: Demonstrate occupancy calculation
    printf("\nTest 3: Occupancy analysis\n");
    int maxActiveBlocks;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, 
                                                           debugKernel, blockSize, 0));
    printf("Max active blocks per CU with block size %d: %d\n", blockSize, maxActiveBlocks);
    
    // Get suggested block size
    int minGridSize, optimalBlockSize;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                                debugKernel, 0, 0));
    printf("Suggested optimal block size: %d\n", optimalBlockSize);
    
    // Test different block sizes
    int testBlockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = sizeof(testBlockSizes) / sizeof(testBlockSizes[0]);
    
    printf("Occupancy for different block sizes:\n");
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
    for (int i = 0; i < numTests; i++) {
        int bs = testBlockSizes[i];
        if (bs > props.maxThreadsPerBlock) continue;
        
        int maxBlocks;
        hipError_t result = hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, 
                                                                         debugKernel, bs, 0);
        if (result == hipSuccess) {
            float occupancy = (maxBlocks * bs / (float)props.maxThreadsPerMultiProcessor) * 100.0f;
            printf("  Block size %d: %d max blocks per CU (%.1f%% occupancy)\n", bs, maxBlocks, occupancy);
        } else {
            printf("  Block size %d: Invalid configuration\n", bs);
        }
    }
    
    // Test 4: Memory bandwidth test
    printf("\nTest 4: Memory bandwidth analysis\n");
    
    // Create HIP events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Test memory copy bandwidth
    HIP_CHECK(hipEventRecord(start, 0));
    HIP_CHECK(hipMemcpy(d_output, d_input, bytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float copyTime;
    HIP_CHECK(hipEventElapsedTime(&copyTime, start, stop));
    
    float bandwidth = (2.0f * bytes / (1024.0f * 1024.0f * 1024.0f)) / (copyTime / 1000.0f);
    printf("Device-to-device memory bandwidth: %.2f GB/s\n", bandwidth);
    
    // Compare with theoretical peak
    float theoreticalBandwidth = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6f;
    float efficiency = (bandwidth / theoreticalBandwidth) * 100.0f;
    printf("Theoretical peak bandwidth: %.2f GB/s\n", theoreticalBandwidth);
    printf("Achieved efficiency: %.1f%%\n", efficiency);
    
    // Test 5: Platform-specific features
    printf("\nTest 5: Platform detection and features\n");
    printf("HIP Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
    printf("Architecture: %s\n", props.gcnArchName);
    printf("Cooperative launch support: %s\n", props.cooperativeLaunch ? "Yes" : "No");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
    printf("Unified addressing: %s\n", props.unifiedAddressing ? "Yes" : "No");
    printf("ECC support: %s\n", props.ECCEnabled ? "Yes" : "No");
#else
    printf("Unknown\n");
#endif
    
    // Cleanup
    free(h_input);
    free(h_output);
    hipFree(d_input);
    hipFree(d_output);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    printf("\nAll tests completed successfully!\n");
    printf("Key debugging tips:\n");
    printf("  1. Use HIP_CHECK macro for all HIP API calls\n");
    printf("  2. Check hipGetLastError() after kernel launches\n");
    printf("  3. Use hipDeviceSynchronize() to catch kernel errors\n");
    printf("  4. Monitor occupancy and memory bandwidth\n");
    printf("  5. Use printf in kernels for debugging (sparingly)\n");
    
    return 0;
}