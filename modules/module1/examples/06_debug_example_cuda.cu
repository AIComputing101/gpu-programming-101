#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Example kernel with potential issues for debugging practice
__global__ void debugKernel(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Debug print (only from first few threads to avoid spam)
    if (tid < 5) {
        printf("Thread %d: processing element %d\n", tid, tid);
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
    
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
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

// Comprehensive error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to check device properties and limits
void checkDeviceLimits() {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    
    printf("Device Information:\n");
    printf("  Name: %s\n", props.name);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Max block dimensions: (%d, %d, %d)\n", 
           props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("  Max grid dimensions: (%d, %d, %d)\n",
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("  Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
    printf("  Warp size: %d\n", props.warpSize);
    
    // Check memory availability
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Memory: %zu MB free of %zu MB total\n", 
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));
    printf("\n");
}

int main() {
    printf("GPU Debugging Examples\n");
    printf("======================\n");
    
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
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Test 1: Basic kernel with debug output
    printf("Test 1: Debug kernel with thread divergence\n");
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Launching kernel with %d blocks of %d threads\n", gridSize, blockSize);
    
    debugKernel<<<gridSize, blockSize>>>(d_input, N);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Debug kernel completed successfully\n\n");
    
    // Test 2: Shared memory example
    printf("Test 2: Shared memory example\n");
    
    sharedMemoryExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    // Check for errors again
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results (first few elements)
    printf("Verification (input -> output for first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %.1f -> %.1f\n", i, h_input[i], h_output[i]);
    }
    
    // Test 3: Demonstrate occupancy calculation
    printf("\nTest 3: Occupancy analysis\n");
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, 
                                                             debugKernel, blockSize, 0));
    printf("Max active blocks per SM with block size %d: %d\n", blockSize, maxActiveBlocks);
    
    // Get suggested block size
    int minGridSize, optimalBlockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                                  debugKernel, 0, 0));
    printf("Suggested optimal block size: %d\n", optimalBlockSize);
    
    // Test different block sizes
    int testBlockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = sizeof(testBlockSizes) / sizeof(testBlockSizes[0]);
    
    printf("Occupancy for different block sizes:\n");
    for (int i = 0; i < numTests; i++) {
        int bs = testBlockSizes[i];
        int maxBlocks;
        cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, 
                                                                           debugKernel, bs, 0);
        if (result == cudaSuccess) {
            printf("  Block size %d: %d max active blocks per SM\n", bs, maxBlocks);
        } else {
            printf("  Block size %d: Invalid configuration\n", bs);
        }
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\nAll tests completed successfully!\n");
    return 0;
}