#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

namespace cg = cooperative_groups;

// Naive reduction - inefficient but educational
__global__ void naiveReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction with divergent warps (inefficient)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Improved reduction - less warp divergence
__global__ void improvedReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Sequential addressing reduces warp divergence
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction with multiple elements per thread
__global__ void optimizedReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load two elements per thread to improve bandwidth utilization
    sdata[tid] = 0.0f;
    if (i < n) sdata[tid] += input[i];
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Modern warp-level reduction using cooperative groups
__global__ void warpReduction(float *input, float *output, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += warp.shfl_down(value, offset);
    }
    
    // Store partial results in shared memory
    __shared__ float warp_sums[32]; // Max 32 warps per block
    if (warp.thread_rank() == 0) {
        warp_sums[warp.meta_group_rank()] = value;
    }
    
    block.sync();
    
    // Final reduction of warp sums by first warp
    if (warp.meta_group_rank() == 0) {
        value = (threadIdx.x < block.size() / 32) ? warp_sums[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += warp.shfl_down(value, offset);
        }
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = value;
        }
    }
}

// Specialized reduction operations
__global__ void minReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : FLT_MAX;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void maxReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : -FLT_MAX;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Segmented reduction - reduce multiple segments in parallel
__global__ void segmentedReduction(float *input, int *segment_flags, 
                                  float *output, int n) {
    __shared__ float sdata[256];
    __shared__ int flags[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and flags
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    flags[tid] = (i < n) ? segment_flags[i] : 1; // End segment if out of bounds
    
    __syncthreads();
    
    // Segmented reduction
    for (int s = 1; s < blockDim.x; s <<= 1) {
        int idx = 2 * s * tid;
        if (idx + s < blockDim.x) {
            if (flags[idx + s] == 0) { // Not a segment boundary
                sdata[idx] += sdata[idx + s];
                flags[idx] = flags[idx + s];
            }
        }
        __syncthreads();
    }
    
    // Write results for segment heads
    if (tid == 0 || flags[tid] == 1) {
        output[blockIdx.x * blockDim.x + tid] = sdata[tid];
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

class ReductionBenchmark {
private:
    float *d_input, *d_output, *d_temp;
    size_t input_size, output_size, temp_size;
    cudaEvent_t start, stop;
    
public:
    ReductionBenchmark(size_t n) : input_size(n) {
        output_size = (n + 255) / 256; // Max blocks needed
        temp_size = output_size;
        
        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Initialize with test data
        initializeData();
    }
    
    ~ReductionBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void initializeData() {
        float *h_input = new float[input_size];
        srand(42); // Fixed seed for reproducible results
        
        for (size_t i = 0; i < input_size; i++) {
            h_input[i] = (rand() % 1000) / 100.0f; // Range [0, 10)
        }
        
        CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), 
                             cudaMemcpyHostToDevice));
        delete[] h_input;
    }
    
    float testReduction(void (*kernel)(float*, float*, int), const char* name) {
        const int threads = 256;
        const int blocks = (input_size + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<blocks, threads>>>(d_input, d_output, input_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        
        // Get partial results and sum on host for verification
        float *h_output = new float[blocks];
        CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        float total = 0.0f;
        for (int i = 0; i < blocks; i++) {
            total += h_output[i];
        }
        
        printf("%-20s: %.3f ms, Result: %.2f\n", name, time, total);
        delete[] h_output;
        
        return time;
    }
    
    void runBenchmarks() {
        printf("=== Reduction Algorithm Benchmarks ===\n");
        printf("Array size: %zu elements (%.2f MB)\n", 
               input_size, (input_size * sizeof(float)) / (1024.0 * 1024.0));
        
        // Test different reduction implementations
        float naive_time = testReduction(naiveReduction, "Naive Reduction");
        float improved_time = testReduction(improvedReduction, "Improved Reduction");
        float optimized_time = testReduction(optimizedReduction, "Optimized Reduction");
        float warp_time = testReduction(warpReduction, "Warp Reduction");
        
        printf("\nSpeedup Analysis:\n");
        printf("Improved vs Naive: %.2fx\n", naive_time / improved_time);
        printf("Optimized vs Naive: %.2fx\n", naive_time / optimized_time);
        printf("Warp vs Naive: %.2fx\n", naive_time / warp_time);
        printf("Warp vs Optimized: %.2fx\n", optimized_time / warp_time);
    }
    
    void testSpecializedReductions() {
        printf("\n=== Specialized Reductions ===\n");
        
        const int threads = 256;
        const int blocks = (input_size + threads - 1) / threads;
        
        // Test min reduction
        CUDA_CHECK(cudaEventRecord(start));
        minReduction<<<blocks, threads>>>(d_input, d_output, input_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float min_time;
        CUDA_CHECK(cudaEventElapsedTime(&min_time, start, stop));
        
        // Get min result
        float *h_output = new float[blocks];
        CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        float min_result = h_output[0];
        for (int i = 1; i < blocks; i++) {
            min_result = fminf(min_result, h_output[i]);
        }
        
        // Test max reduction
        CUDA_CHECK(cudaEventRecord(start));
        maxReduction<<<blocks, threads>>>(d_input, d_output, input_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float max_time;
        CUDA_CHECK(cudaEventElapsedTime(&max_time, start, stop));
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        float max_result = h_output[0];
        for (int i = 1; i < blocks; i++) {
            max_result = fmaxf(max_result, h_output[i]);
        }
        
        printf("Min Reduction: %.3f ms, Result: %.2f\n", min_time, min_result);
        printf("Max Reduction: %.3f ms, Result: %.2f\n", max_time, max_result);
        
        delete[] h_output;
    }
};

// Multi-pass reduction for very large datasets
class MultiPassReduction {
private:
    float *d_temp1, *d_temp2;
    size_t max_elements;
    
public:
    MultiPassReduction(size_t max_elem) : max_elements(max_elem) {
        size_t temp_size = (max_elem + 511) / 512; // Conservative estimate
        CUDA_CHECK(cudaMalloc(&d_temp1, temp_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp2, temp_size * sizeof(float)));
    }
    
    ~MultiPassReduction() {
        cudaFree(d_temp1);
        cudaFree(d_temp2);
    }
    
    float reduce(float *input, size_t n) {
        float *current_input = input;
        float *current_output = d_temp1;
        size_t current_n = n;
        
        while (current_n > 1) {
            int threads = 256;
            int blocks = (current_n + threads * 2 - 1) / (threads * 2);
            
            if (blocks == 1) {
                // Final reduction
                optimizedReduction<<<1, threads>>>(current_input, current_output, current_n);
                break;
            } else {
                // Intermediate reduction
                optimizedReduction<<<blocks, threads>>>(current_input, current_output, current_n);
                
                // Swap buffers
                if (current_input == input) {
                    current_input = d_temp1;
                    current_output = d_temp2;
                } else if (current_input == d_temp1) {
                    current_input = d_temp2;
                    current_output = d_temp1;
                } else {
                    current_input = d_temp1;
                    current_output = d_temp2;
                }
                
                current_n = blocks;
            }
        }
        
        // Copy final result to host
        float result;
        CUDA_CHECK(cudaMemcpy(&result, current_output, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        return result;
    }
};

void demonstrateMultiPassReduction() {
    printf("\n=== Multi-Pass Reduction Demo ===\n");
    
    const size_t large_n = 100 * 1024 * 1024; // 100M elements
    
    // Allocate and initialize large dataset
    float *d_large_data;
    CUDA_CHECK(cudaMalloc(&d_large_data, large_n * sizeof(float)));
    
    // Initialize with pattern
    const int init_threads = 256;
    const int init_blocks = (large_n + init_threads - 1) / init_threads;
    
    // Simple initialization kernel
    auto init_kernel = [] __device__ (float *data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 1.0f; // Each element contributes 1.0
        }
    };
    
    // Initialize data (all 1.0, so sum should be large_n)
    float *h_temp = new float[large_n];
    for (size_t i = 0; i < large_n; i++) {
        h_temp[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_large_data, h_temp, large_n * sizeof(float), 
                         cudaMemcpyHostToDevice));
    delete[] h_temp;
    
    MultiPassReduction reducer(large_n);
    
    auto start = std::chrono::high_resolution_clock::now();
    float result = reducer.reduce(d_large_data, large_n);
    auto end = std::chrono::high_resolution_clock::now();
    
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Large dataset: %zu elements (%.2f GB)\n", 
           large_n, (large_n * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    printf("Multi-pass reduction time: %.3f ms\n", time);
    printf("Result: %.0f (Expected: %zu)\n", result, large_n);
    printf("Accuracy: %.6f%%\n", (result / large_n) * 100.0);
    
    // Calculate effective bandwidth
    double bytes_read = large_n * sizeof(float);
    double bandwidth = (bytes_read / (1024.0 * 1024.0 * 1024.0)) / (time / 1000.0);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    cudaFree(d_large_data);
}

int main() {
    printf("CUDA Reduction Algorithms Demonstration\n");
    printf("======================================\n");
    
    // Check device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("Warp size: %d\n", props.warpSize);
    
    // Run benchmarks with medium-sized dataset
    const size_t test_size = 16 * 1024 * 1024; // 16M elements
    ReductionBenchmark benchmark(test_size);
    
    benchmark.runBenchmarks();
    benchmark.testSpecializedReductions();
    
    // Demonstrate multi-pass reduction for large datasets
    demonstrateMultiPassReduction();
    
    // Educational summary
    printf("\n=== Reduction Algorithm Guidelines ===\n");
    printf("✓ OPTIMIZATION STRATEGIES:\n");
    printf("  - Minimize warp divergence with sequential addressing\n");
    printf("  - Use multiple elements per thread to improve bandwidth\n");
    printf("  - Leverage warp-level primitives (shuffle operations)\n");
    printf("  - Apply cooperative groups for cleaner code\n");
    printf("  - Implement multi-pass reduction for large datasets\n");
    
    printf("\n✓ PERFORMANCE CONSIDERATIONS:\n");
    printf("  - Memory bandwidth often limits performance\n");
    printf("  - Shared memory usage affects occupancy\n");
    printf("  - Synchronization overhead can be significant\n");
    printf("  - Consider specialized reductions (min, max, etc.)\n");
    
    printf("\n💡 BEST PRACTICES:\n");
    printf("  - Profile to identify bottlenecks\n");
    printf("  - Use appropriate data types for your precision needs\n");
    printf("  - Consider using libraries (CUB, Thrust) for production\n");
    printf("  - Test accuracy with large datasets\n");
    printf("  - Benchmark against CPU implementations\n");
    
    printf("\nReduction algorithms demonstration completed successfully!\n");
    return 0;
}