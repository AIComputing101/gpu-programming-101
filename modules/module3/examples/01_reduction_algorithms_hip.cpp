#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

// Naive reduction - inefficient but educational
__global__ void naiveReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction with divergent warps (inefficient)
    for (int s = 1; s < hipBlockDim_x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
    }
}

// Improved reduction - less warp divergence
__global__ void improvedReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Sequential addressing reduces warp divergence
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
    }
}

// Optimized reduction with multiple elements per thread
__global__ void optimizedReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * (hipBlockDim_x * 2) + hipThreadIdx_x;
    
    // Load two elements per thread to improve bandwidth utilization
    sdata[tid] = 0.0f;
    if (i < n) sdata[tid] += input[i];
    if (i + hipBlockDim_x < n) sdata[tid] += input[i + hipBlockDim_x];
    
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
    }
}

// Platform-specific warp-level reduction
#ifdef __HIP_PLATFORM_AMD__
// AMD wavefront-based reduction (64 threads)
__global__ void wavefrontReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    // Load data
    float value = (i < n) ? input[i] : 0.0f;
    
    // Wavefront-level reduction using ballot and shift operations
    const int wavefront_size = 64;
    int wavefront_id = tid / wavefront_size;
    int lane_id = tid % wavefront_size;
    
    // Reduce within wavefront
    for (int offset = wavefront_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
    }
    
    // Store wavefront results in shared memory
    if (lane_id == 0) {
        sdata[wavefront_id] = value;
    }
    
    __syncthreads();
    
    // Final reduction of wavefront results
    if (tid < (hipBlockDim_x / wavefront_size)) {
        value = sdata[tid];
        
        // Reduce remaining elements
        for (int offset = (hipBlockDim_x / wavefront_size) / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                value += sdata[tid + offset];
            }
        }
        
        if (tid == 0) {
            output[hipBlockIdx_x] = value;
        }
    }
}

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA warp-based reduction (32 threads)
__global__ void warpReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    float value = (i < n) ? input[i] : 0.0f;
    
    const int warp_size = 32;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;
    
    // Warp-level reduction using shuffle operations
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
    }
    
    // Store warp results
    if (lane_id == 0) {
        sdata[warp_id] = value;
    }
    
    __syncthreads();
    
    // Final reduction
    if (tid < (hipBlockDim_x / warp_size)) {
        value = sdata[tid];
        
        for (int offset = (hipBlockDim_x / warp_size) / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                value += sdata[tid + offset];
            }
        }
        
        if (tid == 0) {
            output[hipBlockIdx_x] = value;
        }
    }
}
#endif

// Generic cross-platform reduction
__global__ void crossPlatformReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Standard shared memory reduction works on both platforms
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
    }
}

// Specialized reduction operations
__global__ void minReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    sdata[tid] = (i < n) ? input[i] : FLT_MAX;
    __syncthreads();
    
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
    }
}

__global__ void maxReduction(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    sdata[tid] = (i < n) ? input[i] : -FLT_MAX;
    __syncthreads();
    
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[hipBlockIdx_x] = sdata[0];
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

class ReductionBenchmark {
private:
    float *d_input, *d_output, *d_temp;
    size_t input_size, output_size, temp_size;
    hipEvent_t start, stop;
    
public:
    ReductionBenchmark(size_t n) : input_size(n) {
        output_size = (n + 255) / 256; // Max blocks needed
        temp_size = output_size;
        
        HIP_CHECK(hipMalloc(&d_input, input_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, output_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_temp, temp_size * sizeof(float)));
        
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Initialize with test data
        initializeData();
    }
    
    ~ReductionBenchmark() {
        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
    
    void initializeData() {
        float *h_input = new float[input_size];
        srand(42); // Fixed seed for reproducible results
        
        for (size_t i = 0; i < input_size; i++) {
            h_input[i] = (rand() % 1000) / 100.0f; // Range [0, 10)
        }
        
        HIP_CHECK(hipMemcpy(d_input, h_input, input_size * sizeof(float), 
                           hipMemcpyHostToDevice));
        delete[] h_input;
    }
    
    float testReduction(void (*kernel)(float*, float*, int), const char* name) {
        const int threads = 256;
        const int blocks = (input_size + threads - 1) / threads;
        
        HIP_CHECK(hipEventRecord(start));
        kernel<<<blocks, threads>>>(d_input, d_output, input_size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
        
        // Get partial results and sum on host for verification
        float *h_output = new float[blocks];
        HIP_CHECK(hipMemcpy(h_output, d_output, blocks * sizeof(float), 
                           hipMemcpyDeviceToHost));
        
        float total = 0.0f;
        for (int i = 0; i < blocks; i++) {
            total += h_output[i];
        }
        
        printf("%-25s: %.3f ms, Result: %.2f\n", name, time, total);
        delete[] h_output;
        
        return time;
    }
    
    void runBenchmarks() {
        printf("=== HIP Reduction Algorithm Benchmarks ===\n");
        printf("Array size: %zu elements (%.2f MB)\n", 
               input_size, (input_size * sizeof(float)) / (1024.0 * 1024.0));
        
        // Test different reduction implementations
        float naive_time = testReduction(naiveReduction, "Naive Reduction");
        float improved_time = testReduction(improvedReduction, "Improved Reduction");
        float optimized_time = testReduction(optimizedReduction, "Optimized Reduction");
        float cross_time = testReduction(crossPlatformReduction, "Cross-Platform Reduction");
        
        // Test platform-specific optimization
        float platform_time = 0.0f;
        bool has_platform_specific = false;
        
#ifdef __HIP_PLATFORM_AMD__
        platform_time = testReduction(wavefrontReduction, "AMD Wavefront Reduction");
        has_platform_specific = true;
#elif defined(__HIP_PLATFORM_NVIDIA__)
        platform_time = testReduction(warpReduction, "NVIDIA Warp Reduction");
        has_platform_specific = true;
#endif
        
        printf("\nSpeedup Analysis:\n");
        printf("Improved vs Naive: %.2fx\n", naive_time / improved_time);
        printf("Optimized vs Naive: %.2fx\n", naive_time / optimized_time);
        printf("Cross-Platform vs Naive: %.2fx\n", naive_time / cross_time);
        
        if (has_platform_specific) {
            printf("Platform-Specific vs Naive: %.2fx\n", naive_time / platform_time);
            printf("Platform-Specific vs Optimized: %.2fx\n", optimized_time / platform_time);
        }
    }
    
    void testSpecializedReductions() {
        printf("\n=== Specialized Reductions ===\n");
        
        const int threads = 256;
        const int blocks = (input_size + threads - 1) / threads;
        
        // Test min reduction
        HIP_CHECK(hipEventRecord(start));
        minReduction<<<blocks, threads>>>(d_input, d_output, input_size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float min_time;
        HIP_CHECK(hipEventElapsedTime(&min_time, start, stop));
        
        // Get min result
        float *h_output = new float[blocks];
        HIP_CHECK(hipMemcpy(h_output, d_output, blocks * sizeof(float), 
                           hipMemcpyDeviceToHost));
        
        float min_result = h_output[0];
        for (int i = 1; i < blocks; i++) {
            min_result = fminf(min_result, h_output[i]);
        }
        
        // Test max reduction
        HIP_CHECK(hipEventRecord(start));
        maxReduction<<<blocks, threads>>>(d_input, d_output, input_size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float max_time;
        HIP_CHECK(hipEventElapsedTime(&max_time, start, stop));
        
        HIP_CHECK(hipMemcpy(h_output, d_output, blocks * sizeof(float), 
                           hipMemcpyDeviceToHost));
        
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
        HIP_CHECK(hipMalloc(&d_temp1, temp_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_temp2, temp_size * sizeof(float)));
    }
    
    ~MultiPassReduction() {
        hipFree(d_temp1);
        hipFree(d_temp2);
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
        HIP_CHECK(hipMemcpy(&result, current_output, sizeof(float), 
                           hipMemcpyDeviceToHost));
        return result;
    }
};

void demonstrateMultiPassReduction() {
    printf("\n=== Multi-Pass Reduction Demo ===\n");
    
    const size_t large_n = 64 * 1024 * 1024; // 64M elements
    
    // Allocate and initialize large dataset
    float *d_large_data;
    HIP_CHECK(hipMalloc(&d_large_data, large_n * sizeof(float)));
    
    // Initialize data (all 1.0, so sum should be large_n)
    float *h_temp = new float[large_n];
    for (size_t i = 0; i < large_n; i++) {
        h_temp[i] = 1.0f;
    }
    HIP_CHECK(hipMemcpy(d_large_data, h_temp, large_n * sizeof(float), 
                       hipMemcpyHostToDevice));
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
    
    hipFree(d_large_data);
}

int main() {
    printf("HIP Reduction Algorithms Demonstration\n");
    printf("=====================================\n");
    
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
    printf("Compute units: %d\n", props.multiProcessorCount);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
    printf("Warp size: %d\n", props.warpSize);
    printf("SM count: %d\n", props.multiProcessorCount);
#else
    printf("Unknown\n");
#endif
    
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("Shared memory per block: %zu KB\n", props.sharedMemPerBlock / 1024);
    
    // Run benchmarks with medium-sized dataset
    const size_t test_size = 16 * 1024 * 1024; // 16M elements
    ReductionBenchmark benchmark(test_size);
    
    benchmark.runBenchmarks();
    benchmark.testSpecializedReductions();
    
    // Demonstrate multi-pass reduction for large datasets
    demonstrateMultiPassReduction();
    
    // Platform-specific analysis
    printf("\n=== Platform-Specific Optimizations ===\n");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD GCN/RDNA Optimizations:\n");
    printf("✓ Optimize for 64-thread wavefront execution\n");
    printf("✓ Use LDS (Local Data Share) efficiently\n");
    printf("✓ Consider memory bank conflicts\n");
    printf("✓ Leverage SIMD instruction scheduling\n");
    printf("✓ Utilize high memory bandwidth of HBM\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA Optimizations:\n");
    printf("✓ Optimize for 32-thread warp execution\n");
    printf("✓ Use shared memory with bank conflict avoidance\n");
    printf("✓ Leverage warp-level primitives (__shfl)\n");
    printf("✓ Consider texture cache for read-only data\n");
    printf("✓ Utilize tensor cores when applicable\n");
#endif
    
    // Educational summary
    printf("\n=== HIP Reduction Algorithm Guidelines ===\n");
    printf("✓ CROSS-PLATFORM CONSIDERATIONS:\n");
    printf("  - Use HIP intrinsics for maximum portability\n");
    printf("  - Test on both AMD and NVIDIA hardware\n");
    printf("  - Consider different warp/wavefront sizes\n");
    printf("  - Platform-specific optimizations can provide benefits\n");
    
    printf("\n✓ OPTIMIZATION STRATEGIES:\n");
    printf("  - Minimize warp/wavefront divergence\n");
    printf("  - Use multiple elements per thread for bandwidth\n");
    printf("  - Leverage warp-level primitives when available\n");
    printf("  - Implement multi-pass reduction for large datasets\n");
    
    printf("\n💡 BEST PRACTICES:\n");
    printf("  - Profile on target hardware for best results\n");
    printf("  - Consider using vendor-optimized libraries\n");
    printf("  - Test accuracy with large datasets\n");
    printf("  - Benchmark against CPU implementations\n");
    printf("  - Use appropriate precision for your use case\n");
    
    printf("\nHIP reduction algorithms demonstration completed successfully!\n");
    return 0;
}