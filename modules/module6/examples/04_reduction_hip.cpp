/**
 * Module 6: Fundamental Parallel Algorithms - Reduction Operations (HIP)
 * 
 * Comprehensive implementation of parallel reduction algorithms optimized for AMD GPU architectures.
 * This example demonstrates various reduction strategies adapted for ROCm/HIP with wavefront-aware
 * optimizations and performance analysis.
 * 
 * Topics Covered:
 * - Wavefront-aware reduction algorithms
 * - LDS (Local Data Share) optimization
 * - ROCm-specific performance optimizations
 * - Cross-platform compatibility patterns
 * - Multi-pass reduction for large datasets
 * 
 * Performance Targets:
 * - Memory bandwidth efficiency > 80% for memory-bound operations
 * - Wavefront utilization > 90% for compute-intensive reductions
 * - Scalability across different problem sizes
 */

#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <functional>
#include <cfloat>

// AMD GPU typically has 64-thread wavefronts (vs 32-thread warps on NVIDIA)
constexpr int WAVEFRONT_SIZE = 64;

__device__ __forceinline__ float wavefront_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float wavefront_reduce_max(float val) {
    for (int offset = WAVEFRONT_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float wavefront_reduce_min(float val) {
    for (int offset = WAVEFRONT_SIZE/2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down(val, offset));
    }
    return val;
}

// Naive reduction - global memory only
__global__ void reduction_naive(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Store partial sum in global memory
    atomicAdd(output, sum);
}

// LDS (Local Data Share) reduction with bank conflict avoidance
__global__ void reduction_lds_memory(float* input, float* output, int n) {
    __shared__ float sdata[256];  // LDS equivalent to CUDA shared memory
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into LDS with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    
    __syncthreads();
    
    // Perform reduction in LDS
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized LDS reduction avoiding bank conflicts (AMD-specific optimization)
__global__ void reduction_optimized_lds(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load two elements per thread and perform first level of reduction
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    // Continue with grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x * 2; i < n; i += blockDim.x * gridDim.x * 2) {
        sum += input[i];
        if (i + blockDim.x < n) sum += input[i + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Unrolled reduction optimized for AMD architecture
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Final wavefront reduction (64 threads for AMD)
    if (tid < 64) {
        volatile float* vsdata = sdata;
        if (blockDim.x >= 128) vsdata[tid] += vsdata[tid + 64];
        if (blockDim.x >= 64) {
            vsdata[tid] += vsdata[tid + 32];
            vsdata[tid] += vsdata[tid + 16];
            vsdata[tid] += vsdata[tid + 8];
            vsdata[tid] += vsdata[tid + 4];
            vsdata[tid] += vsdata[tid + 2];
            vsdata[tid] += vsdata[tid + 1];
        }
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Wavefront-level primitive reduction using shuffle operations
__global__ void reduction_wavefront_primitive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WAVEFRONT_SIZE;
    int wavefront_id = tid / WAVEFRONT_SIZE;
    
    // Each thread loads data with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Wavefront-level reduction
    sum = wavefront_reduce_sum(sum);
    
    // LDS for inter-wavefront communication (adjust for AMD's typically smaller wavefront count)
    __shared__ float wavefront_results[16];  // Max 16 wavefronts per workgroup typically
    
    // Write wavefront results to LDS
    if (lane == 0 && wavefront_id < 16) {
        wavefront_results[wavefront_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction by first wavefront
    if (wavefront_id == 0) {
        float wavefront_sum = (tid < (blockDim.x / WAVEFRONT_SIZE)) ? wavefront_results[tid] : 0.0f;
        wavefront_sum = wavefront_reduce_sum(wavefront_sum);
        
        if (tid == 0) {
            output[blockIdx.x] = wavefront_sum;
        }
    }
}

// ROCm-optimized reduction with memory coalescing patterns
__global__ void reduction_rocm_optimized(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Coalesced memory access pattern optimized for AMD memory hierarchy
    float sum = 0.0f;
    
    // Process 4 elements per thread for better memory throughput
    int stride = blockDim.x * gridDim.x;
    for (int i = idx * 4; i < n; i += stride * 4) {
        if (i < n) sum += input[i];
        if (i + 1 < n) sum += input[i + 1];
        if (i + 2 < n) sum += input[i + 2];
        if (i + 3 < n) sum += input[i + 3];
    }
    
    // Wavefront-level reduction
    sum = wavefront_reduce_sum(sum);
    
    // Use LDS for inter-wavefront communication
    __shared__ float lds_buffer[256];
    int wavefront_id = tid / WAVEFRONT_SIZE;
    int lane = tid % WAVEFRONT_SIZE;
    
    if (lane == 0) {
        lds_buffer[wavefront_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction
    if (wavefront_id == 0) {
        float final_sum = (tid < (blockDim.x / WAVEFRONT_SIZE)) ? lds_buffer[tid] : 0.0f;
        final_sum = wavefront_reduce_sum(final_sum);
        
        if (tid == 0) {
            output[blockIdx.x] = final_sum;
        }
    }
}

// Multi-pass reduction for very large datasets
float multi_pass_reduction(float* d_input, int n, void (*reduction_kernel)(float*, float*, int)) {
    float* d_temp = d_input;
    float* d_output;
    int current_size = n;
    
    std::vector<float*> temp_buffers;
    
    while (current_size > 1) {
        int num_blocks = std::min(65535, (current_size + 255) / 256);
        
        // Allocate output buffer
        HIP_CHECK(hipMalloc(&d_output, num_blocks * sizeof(float)));
        temp_buffers.push_back(d_output);
        
        // Launch reduction kernel
        int block_size = 256;
        dim3 grid(num_blocks);
        dim3 block(block_size);
        
        hipLaunchKernelGGL(reduction_kernel, grid, block, 0, 0, d_temp, d_output, current_size);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        d_temp = d_output;
        current_size = num_blocks;
    }
    
    // Get final result
    float result;
    HIP_CHECK(hipMemcpy(&result, d_temp, sizeof(float), hipMemcpyDeviceToHost));
    
    // Clean up temporary buffers
    for (auto ptr : temp_buffers) {
        HIP_CHECK(hipFree(ptr));
    }
    
    return result;
}

// Performance measurement utilities
class PerformanceTimer {
private:
    hipEvent_t start_event, stop_event;
    
public:
    PerformanceTimer() {
        HIP_CHECK(hipEventCreate(&start_event));
        HIP_CHECK(hipEventCreate(&stop_event));
    }
    
    ~PerformanceTimer() {
        HIP_CHECK(hipEventDestroy(start_event));
        HIP_CHECK(hipEventDestroy(stop_event));
    }
    
    void start() {
        HIP_CHECK(hipEventRecord(start_event));
    }
    
    float stop() {
        HIP_CHECK(hipEventRecord(stop_event));
        HIP_CHECK(hipEventSynchronize(stop_event));
        
        float elapsed_ms;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

// CPU reference implementation
float cpu_reduction(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

// Test framework
struct ReductionTest {
    std::string name;
    std::function<float(float*, int)> gpu_function;
    bool requires_multi_pass;
    
    ReductionTest(const std::string& n, std::function<float(float*, int)> func, bool multi = false) 
        : name(n), gpu_function(func), requires_multi_pass(multi) {}
};

void run_reduction_benchmarks() {
    std::cout << "\n=== HIP Reduction Algorithms Benchmark ===\n";
    
    // Test different problem sizes
    std::vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    // Initialize test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int size : sizes) {
        std::cout << "\nTesting with " << size << " elements:\n";
        std::cout << std::string(50, '-') << "\n";
        
        // Generate test data
        std::vector<float> h_data(size);
        for (int i = 0; i < size; ++i) {
            h_data[i] = dis(gen);
        }
        
        // Allocate GPU memory
        float* d_data;
        HIP_CHECK(hipMalloc(&d_data, size * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_data, h_data.data(), size * sizeof(float), hipMemcpyHostToDevice));
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        float cpu_result = cpu_reduction(h_data);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        std::cout << "CPU Reference:           " << std::fixed << std::setprecision(6) 
                  << cpu_result << " (Time: " << std::setprecision(3) << cpu_time << " ms)\n";
        
        // Test different GPU implementations
        std::vector<ReductionTest> tests = {
            ReductionTest("LDS Memory", [](float* d_input, int n) {
                return multi_pass_reduction(d_input, n, reduction_lds_memory);
            }, true),
            ReductionTest("Optimized LDS", [](float* d_input, int n) {
                return multi_pass_reduction(d_input, n, reduction_optimized_lds);
            }, true),
            ReductionTest("Wavefront Primitive", [d_data](float* d_input, int n) {
                int num_blocks = std::min(65535, (n + 255) / 256);
                float* d_output;
                HIP_CHECK(hipMalloc(&d_output, num_blocks * sizeof(float)));
                
                dim3 grid(num_blocks);
                dim3 block(256);
                hipLaunchKernelGGL(reduction_wavefront_primitive, grid, block, 0, 0, d_input, d_output, n);
                HIP_CHECK(hipDeviceSynchronize());
                
                float result = multi_pass_reduction(d_output, num_blocks, reduction_wavefront_primitive);
                HIP_CHECK(hipFree(d_output));
                return result;
            }),
            ReductionTest("ROCm Optimized", [d_data](float* d_input, int n) {
                int num_blocks = std::min(65535, (n + 255) / 256);
                float* d_output;
                HIP_CHECK(hipMalloc(&d_output, num_blocks * sizeof(float)));
                
                dim3 grid(num_blocks);
                dim3 block(256);
                hipLaunchKernelGGL(reduction_rocm_optimized, grid, block, 0, 0, d_input, d_output, n);
                HIP_CHECK(hipDeviceSynchronize());
                
                float result = multi_pass_reduction(d_output, num_blocks, reduction_rocm_optimized);
                HIP_CHECK(hipFree(d_output));
                return result;
            })
        };
        
        for (auto& test : tests) {
            PerformanceTimer timer;
            
            timer.start();
            float gpu_result = test.gpu_function(d_data, size);
            float gpu_time = timer.stop();
            
            float error = std::abs(gpu_result - cpu_result) / std::abs(cpu_result);
            float speedup = cpu_time / gpu_time;
            
            // Calculate memory bandwidth (GB/s)
            double bytes_transferred = size * sizeof(float);
            double bandwidth = bytes_transferred / (gpu_time * 1e6);  // GB/s
            
            std::cout << std::setw(20) << test.name << ": " 
                      << std::fixed << std::setprecision(6) << gpu_result
                      << " (Error: " << std::scientific << std::setprecision(2) << error
                      << ", Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms"
                      << ", Speedup: " << std::setprecision(1) << speedup << "x"
                      << ", BW: " << std::setprecision(1) << bandwidth << " GB/s)\n";
        }
        
        HIP_CHECK(hipFree(d_data));
    }
}

// Specialized reduction operations for AMD GPUs
__global__ void reduction_max_kernel(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    sdata[tid] = max_val;
    
    __syncthreads();
    
    // Wavefront-level max reduction (64 threads)
    if (tid < WAVEFRONT_SIZE) {
        max_val = wavefront_reduce_max(sdata[tid]);
        if (tid == 0) {
            atomicMax((int*)output, __float_as_int(max_val));
        }
    }
}

__global__ void reduction_min_max_kernel(float* input, float* min_output, float* max_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        local_min = fminf(local_min, input[i]);
        local_max = fmaxf(local_max, input[i]);
    }
    
    // Wavefront-level reductions
    local_min = wavefront_reduce_min(local_min);
    local_max = wavefront_reduce_max(local_max);
    
    // Store results (simplified - should use proper reduction)
    if ((threadIdx.x % WAVEFRONT_SIZE) == 0) {
        atomicMin((int*)min_output, __float_as_int(local_min));
        atomicMax((int*)max_output, __float_as_int(local_max));
    }
}

void demonstrate_specialized_reductions() {
    std::cout << "\n=== Specialized Reduction Operations ===\n";
    
    const int n = 1000000;
    
    // Generate test data
    std::vector<float> h_data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    
    for (int i = 0; i < n; ++i) {
        h_data[i] = dis(gen);
    }
    
    // Allocate GPU memory
    float *d_data, *d_min, *d_max;
    HIP_CHECK(hipMalloc(&d_data, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_min, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_max, sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    // Initialize min/max values
    float init_min = FLT_MAX, init_max = -FLT_MAX;
    HIP_CHECK(hipMemcpy(d_min, &init_min, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_max, &init_max, sizeof(float), hipMemcpyHostToDevice));
    
    // Launch kernel
    dim3 grid(256);
    dim3 block(256);
    hipLaunchKernelGGL(reduction_min_max_kernel, grid, block, 0, 0, d_data, d_min, d_max, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Get results
    float gpu_min, gpu_max;
    HIP_CHECK(hipMemcpy(&gpu_min, d_min, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&gpu_max, d_max, sizeof(float), hipMemcpyDeviceToHost));
    
    // CPU reference
    auto cpu_min_max = std::minmax_element(h_data.begin(), h_data.end());
    
    std::cout << "Min value - GPU: " << gpu_min << ", CPU: " << *cpu_min_max.first 
              << " (Error: " << std::abs(gpu_min - *cpu_min_max.first) << ")\n";
    std::cout << "Max value - GPU: " << gpu_max << ", CPU: " << *cpu_min_max.second
              << " (Error: " << std::abs(gpu_max - *cpu_min_max.second) << ")\n";
    
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_min));
    HIP_CHECK(hipFree(d_max));
}

int main() {
    std::cout << "HIP Reduction Algorithms - Comprehensive Implementation\n";
    std::cout << "=======================================================\n";
    
    // Check HIP device properties
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Wavefront Size: " << WAVEFRONT_SIZE << "\n";
    std::cout << "Max Work Group Size: " << props.maxThreadsPerBlock << "\n\n";
    
    try {
        run_reduction_benchmarks();
        demonstrate_specialized_reductions();
        
        std::cout << "\n=== Performance Analysis Summary ===\n";
        std::cout << "1. Wavefront primitives optimized for 64-thread AMD wavefronts\n";
        std::cout << "2. LDS (Local Data Share) optimization crucial for performance\n";
        std::cout << "3. Memory coalescing patterns optimized for AMD memory hierarchy\n";
        std::cout << "4. ROCm-specific optimizations provide best performance\n";
        std::cout << "5. Multi-pass reduction necessary for very large datasets\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}