/**
 * Module 6: Fundamental Parallel Algorithms - Reduction Operations (CUDA)
 * 
 * Comprehensive implementation of parallel reduction algorithms optimized for GPU architectures.
 * This example demonstrates various reduction strategies from naive approaches to highly
 * optimized warp-level primitives with performance analysis.
 * 
 * Topics Covered:
 * - Naive global memory reduction
 * - Shared memory optimization with bank conflict avoidance
 * - Warp-level primitives and shuffle operations
 * - Multiple passes for large datasets
 * - Performance comparison and analysis
 * 
 * Performance Targets:
 * - Memory bandwidth efficiency > 80% for memory-bound operations
 * - Warp utilization > 90% for compute-intensive reductions
 * - Scalability across different problem sizes
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>

namespace cg = cooperative_groups;

// Utility macros and functions
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
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

// Shared memory reduction with bank conflict avoidance
__global__ void reduction_shared_memory(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    
    __syncthreads();
    
    // Perform reduction in shared memory
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

// Optimized shared memory reduction avoiding bank conflicts
__global__ void reduction_optimized_shared(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
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
    
    // Unrolled reduction for common block sizes
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
    
    // Final warp reduction
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Warp-level primitive reduction using shuffle operations
__global__ void reduction_warp_primitive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    // Each thread loads data with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Shared memory for inter-warp communication
    __shared__ float warp_results[32];  // Max 32 warps per block
    
    // Write warp results to shared memory
    if (lane == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float warp_sum = (tid < (blockDim.x / warpSize)) ? warp_results[tid] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        
        if (tid == 0) {
            output[blockIdx.x] = warp_sum;
        }
    }
}

// Cooperative groups reduction (requires compute capability 6.0+)
__global__ void reduction_cooperative_groups(float* input, float* output, int n) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Tile-level (warp-level) reduction
    sum = cg::reduce(tile, sum, cg::plus<float>());
    
    // Shared memory for storing tile results
    __shared__ float tile_results[32];
    
    // Store tile results
    if (tile.thread_rank() == 0) {
        tile_results[tile.meta_group_rank()] = sum;
    }
    
    block.sync();
    
    // Block-level reduction using first tile
    if (tile.meta_group_rank() == 0) {
        float block_sum = (tile.thread_rank() < (blockDim.x / tile.size())) ? 
                         tile_results[tile.thread_rank()] : 0.0f;
        block_sum = cg::reduce(tile, block_sum, cg::plus<float>());
        
        if (tile.thread_rank() == 0) {
            output[blockIdx.x] = block_sum;
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
        CUDA_CHECK(cudaMalloc(&d_output, num_blocks * sizeof(float)));
        temp_buffers.push_back(d_output);
        
        // Launch reduction kernel
        int block_size = 256;
        dim3 grid(num_blocks);
        dim3 block(block_size);
        
        size_t shared_mem_size = block_size * sizeof(float);
        reduction_kernel<<<grid, block, shared_mem_size>>>(d_temp, d_output, current_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        d_temp = d_output;
        current_size = num_blocks;
    }
    
    // Get final result
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Clean up temporary buffers
    for (auto ptr : temp_buffers) {
        cudaFree(ptr);
    }
    
    return result;
}

// Performance measurement utilities
class PerformanceTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    PerformanceTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~PerformanceTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
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
    std::cout << "\n=== CUDA Reduction Algorithms Benchmark ===\n";
    
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
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        float cpu_result = cpu_reduction(h_data);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        std::cout << "CPU Reference:           " << std::fixed << std::setprecision(6) 
                  << cpu_result << " (Time: " << std::setprecision(3) << cpu_time << " ms)\n";
        
        // Test different GPU implementations
        std::vector<ReductionTest> tests = {
            ReductionTest("Shared Memory", [](float* d_input, int n) {
                return multi_pass_reduction(d_input, n, reduction_shared_memory);
            }, true),
            ReductionTest("Optimized Shared", [](float* d_input, int n) {
                return multi_pass_reduction(d_input, n, reduction_optimized_shared);
            }, true),
            ReductionTest("Warp Primitive", [d_data](float* d_input, int n) {
                int num_blocks = std::min(65535, (n + 255) / 256);
                float* d_output;
                CUDA_CHECK(cudaMalloc(&d_output, num_blocks * sizeof(float)));
                
                dim3 grid(num_blocks);
                dim3 block(256);
                reduction_warp_primitive<<<grid, block>>>(d_input, d_output, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                float result = multi_pass_reduction(d_output, num_blocks, reduction_warp_primitive);
                cudaFree(d_output);
                return result;
            }),
            ReductionTest("Cooperative Groups", [d_data](float* d_input, int n) {
                int num_blocks = std::min(65535, (n + 255) / 256);
                float* d_output;
                CUDA_CHECK(cudaMalloc(&d_output, num_blocks * sizeof(float)));
                
                dim3 grid(num_blocks);
                dim3 block(256);
                reduction_cooperative_groups<<<grid, block>>>(d_input, d_output, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                float result = multi_pass_reduction(d_output, num_blocks, reduction_cooperative_groups);
                cudaFree(d_output);
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
        
        cudaFree(d_data);
    }
}

// Specialized reduction operations
__global__ void reduction_max_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    sdata[tid] = max_val;
    
    __syncthreads();
    
    // Warp-level max reduction
    for (int s = 1; s < warpSize; s *= 2) {
        float val = __shfl_xor_sync(0xffffffff, sdata[tid], s);
        sdata[tid] = fmaxf(sdata[tid], val);
    }
    
    // Inter-warp reduction
    if ((tid & (warpSize - 1)) == 0) {
        atomicMax((int*)output, __float_as_int(sdata[tid]));
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
    
    // Warp-level reductions
    local_min = warp_reduce_min(local_min);
    local_max = warp_reduce_max(local_max);
    
    // Store results (simplified - should use proper reduction)
    if ((threadIdx.x & (warpSize - 1)) == 0) {
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
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize min/max values
    float init_min = FLT_MAX, init_max = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 grid(256);
    dim3 block(256);
    reduction_min_max_kernel<<<grid, block>>>(d_data, d_min, d_max, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    float gpu_min, gpu_max;
    CUDA_CHECK(cudaMemcpy(&gpu_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gpu_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference
    auto cpu_min_max = std::minmax_element(h_data.begin(), h_data.end());
    
    std::cout << "Min value - GPU: " << gpu_min << ", CPU: " << *cpu_min_max.first 
              << " (Error: " << std::abs(gpu_min - *cpu_min_max.first) << ")\n";
    std::cout << "Max value - GPU: " << gpu_max << ", CPU: " << *cpu_min_max.second
              << " (Error: " << std::abs(gpu_max - *cpu_min_max.second) << ")\n";
    
    cudaFree(d_data);
    cudaFree(d_min);
    cudaFree(d_max);
}

int main() {
    std::cout << "CUDA Reduction Algorithms - Comprehensive Implementation\n";
    std::cout << "=========================================================\n";
    
    // Check CUDA device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Warp Size: " << props.warpSize << "\n\n";
    
    try {
        run_reduction_benchmarks();
        demonstrate_specialized_reductions();
        
        std::cout << "\n=== Performance Analysis Summary ===\n";
        std::cout << "1. Warp primitives provide best performance for most cases\n";
        std::cout << "2. Cooperative groups offer cleaner, more maintainable code\n";
        std::cout << "3. Shared memory optimization crucial for memory-bound reductions\n";
        std::cout << "4. Multi-pass reduction necessary for very large datasets\n";
        std::cout << "5. Memory bandwidth typically limits performance\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}