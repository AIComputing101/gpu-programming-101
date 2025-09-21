/**
 * Module 6: Fundamental Parallel Algorithms - Prefix Sum (Scan) Operations (CUDA)
 * 
 * Comprehensive implementation of parallel prefix sum algorithms optimized for GPU architectures.
 * This example demonstrates various scan strategies including Hillis-Steele, Blelloch, and
 * hierarchical approaches with detailed performance analysis.
 * 
 * Topics Covered:
 * - Inclusive and exclusive prefix sums
 * - Hillis-Steele scan (work-inefficient but step-efficient)
 * - Blelloch scan (work-efficient with up-sweep and down-sweep phases)
 * - Large array scan with hierarchical approaches
 * - Applications: stream compaction, radix sort support
 * 
 * Performance Targets:
 * - Work complexity: O(N) for Blelloch, O(N log N) for Hillis-Steele
 * - Step complexity: O(log N) for both algorithms
 * - Memory bandwidth efficiency > 70% for large arrays
 * - Correctness validation against CPU reference
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
#include <numeric>

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

// Bank conflict avoidance for shared memory access
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// Forward declarations
__global__ void blelloch_scan_with_totals(float* input, float* output, float* block_totals, int n);
__global__ void add_increments(float* data, float* increments, int n);

// Hillis-Steele Scan (Inclusive) - Work inefficient but step efficient
__global__ void hillis_steele_scan_inclusive(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;  // Identity element for addition
    }
    
    __syncthreads();
    
    // Up-sweep phase (parallel prefix sum)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride && idx < n) {
            val = temp[tid - stride];
        }
        
        __syncthreads();
        
        if (tid >= stride && idx < n) {
            temp[tid] += val;
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// Hillis-Steele Scan (Exclusive) - Shift results right
__global__ void hillis_steele_scan_exclusive(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Perform inclusive scan first
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride && idx < n) {
            val = temp[tid - stride];
        }
        
        __syncthreads();
        
        if (tid >= stride && idx < n) {
            temp[tid] += val;
        }
        
        __syncthreads();
    }
    
    // Convert to exclusive by shifting right
    __syncthreads();
    if (idx < n) {
        if (tid == 0) {
            output[idx] = 0.0f;  // First element is 0 for exclusive scan
        } else {
            output[idx] = temp[tid - 1];
        }
    }
}

// Blelloch Scan (Work-efficient) - Up-sweep and down-sweep phases
__global__ void blelloch_scan_exclusive(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Up-sweep phase (reduce phase)
    int n_threads = blockDim.x;
    for (int stride = 1; stride < n_threads; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n_threads) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element (set to identity)
    if (tid == 0) {
        temp[n_threads - 1] = 0.0f;
    }
    
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = n_threads / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n_threads) {
            float t = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = t;
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// Optimized Blelloch scan with bank conflict avoidance
__global__ void blelloch_scan_optimized(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bank conflict free loading
    int ai = tid;
    int bi = tid + (n / 2);
    int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
    int bank_offset_b = CONFLICT_FREE_OFFSET(bi);
    
    if (idx < n) {
        temp[ai + bank_offset_a] = input[idx];
    } else {
        temp[ai + bank_offset_a] = 0.0f;
    }
    
    if (idx + (n / 2) < n) {
        temp[bi + bank_offset_b] = input[idx + (n / 2)];
    } else {
        temp[bi + bank_offset_b] = 0.0f;
    }
    
    int offset = 1;
    
    // Up-sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            temp[bi] += temp[ai];
        }
        
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        int last_idx = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        temp[last_idx] = 0.0f;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results to global memory
    if (idx < n) {
        output[idx] = temp[ai + bank_offset_a];
    }
    if (idx + (n / 2) < n) {
        output[idx + (n / 2)] = temp[bi + bank_offset_b];
    }
}

// Warp-level scan using shuffle operations (CUDA 9.0+)
__device__ float warp_scan_inclusive(float val) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x >= offset) val += temp;
    }
    return val;
}

__device__ float warp_scan_exclusive(float val) {
    float inclusive = warp_scan_inclusive(val);
    return __shfl_up_sync(0xffffffff, inclusive, 1, warpSize);
}

__global__ void warp_scan_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level inclusive scan
    float warp_result = warp_scan_inclusive(val);
    
    // Shared memory for warp results
    __shared__ float warp_results[32];  // Max 32 warps per block
    
    // Store warp totals
    if (lane == warpSize - 1) {
        warp_results[warp_id] = warp_result;
    }
    
    __syncthreads();
    
    // Scan the warp results (only first warp participates)
    if (warp_id == 0) {
        float warp_sum = (threadIdx.x < blockDim.x / warpSize) ? warp_results[threadIdx.x] : 0.0f;
        warp_sum = warp_scan_exclusive(warp_sum);
        warp_results[threadIdx.x] = warp_sum;
    }
    
    __syncthreads();
    
    // Add warp offset to get final result
    float result = warp_result;
    if (warp_id > 0) {
        result += warp_results[warp_id];
    }
    
    // Convert to exclusive scan
    float exclusive_result = __shfl_up_sync(0xffffffff, result, 1);
    if (lane == 0) exclusive_result = 0.0f;
    
    if (idx < n) {
        output[idx] = exclusive_result;
    }
}

// Large array scan using hierarchical approach
void hierarchical_scan(float* d_input, float* d_output, int n) {
    const int BLOCK_SIZE = 512;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate temporary arrays
    float* d_block_sums;
    float* d_block_increments;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_increments, num_blocks * sizeof(float)));
    
    // Phase 1: Scan each block and store block sums
    dim3 grid1(num_blocks);
    dim3 block1(BLOCK_SIZE);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    blelloch_scan_with_totals<<<grid1, block1, shared_mem_size>>>(
        d_input, d_output, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 2: Scan the block sums
    if (num_blocks > 1) {
        hierarchical_scan(d_block_sums, d_block_increments, num_blocks);
        
        // Phase 3: Add increments to each block
        add_increments<<<grid1, block1>>>(d_output, d_block_increments, n);
        CUDA_CHECK(cudaGetLastError());
    }
    
    cudaFree(d_block_sums);
    cudaFree(d_block_increments);
}

// Helper kernel for hierarchical scan - scan with block totals
__global__ void blelloch_scan_with_totals(float* input, float* output, float* block_totals, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and perform scan (similar to blelloch_scan_exclusive)
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Blelloch scan implementation (abbreviated)
    // ... (same as blelloch_scan_exclusive)
    
    // Store the total for this block
    if (tid == 0 && blockIdx.x < gridDim.x) {
        block_totals[blockIdx.x] = temp[blockDim.x - 1] + input[min(idx + blockDim.x - 1, n - 1)];
    }
    
    // Write result
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// Helper kernel to add increments to blocks
__global__ void add_increments(float* data, float* increments, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && blockIdx.x > 0) {
        data[idx] += increments[blockIdx.x - 1];
    }
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
std::vector<float> cpu_exclusive_scan(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    std::exclusive_scan(input.begin(), input.end(), output.begin(), 0.0f);
    return output;
}

std::vector<float> cpu_inclusive_scan(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    std::inclusive_scan(input.begin(), input.end(), output.begin());
    return output;
}

// Test framework
void test_scan_correctness() {
    std::cout << "\n=== Scan Correctness Tests ===\n";
    
    // Test with small known arrays
    std::vector<float> test_input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected_exclusive = {0, 1, 3, 6, 10, 15, 21, 28};
    std::vector<float> expected_inclusive = {1, 3, 6, 10, 15, 21, 28, 36};
    
    int n = test_input.size();
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, test_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test exclusive scan
    dim3 grid(1);
    dim3 block(8);
    size_t shared_mem = 8 * sizeof(float);
    
    blelloch_scan_exclusive<<<grid, block, shared_mem>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> gpu_result(n);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Exclusive Scan Test:\n";
    std::cout << "Input:    ";
    for (float val : test_input) std::cout << std::setw(3) << val << " ";
    std::cout << "\nExpected: ";
    for (float val : expected_exclusive) std::cout << std::setw(3) << val << " ";
    std::cout << "\nGPU:      ";
    for (float val : gpu_result) std::cout << std::setw(3) << val << " ";
    
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i] - expected_exclusive[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << " -> " << (correct ? "PASS" : "FAIL") << "\n";
    
    // Test inclusive scan
    hillis_steele_scan_inclusive<<<grid, block, shared_mem>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "\nInclusive Scan Test:\n";
    std::cout << "Input:    ";
    for (float val : test_input) std::cout << std::setw(3) << val << " ";
    std::cout << "\nExpected: ";
    for (float val : expected_inclusive) std::cout << std::setw(3) << val << " ";
    std::cout << "\nGPU:      ";
    for (float val : gpu_result) std::cout << std::setw(3) << val << " ";
    
    correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i] - expected_inclusive[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << " -> " << (correct ? "PASS" : "FAIL") << "\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void run_scan_benchmarks() {
    std::cout << "\n=== CUDA Scan Algorithms Benchmark ===\n";
    
    std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    
    for (int size : sizes) {
        std::cout << "\nTesting with " << size << " elements:\n";
        std::cout << std::string(50, '-') << "\n";
        
        // Generate random test data
        std::vector<float> h_input(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 10.0f);
        
        for (int i = 0; i < size; ++i) {
            h_input[i] = dis(gen);
        }
        
        // CPU reference timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_result = cpu_exclusive_scan(h_input);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        std::cout << "CPU Reference:     " << std::fixed << std::setprecision(3) 
                  << cpu_time << " ms\n";
        
        // GPU memory allocation
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Test different scan implementations
        struct ScanTest {
            std::string name;
            std::function<void()> kernel_launch;
        };
        
        std::vector<ScanTest> tests;
        
        // Only test algorithms that can handle the array size
        if (size <= 1024) {
            tests.push_back({"Hillis-Steele", [&]() {
                dim3 grid(1);
                dim3 block(size);
                hillis_steele_scan_exclusive<<<grid, block, size * sizeof(float)>>>(d_input, d_output, size);
            }});
            
            tests.push_back({"Blelloch", [&]() {
                dim3 grid(1);
                dim3 block(size);
                blelloch_scan_exclusive<<<grid, block, size * sizeof(float)>>>(d_input, d_output, size);
            }});
        }
        
        tests.push_back({"Warp-level", [&]() {
            int blocks = (size + 512 - 1) / 512;
            dim3 grid(blocks);
            dim3 block(512);
            warp_scan_kernel<<<grid, block>>>(d_input, d_output, size);
        }});
        
        for (auto& test : tests) {
            PerformanceTimer timer;
            
            timer.start();
            test.kernel_launch();
            CUDA_CHECK(cudaGetLastError());
            float gpu_time = timer.stop();
            
            // Verify correctness for smaller arrays
            if (size <= 16384) {
                std::vector<float> gpu_result(size);
                CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
                
                float max_error = 0.0f;
                for (int i = 0; i < size; ++i) {
                    max_error = std::max(max_error, std::abs(gpu_result[i] - cpu_result[i]));
                }
                
                float speedup = cpu_time / gpu_time;
                double bandwidth = (size * sizeof(float) * 2) / (gpu_time * 1e6);  // Read + Write
                
                std::cout << std::setw(15) << test.name << ": " 
                          << std::fixed << std::setprecision(3) << gpu_time << " ms"
                          << " (Speedup: " << std::setprecision(1) << speedup << "x"
                          << ", Error: " << std::scientific << std::setprecision(2) << max_error
                          << ", BW: " << std::fixed << std::setprecision(1) << bandwidth << " GB/s)\n";
            } else {
                float speedup = cpu_time / gpu_time;
                double bandwidth = (size * sizeof(float) * 2) / (gpu_time * 1e6);
                
                std::cout << std::setw(15) << test.name << ": " 
                          << std::fixed << std::setprecision(3) << gpu_time << " ms"
                          << " (Speedup: " << std::setprecision(1) << speedup << "x"
                          << ", BW: " << std::setprecision(1) << bandwidth << " GB/s)\n";
            }
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
}

// Stream compaction example using prefix sum
__global__ void mark_valid_elements(float* input, int* marks, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        marks[idx] = (input[idx] > threshold) ? 1 : 0;
    }
}

__global__ void compact_array(float* input, int* scan_result, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && input[idx] > 0) {  // Assuming positive values are valid
        int output_pos = scan_result[idx];
        output[output_pos] = input[idx];
    }
}

void demonstrate_stream_compaction() {
    std::cout << "\n=== Stream Compaction Example ===\n";
    
    const int n = 1000;
    const float threshold = 5.0f;
    
    // Generate test data with some elements below threshold
    std::vector<float> h_input(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    
    for (int i = 0; i < n; ++i) {
        h_input[i] = dis(gen);
    }
    
    // Count valid elements for verification
    int expected_count = std::count_if(h_input.begin(), h_input.end(), 
                                      [threshold](float x) { return x > threshold; });
    
    // GPU memory allocation
    float *d_input, *d_output;
    int *d_marks, *d_scan;
    
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_marks, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Step 1: Mark valid elements
    dim3 grid((n + 255) / 256);
    dim3 block(256);
    mark_valid_elements<<<grid, block>>>(d_input, d_marks, threshold, n);
    
    // Step 2: Exclusive scan of marks
    // (Using simplified version - in practice would use hierarchical scan)
    warp_scan_kernel<<<grid, block>>>((float*)d_marks, (float*)d_scan, n);
    
    // Step 3: Compact array
    compact_array<<<grid, block>>>(d_input, d_scan, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verify results
    std::vector<int> h_scan(n);
    CUDA_CHECK(cudaMemcpy(h_scan.data(), d_scan, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    int gpu_count = h_scan[n-1];  // Last element gives total count
    
    std::cout << "Input elements: " << n << "\n";
    std::cout << "Threshold: " << threshold << "\n";
    std::cout << "Valid elements - Expected: " << expected_count << ", GPU: " << gpu_count << "\n";
    std::cout << "Compaction " << (expected_count == gpu_count ? "PASSED" : "FAILED") << "\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_marks);
    cudaFree(d_scan);
}

int main() {
    std::cout << "CUDA Prefix Sum (Scan) Algorithms - Comprehensive Implementation\n";
    std::cout << "================================================================\n";
    
    // Check CUDA device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory per Block: " << props.sharedMemPerBlock << " bytes\n\n";
    
    try {
        test_scan_correctness();
        run_scan_benchmarks();
        demonstrate_stream_compaction();
        
        std::cout << "\n=== Scan Algorithm Summary ===\n";
        std::cout << "1. Hillis-Steele: O(N log N) work, O(log N) steps, simpler implementation\n";
        std::cout << "2. Blelloch: O(N) work, O(log N) steps, work-efficient\n";
        std::cout << "3. Warp-level: Best performance for modern GPUs\n";
        std::cout << "4. Hierarchical: Required for very large arrays\n";
        std::cout << "5. Applications: Stream compaction, sorting, parallel algorithms\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}