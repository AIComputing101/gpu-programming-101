/**
 * Module 6: Fundamental Parallel Algorithms - Prefix Sum (Scan) Operations (HIP)
 * 
 * Comprehensive implementation of parallel prefix sum algorithms optimized for AMD GPU architectures.
 * This example demonstrates various scan strategies adapted for ROCm/HIP with wavefront-aware
 * optimizations and LDS utilization patterns.
 * 
 * Topics Covered:
 * - Wavefront-aware scan algorithms (64-thread wavefronts)
 * - LDS (Local Data Share) optimization for AMD GPUs
 * - Inclusive and exclusive prefix sums
 * - Hierarchical approaches for large arrays
 * - Stream compaction applications
 * 
 * Performance Targets:
 * - Work complexity: O(N) for work-efficient algorithms
 * - Step complexity: O(log N) for parallel algorithms
 * - Memory bandwidth efficiency > 70% for large arrays
 * - Wavefront utilization > 90%
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <functional>

// Utility macros and functions
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// AMD GPU typically has 64-thread wavefronts
constexpr int WAVEFRONT_SIZE = 64;

// Wavefront-level scan using shuffle operations
__device__ float wavefront_scan_inclusive(float val) {
    for (int offset = 1; offset < WAVEFRONT_SIZE; offset *= 2) {
        float temp = __shfl_up(val, offset);
        if ((threadIdx.x % WAVEFRONT_SIZE) >= offset) val += temp;
    }
    return val;
}

__device__ float wavefront_scan_exclusive(float val) {
    float inclusive = wavefront_scan_inclusive(val);
    return __shfl_up(inclusive, 1);
}

// Hillis-Steele Scan (Inclusive) - Work inefficient but step efficient
__global__ void hillis_steele_scan_inclusive(float* input, float* output, int n) {
    __shared__ float temp[256];  // LDS memory
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into LDS
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Up-sweep phase optimized for AMD wavefronts
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

// Hillis-Steele Scan (Exclusive)
__global__ void hillis_steele_scan_exclusive(float* input, float* output, int n) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into LDS
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
            output[idx] = 0.0f;
        } else {
            output[idx] = temp[tid - 1];
        }
    }
}

// Blelloch Scan (Work-efficient) adapted for AMD architecture
__global__ void blelloch_scan_exclusive(float* input, float* output, int n) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into LDS
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
    
    // Clear the last element
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

// LDS bank conflict free optimization for AMD GPUs
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * NUM_BANKS))

__global__ void blelloch_scan_lds_optimized(float* input, float* output, int n) {
    __shared__ float temp[512 + 512/NUM_BANKS];  // Extra space for conflict avoidance
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bank conflict free loading for AMD LDS
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

// Wavefront-level scan kernel optimized for AMD
__global__ void wavefront_scan_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WAVEFRONT_SIZE;
    int wavefront_id = threadIdx.x / WAVEFRONT_SIZE;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Wavefront-level inclusive scan
    float wavefront_result = wavefront_scan_inclusive(val);
    
    // LDS for wavefront results (AMD typically has fewer wavefronts per workgroup)
    __shared__ float wavefront_results[16];  // Max 16 wavefronts per workgroup
    
    // Store wavefront totals
    if (lane == WAVEFRONT_SIZE - 1 && wavefront_id < 16) {
        wavefront_results[wavefront_id] = wavefront_result;
    }
    
    __syncthreads();
    
    // Scan the wavefront results (only first wavefront participates)
    if (wavefront_id == 0 && threadIdx.x < (blockDim.x / WAVEFRONT_SIZE)) {
        float wavefront_sum = (threadIdx.x < blockDim.x / WAVEFRONT_SIZE) ? wavefront_results[threadIdx.x] : 0.0f;
        wavefront_sum = wavefront_scan_exclusive(wavefront_sum);
        if (threadIdx.x < 16) wavefront_results[threadIdx.x] = wavefront_sum;
    }
    
    __syncthreads();
    
    // Add wavefront offset to get final result
    float result = wavefront_result;
    if (wavefront_id > 0) {
        result += wavefront_results[wavefront_id];
    }
    
    // Convert to exclusive scan
    float exclusive_result = __shfl_up(result, 1);
    if (lane == 0) exclusive_result = 0.0f;
    
    if (idx < n) {
        output[idx] = exclusive_result;
    }
}

// ROCm-optimized scan with memory coalescing
__global__ void rocm_optimized_scan(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int wavefront_id = tid / WAVEFRONT_SIZE;
    int lane = tid % WAVEFRONT_SIZE;
    
    // Load data with optimal memory access pattern for AMD
    float val = 0.0f;
    if (idx < n) {
        val = input[idx];
    }
    
    // Wavefront-level scan
    float wavefront_sum = wavefront_scan_inclusive(val);
    
    // Use LDS for inter-wavefront communication
    __shared__ float lds_buffer[256];
    
    // Store wavefront results
    if (lane == WAVEFRONT_SIZE - 1) {
        lds_buffer[wavefront_id] = wavefront_sum;
    }
    
    __syncthreads();
    
    // Scan wavefront results using first wavefront
    if (wavefront_id == 0) {
        float wf_val = (tid < (blockDim.x / WAVEFRONT_SIZE)) ? lds_buffer[tid] : 0.0f;
        wf_val = wavefront_scan_exclusive(wf_val);
        if (tid < (blockDim.x / WAVEFRONT_SIZE)) lds_buffer[tid] = wf_val;
    }
    
    __syncthreads();
    
    // Combine results
    float final_result = wavefront_sum;
    if (wavefront_id > 0) {
        final_result += lds_buffer[wavefront_id];
    }
    
    // Convert to exclusive
    float exclusive = __shfl_up(final_result, 1);
    if (lane == 0) exclusive = 0.0f;
    
    if (idx < n) {
        output[idx] = exclusive;
    }
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
        hipEventDestroy(start_event);
        hipEventDestroy(stop_event);
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

// CPU reference implementations
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

// Test framework for correctness
void test_scan_correctness() {
    std::cout << "\n=== Scan Correctness Tests ===\n";
    
    // Test with small known arrays
    std::vector<float> test_input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected_exclusive = {0, 1, 3, 6, 10, 15, 21, 28};
    std::vector<float> expected_inclusive = {1, 3, 6, 10, 15, 21, 28, 36};
    
    int n = test_input.size();
    
    // Allocate GPU memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_input, test_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    // Test exclusive scan
    dim3 grid(1);
    dim3 block(8);
    
    hipLaunchKernelGGL(blelloch_scan_exclusive, grid, block, 0, 0, d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    std::vector<float> gpu_result(n);
    HIP_CHECK(hipMemcpy(gpu_result.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
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
    hipLaunchKernelGGL(hillis_steele_scan_inclusive, grid, block, 0, 0, d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(gpu_result.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
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
    
    hipFree(d_input);
    hipFree(d_output);
}

void run_scan_benchmarks() {
    std::cout << "\n=== HIP Scan Algorithms Benchmark ===\n";
    
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
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_input, h_input.data(), size * sizeof(float), hipMemcpyHostToDevice));
        
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
                hipLaunchKernelGGL(hillis_steele_scan_exclusive, grid, block, 0, 0, d_input, d_output, size);
            }});
            
            tests.push_back({"Blelloch", [&]() {
                dim3 grid(1);
                dim3 block(size);
                hipLaunchKernelGGL(blelloch_scan_exclusive, grid, block, 0, 0, d_input, d_output, size);
            }});
        }
        
        tests.push_back({"Wavefront-level", [&]() {
            int blocks = (size + 256 - 1) / 256;
            dim3 grid(blocks);
            dim3 block(256);
            hipLaunchKernelGGL(wavefront_scan_kernel, grid, block, 0, 0, d_input, d_output, size);
        }});
        
        tests.push_back({"ROCm Optimized", [&]() {
            int blocks = (size + 256 - 1) / 256;
            dim3 grid(blocks);
            dim3 block(256);
            hipLaunchKernelGGL(rocm_optimized_scan, grid, block, 0, 0, d_input, d_output, size);
        }});
        
        for (auto& test : tests) {
            PerformanceTimer timer;
            
            timer.start();
            test.kernel_launch();
            HIP_CHECK(hipGetLastError());
            float gpu_time = timer.stop();
            
            // Verify correctness for smaller arrays
            if (size <= 16384) {
                std::vector<float> gpu_result(size);
                HIP_CHECK(hipMemcpy(gpu_result.data(), d_output, size * sizeof(float), hipMemcpyDeviceToHost));
                
                float max_error = 0.0f;
                for (int i = 0; i < size; ++i) {
                    max_error = std::max(max_error, std::abs(gpu_result[i] - cpu_result[i]));
                }
                
                float speedup = cpu_time / gpu_time;
                double bandwidth = (size * sizeof(float) * 2) / (gpu_time * 1e6);  // Read + Write
                
                std::cout << std::setw(20) << test.name << ": " 
                          << std::fixed << std::setprecision(3) << gpu_time << " ms"
                          << " (Speedup: " << std::setprecision(1) << speedup << "x"
                          << ", Error: " << std::scientific << std::setprecision(2) << max_error
                          << ", BW: " << std::fixed << std::setprecision(1) << bandwidth << " GB/s)\n";
            } else {
                float speedup = cpu_time / gpu_time;
                double bandwidth = (size * sizeof(float) * 2) / (gpu_time * 1e6);
                
                std::cout << std::setw(20) << test.name << ": " 
                          << std::fixed << std::setprecision(3) << gpu_time << " ms"
                          << " (Speedup: " << std::setprecision(1) << speedup << "x"
                          << ", BW: " << std::setprecision(1) << bandwidth << " GB/s)\n";
            }
        }
        
        hipFree(d_input);
        hipFree(d_output);
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
    
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_marks, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_scan, n * sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    // Step 1: Mark valid elements
    dim3 grid((n + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(mark_valid_elements, grid, block, 0, 0, d_input, d_marks, threshold, n);
    
    // Step 2: Exclusive scan of marks
    hipLaunchKernelGGL(wavefront_scan_kernel, grid, block, 0, 0, (float*)d_marks, (float*)d_scan, n);
    
    // Step 3: Compact array
    hipLaunchKernelGGL(compact_array, grid, block, 0, 0, d_input, d_scan, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Verify results
    std::vector<int> h_scan(n);
    HIP_CHECK(hipMemcpy(h_scan.data(), d_scan, n * sizeof(int), hipMemcpyDeviceToHost));
    
    int gpu_count = h_scan[n-1];  // Last element gives total count
    
    std::cout << "Input elements: " << n << "\n";
    std::cout << "Threshold: " << threshold << "\n";
    std::cout << "Valid elements - Expected: " << expected_count << ", GPU: " << gpu_count << "\n";
    std::cout << "Compaction " << (expected_count == gpu_count ? "PASSED" : "FAILED") << "\n";
    
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_marks);
    hipFree(d_scan);
}

int main() {
    std::cout << "HIP Prefix Sum (Scan) Algorithms - Comprehensive Implementation\n";
    std::cout << "==============================================================\n";
    
    // Check HIP device properties
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Wavefront Size: " << WAVEFRONT_SIZE << "\n";
    std::cout << "LDS Size per Workgroup: " << props.sharedMemPerBlock << " bytes\n\n";
    
    try {
        test_scan_correctness();
        run_scan_benchmarks();
        demonstrate_stream_compaction();
        
        std::cout << "\n=== Scan Algorithm Summary (AMD-Optimized) ===\n";
        std::cout << "1. Wavefront-level: Optimized for 64-thread AMD wavefronts\n";
        std::cout << "2. LDS optimization: Efficient use of Local Data Share memory\n";
        std::cout << "3. ROCm patterns: Memory coalescing optimized for AMD architecture\n";
        std::cout << "4. Applications: Stream compaction, sorting, parallel algorithms\n";
        std::cout << "5. Performance: Focus on memory bandwidth and wavefront utilization\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}