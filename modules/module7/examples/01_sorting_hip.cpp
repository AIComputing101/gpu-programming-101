/**
 * Module 7: Advanced Algorithmic Patterns - Sorting Algorithms (HIP)
 * 
 * Comprehensive implementation of advanced parallel sorting algorithms optimized for AMD GPU architectures.
 * This example demonstrates sorting strategies adapted for ROCm/HIP with wavefront-aware optimizations
 * and LDS utilization patterns specific to AMD hardware.
 * 
 * Topics Covered:
 * - Wavefront-aware sorting for 64-thread AMD wavefronts
 * - LDS-optimized bitonic sort networks
 * - Radix sort with AMD memory hierarchy optimization
 * - ROCm Thrust integration and performance comparison
 * - Cross-platform sorting performance analysis
 */

#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities
#include <hip/hip_cooperative_groups.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <functional>

// ROCm Thrust equivalent
#ifdef __HIP_PLATFORM_HCC__
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

namespace cg = cooperative_groups;

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int WAVEFRONT_SIZE = 64;

// Bitonic sorting network adapted for AMD architecture
__device__ void bitonic_compare_and_swap(int* data, int i, int j, bool ascending) {
    if ((data[i] > data[j]) == ascending) {
        int temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

__global__ void bitonic_sort_kernel(int* data, int n, int k, int j, bool ascending) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ij = idx ^ j;  // XOR operation for bitonic comparison
    
    if (ij > idx && idx < n && ij < n) {
        bool direction = ((idx & k) == 0) ? ascending : !ascending;
        if ((data[idx] > data[ij]) == direction) {
            int temp = data[idx];
            data[idx] = data[ij];
            data[ij] = temp;
        }
    }
}

// LDS-optimized bitonic sort for AMD GPUs
__global__ void bitonic_sort_lds_amd(int* data, int n) {
    __shared__ int lds_data[256 + 8];  // Extra space to avoid LDS bank conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into LDS with conflict avoidance
    int lds_idx = tid + tid/32;  // Add offset to avoid bank conflicts
    if (idx < n) {
        lds_data[lds_idx] = data[idx];
    } else {
        lds_data[lds_idx] = INT_MAX;  // Padding for power-of-2 requirement
    }
    
    __syncthreads();
    
    // Bitonic sort in LDS optimized for AMD wavefronts
    int wavefront_id = tid / WAVEFRONT_SIZE;
    int lane = tid % WAVEFRONT_SIZE;
    
    // First sort within wavefronts
    for (int k = 2; k <= WAVEFRONT_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner = tid ^ j;
            int partner_lds_idx = partner + partner/32;
            
            if (partner > tid && (partner / WAVEFRONT_SIZE) == wavefront_id) {
                bool ascending = ((tid & k) == 0);
                if ((lds_data[lds_idx] > lds_data[partner_lds_idx]) == ascending) {
                    int temp = lds_data[lds_idx];
                    lds_data[lds_idx] = lds_data[partner_lds_idx];
                    lds_data[partner_lds_idx] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Then merge between wavefronts
    for (int k = WAVEFRONT_SIZE * 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner = tid ^ j;
            int partner_lds_idx = partner + partner/32;
            
            if (partner > tid) {
                bool ascending = ((tid & k) == 0);
                if ((lds_data[lds_idx] > lds_data[partner_lds_idx]) == ascending) {
                    int temp = lds_data[lds_idx];
                    lds_data[lds_idx] = lds_data[partner_lds_idx];
                    lds_data[partner_lds_idx] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (idx < n) {
        data[idx] = lds_data[lds_idx];
    }
}

// Radix sort optimized for AMD memory hierarchy
__global__ void radix_count_amd(int* input, int* histogram, int n, int bit_shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better memory throughput
    for (int i = idx * 4; i < n; i += stride * 4) {
        // Process 4 elements at once
        if (i < n) {
            int digit = (input[i] >> bit_shift) & 0xF;
            atomicAdd(&histogram[digit], 1);
        }
        if (i + 1 < n) {
            int digit = (input[i + 1] >> bit_shift) & 0xF;
            atomicAdd(&histogram[digit], 1);
        }
        if (i + 2 < n) {
            int digit = (input[i + 2] >> bit_shift) & 0xF;
            atomicAdd(&histogram[digit], 1);
        }
        if (i + 3 < n) {
            int digit = (input[i + 3] >> bit_shift) & 0xF;
            atomicAdd(&histogram[digit], 1);
        }
    }
}

__global__ void radix_scan_amd(int* histogram, int* prefix_sum, int num_bins) {
    // Wavefront-aware scan implementation
    int tid = threadIdx.x;
    
    if (tid == 0) {
        prefix_sum[0] = 0;
        for (int i = 1; i < num_bins; ++i) {
            prefix_sum[i] = prefix_sum[i-1] + histogram[i-1];
        }
    }
}

__global__ void radix_scatter_amd(int* input, int* output, int* prefix_sum, 
                                  int n, int bit_shift) {
    __shared__ int lds_prefix[16 + 1];  // +1 to avoid LDS bank conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize LDS prefix sums with conflict avoidance
    if (tid < 16) {
        lds_prefix[tid] = prefix_sum[tid];
    }
    
    __syncthreads();
    
    // Scatter with coalesced memory access
    if (idx * 4 < n) {
        // Process 4 elements for better memory throughput
        for (int i = 0; i < 4 && idx * 4 + i < n; ++i) {
            int value = input[idx * 4 + i];
            int digit = (value >> bit_shift) & 0xF;
            int pos = atomicAdd(&lds_prefix[digit], 1);
            output[pos] = value;
        }
    }
}

// Wavefront-level sorting using shuffle operations
__device__ void wavefront_bitonic_sort(int* data, int n) {
    int tid = threadIdx.x;
    int lane = tid % WAVEFRONT_SIZE;
    int wavefront_id = tid / WAVEFRONT_SIZE;
    
    if (tid >= n) return;
    
    int value = data[tid];
    
    // Bitonic sort within wavefront using shuffle operations
    for (int k = 2; k <= WAVEFRONT_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner_lane = lane ^ j;
            int partner_value = __shfl(value, partner_lane);
            
            bool ascending = ((lane & k) == 0);
            if (partner_lane > lane) {
                if ((value > partner_value) == ascending) {
                    value = partner_value;
                }
            }
        }
    }
    
    data[tid] = value;
}

__global__ void wavefront_sort_kernel(int* data, int n) {
    wavefront_bitonic_sort(data, n);
    __syncthreads();
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

// Sorting algorithm implementations
void bitonic_sort_amd(int* d_data, int n) {
    // Ensure n is power of 2 for bitonic sort
    int padded_n = 1;
    while (padded_n < n) padded_n *= 2;
    
    if (padded_n > n) {
        // Pad with MAX_INT
        HIP_CHECK(hipMemset(d_data + n, 0xFF, (padded_n - n) * sizeof(int)));
    }
    
    if (padded_n <= 256) {
        // Use LDS-optimized version for small arrays
        dim3 block(padded_n);
        dim3 grid(1);
        hipLaunchKernelGGL(bitonic_sort_lds_amd, grid, block, 0, 0, d_data, n);
    } else {
        // Use global memory version for larger arrays
        dim3 block(256);
        dim3 grid((padded_n + block.x - 1) / block.x);
        
        for (int k = 2; k <= padded_n; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                hipLaunchKernelGGL(bitonic_sort_kernel, grid, block, 0, 0, d_data, padded_n, k, j, true);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());
            }
        }
    }
}

void radix_sort_amd(int* d_data, int n) {
    const int radix = 4;  // 4-bit radix
    const int num_bins = 1 << radix;
    const int num_passes = 32 / radix;  // For 32-bit integers
    
    int *d_temp, *d_histogram, *d_prefix_sum;
    HIP_CHECK(hipMalloc(&d_temp, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_histogram, num_bins * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_prefix_sum, num_bins * sizeof(int)));
    
    int* current_input = d_data;
    int* current_output = d_temp;
    
    for (int pass = 0; pass < num_passes; ++pass) {
        int bit_shift = pass * radix;
        
        // Clear histogram
        HIP_CHECK(hipMemset(d_histogram, 0, num_bins * sizeof(int)));
        
        // Count digits with AMD optimization
        dim3 grid((n + 1023) / 1024);  // Process 4 elements per thread
        dim3 block(256);
        hipLaunchKernelGGL(radix_count_amd, grid, block, 0, 0, current_input, d_histogram, n, bit_shift);
        
        // Compute prefix sums
        hipLaunchKernelGGL(radix_scan_amd, dim3(1), dim3(1), 0, 0, d_histogram, d_prefix_sum, num_bins);
        
        // Scatter elements
        hipLaunchKernelGGL(radix_scatter_amd, grid, block, 0, 0, 
                          current_input, current_output, d_prefix_sum, n, bit_shift);
        
        // Swap input and output
        std::swap(current_input, current_output);
    }
    
    // Copy result back if needed
    if (current_input != d_data) {
        HIP_CHECK(hipMemcpy(d_data, current_input, n * sizeof(int), hipMemcpyDeviceToDevice));
    }
    
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_histogram));
    HIP_CHECK(hipFree(d_prefix_sum));
}

// Test framework
struct SortingTest {
    std::string name;
    std::function<void(int*, int)> sort_function;
    
    SortingTest(const std::string& n, std::function<void(int*, int)> func) 
        : name(n), sort_function(func) {}
};

void run_sorting_benchmarks() {
    std::cout << "\n=== HIP Sorting Algorithms Benchmark ===\n";
    
    std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    
    for (int size : sizes) {
        std::cout << "\nTesting with " << size << " elements:\n";
        std::cout << std::string(50, '-') << "\n";
        
        // Generate random test data
        std::vector<int> h_data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1000000);
        
        for (int i = 0; i < size; ++i) {
            h_data[i] = dis(gen);
        }
        
        // CPU reference timing
        auto h_data_copy = h_data;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        std::sort(h_data_copy.begin(), h_data_copy.end());
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        std::cout << "CPU std::sort:       " << std::fixed << std::setprecision(3) 
                  << cpu_time << " ms\n";
        
        // GPU memory allocation
        int* d_data;
        size_t padded_size = size;
        
        // For bitonic sort, ensure power of 2
        int temp_size = 1;
        while (temp_size < size) temp_size *= 2;
        padded_size = temp_size;
        
        HIP_CHECK(hipMalloc(&d_data, padded_size * sizeof(int)));
        
        // Test different sorting implementations
        std::vector<SortingTest> tests = {
            SortingTest("Bitonic Sort AMD", [](int* d_input, int n) {
                bitonic_sort_amd(d_input, n);
            }),
            SortingTest("Radix Sort AMD", [](int* d_input, int n) {
                radix_sort_amd(d_input, n);
            })
        };
        
        // Add Thrust sort if available
        #ifdef __HIP_PLATFORM_HCC__
        tests.push_back(SortingTest("HIP Thrust Sort", [](int* d_input, int n) {
            thrust::sort(thrust::device, d_input, d_input + n);
        }));
        #endif
        
        for (auto& test : tests) {
            // Copy data to GPU
            HIP_CHECK(hipMemcpy(d_data, h_data.data(), size * sizeof(int), hipMemcpyHostToDevice));
            
            PerformanceTimer timer;
            
            timer.start();
            test.sort_function(d_data, size);
            HIP_CHECK(hipDeviceSynchronize());
            float gpu_time = timer.stop();
            
            // Verify correctness
            std::vector<int> gpu_result(size);
            HIP_CHECK(hipMemcpy(gpu_result.data(), d_data, size * sizeof(int), hipMemcpyDeviceToHost));
            
            bool correct = std::is_sorted(gpu_result.begin(), gpu_result.end());
            if (correct) {
                // Additional verification against CPU result
                for (int i = 0; i < size && correct; ++i) {
                    if (gpu_result[i] != h_data_copy[i]) {
                        correct = false;
                    }
                }
            }
            
            float speedup = cpu_time / gpu_time;
            double throughput = size / (gpu_time * 1e3);  // Elements per second
            
            std::cout << std::setw(20) << test.name << ": " 
                      << std::fixed << std::setprecision(3) << gpu_time << " ms"
                      << " (Speedup: " << std::setprecision(1) << speedup << "x"
                      << ", Throughput: " << std::setprecision(1) << throughput/1e6 << "M elem/s"
                      << ", " << (correct ? "PASS" : "FAIL") << ")\n";
        }
        
        HIP_CHECK(hipFree(d_data));
    }
}

// AMD-specific performance analysis
void test_wavefront_optimization() {
    std::cout << "\n=== Wavefront Optimization Analysis ===\n";
    
    const int size = WAVEFRONT_SIZE * 16;  // 16 wavefronts
    
    std::vector<int> h_data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000);
    
    for (int i = 0; i < size; ++i) {
        h_data[i] = dis(gen);
    }
    
    int* d_data;
    HIP_CHECK(hipMalloc(&d_data, size * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), size * sizeof(int), hipMemcpyHostToDevice));
    
    PerformanceTimer timer;
    timer.start();
    
    dim3 block(size);
    dim3 grid(1);
    hipLaunchKernelGGL(wavefront_sort_kernel, grid, block, 0, 0, d_data, size);
    HIP_CHECK(hipDeviceSynchronize());
    
    float gpu_time = timer.stop();
    
    std::vector<int> gpu_result(size);
    HIP_CHECK(hipMemcpy(gpu_result.data(), d_data, size * sizeof(int), hipMemcpyDeviceToHost));
    
    bool correct = std::is_sorted(gpu_result.begin(), gpu_result.end());
    
    std::cout << "Wavefront-level sort (" << size << " elements): "
              << std::fixed << std::setprecision(3) << gpu_time << " ms"
              << " (" << (correct ? "PASS" : "FAIL") << ")\n";
    
    HIP_CHECK(hipFree(d_data));
}

int main() {
    std::cout << "HIP Advanced Sorting Algorithms - AMD GPU Optimized Implementation\n";
    std::cout << "=================================================================\n";
    
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
        run_sorting_benchmarks();
        test_wavefront_optimization();
        
        std::cout << "\n=== HIP Sorting Algorithm Analysis Summary ===\n";
        std::cout << "1. LDS optimization crucial for small array sorting performance\n";
        std::cout << "2. Wavefront-aware algorithms leverage 64-thread AMD wavefronts\n";
        std::cout << "3. Memory coalescing patterns optimized for AMD memory hierarchy\n";
        std::cout << "4. Bank conflict avoidance in LDS improves performance\n";
        std::cout << "5. ROCm Thrust provides highly optimized library implementations\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}