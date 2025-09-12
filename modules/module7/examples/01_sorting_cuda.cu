/**
 * Module 7: Advanced Algorithmic Patterns - Sorting Algorithms (CUDA)
 * 
 * Comprehensive implementation of advanced parallel sorting algorithms optimized for GPU architectures.
 * This example demonstrates bitonic sort, radix sort, merge sort, and hybrid approaches with detailed
 * performance analysis and scalability evaluation.
 * 
 * Topics Covered:
 * - Bitonic sorting networks with data-independent execution
 * - Parallel radix sort with multi-digit processing
 * - Merge sort with hierarchical divide-and-conquer
 * - Hybrid sorting with adaptive algorithm selection
 * - Performance comparison and analysis
 * 
 * Performance Targets:
 * - Sorting throughput > 1B keys/second for 32-bit integers
 * - Scalability across different problem sizes
 * - Memory bandwidth utilization > 80%
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
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Bitonic sorting network implementation
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

// Shared memory bitonic sort for small arrays
__global__ void bitonic_sort_shared(int* data, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = INT_MAX;  // Padding for power-of-2 requirement
    }
    
    __syncthreads();
    
    // Bitonic sort in shared memory
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ij = tid ^ j;
            if (ij > tid) {
                bool ascending = ((tid & k) == 0);
                if ((sdata[tid] > sdata[ij]) == ascending) {
                    int temp = sdata[tid];
                    sdata[tid] = sdata[ij];
                    sdata[ij] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Radix sort implementation
__global__ void radix_count_kernel(int* input, int* histogram, int n, int bit_shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int digit = (input[i] >> bit_shift) & 0xF;  // 4-bit radix
        atomicAdd(&histogram[digit], 1);
    }
}

__global__ void radix_scan_kernel(int* histogram, int* prefix_sum, int num_bins) {
    // Simple sequential scan for small number of bins
    if (threadIdx.x == 0) {
        prefix_sum[0] = 0;
        for (int i = 1; i < num_bins; ++i) {
            prefix_sum[i] = prefix_sum[i-1] + histogram[i-1];
        }
    }
}

__global__ void radix_scatter_kernel(int* input, int* output, int* prefix_sum, 
                                    int n, int bit_shift) {
    extern __shared__ int local_prefix[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local prefix sums
    if (tid < 16) {
        local_prefix[tid] = prefix_sum[tid];
    }
    
    __syncthreads();
    
    if (idx < n) {
        int value = input[idx];
        int digit = (value >> bit_shift) & 0xF;
        int pos = atomicAdd(&local_prefix[digit], 1);
        output[pos] = value;
    }
}

// Parallel merge sort
__device__ void merge_path(int* a, int a_len, int* b, int b_len, int diagonal,
                          int* a_idx, int* b_idx) {
    int begin = max(0, diagonal - b_len);
    int end = min(diagonal, a_len);
    
    while (begin < end) {
        int mid = (begin + end) / 2;
        if (a[mid] <= b[diagonal - 1 - mid]) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    
    *a_idx = begin;
    *b_idx = diagonal - begin;
}

__global__ void merge_kernel(int* input, int* output, int* segment_begins, 
                           int num_segments, int segment_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_segments / 2) return;
    
    int left_start = segment_begins[tid * 2];
    int left_size = segment_begins[tid * 2 + 1] - left_start;
    int right_start = segment_begins[tid * 2 + 1];
    int right_size = (tid * 2 + 2 < num_segments) ? 
                    segment_begins[tid * 2 + 2] - right_start : 
                    segment_size - right_start;
    
    // Merge two sorted segments
    int i = 0, j = 0, k = left_start;
    while (i < left_size && j < right_size) {
        if (input[left_start + i] <= input[right_start + j]) {
            output[k++] = input[left_start + i++];
        } else {
            output[k++] = input[right_start + j++];
        }
    }
    
    while (i < left_size) {
        output[k++] = input[left_start + i++];
    }
    
    while (j < right_size) {
        output[k++] = input[right_start + j++];
    }
}

// Hybrid sorting algorithm selection
__device__ int select_sorting_algorithm(int n, float entropy) {
    if (n < 1024) return 0;  // Bitonic sort for small arrays
    if (entropy < 0.5f) return 1;  // Radix sort for low entropy
    return 2;  // Merge sort for general case
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

// Bitonic sort implementation
void bitonic_sort(int* d_data, int n) {
    // Ensure n is power of 2 for bitonic sort
    int padded_n = 1;
    while (padded_n < n) padded_n *= 2;
    
    if (padded_n > n) {
        // Pad with MAX_INT
        CUDA_CHECK(cudaMemset(d_data + n, 0xFF, (padded_n - n) * sizeof(int)));
    }
    
    dim3 block(256);
    dim3 grid((padded_n + block.x - 1) / block.x);
    
    // Bitonic sorting network
    for (int k = 2; k <= padded_n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonic_sort_kernel<<<grid, block>>>(d_data, padded_n, k, j, true);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// Radix sort implementation
void radix_sort(int* d_data, int n) {
    const int radix = 4;  // 4-bit radix
    const int num_bins = 1 << radix;
    const int num_passes = 32 / radix;  // For 32-bit integers
    
    int *d_temp, *d_histogram, *d_prefix_sum;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, num_bins * sizeof(int)));
    
    int* current_input = d_data;
    int* current_output = d_temp;
    
    for (int pass = 0; pass < num_passes; ++pass) {
        int bit_shift = pass * radix;
        
        // Clear histogram
        CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
        
        // Count digits
        dim3 grid((n + 255) / 256);
        dim3 block(256);
        radix_count_kernel<<<grid, block>>>(current_input, d_histogram, n, bit_shift);
        
        // Compute prefix sums
        radix_scan_kernel<<<1, 1>>>(d_histogram, d_prefix_sum, num_bins);
        
        // Scatter elements
        size_t shared_mem = num_bins * sizeof(int);
        radix_scatter_kernel<<<grid, block, shared_mem>>>(
            current_input, current_output, d_prefix_sum, n, bit_shift);
        
        // Swap input and output
        std::swap(current_input, current_output);
    }
    
    // Copy result back if needed
    if (current_input != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, current_input, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    cudaFree(d_temp);
    cudaFree(d_histogram);
    cudaFree(d_prefix_sum);
}

// Test framework
struct SortingTest {
    std::string name;
    std::function<void(int*, int)> sort_function;
    
    SortingTest(const std::string& n, std::function<void(int*, int)> func) 
        : name(n), sort_function(func) {}
};

void run_sorting_benchmarks() {
    std::cout << "\n=== CUDA Sorting Algorithms Benchmark ===\n";
    
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
        
        CUDA_CHECK(cudaMalloc(&d_data, padded_size * sizeof(int)));
        
        // Test different sorting implementations
        std::vector<SortingTest> tests = {
            SortingTest("Bitonic Sort", [](int* d_input, int n) {
                bitonic_sort(d_input, n);
            }),
            SortingTest("Radix Sort", [](int* d_input, int n) {
                radix_sort(d_input, n);
            }),
            SortingTest("Thrust Sort", [](int* d_input, int n) {
                thrust::sort(thrust::device_ptr<int>(d_input), 
                           thrust::device_ptr<int>(d_input + n));
            })
        };
        
        for (auto& test : tests) {
            // Copy data to GPU
            CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(int), cudaMemcpyHostToDevice));
            
            PerformanceTimer timer;
            
            timer.start();
            test.sort_function(d_data, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            float gpu_time = timer.stop();
            
            // Verify correctness
            std::vector<int> gpu_result(size);
            CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
            
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
            
            std::cout << std::setw(15) << test.name << ": " 
                      << std::fixed << std::setprecision(3) << gpu_time << " ms"
                      << " (Speedup: " << std::setprecision(1) << speedup << "x"
                      << ", Throughput: " << std::setprecision(1) << throughput/1e6 << "M elem/s"
                      << ", " << (correct ? "PASS" : "FAIL") << ")\n";
        }
        
        cudaFree(d_data);
    }
}

// Data pattern analysis
void test_different_data_patterns() {
    std::cout << "\n=== Data Pattern Analysis ===\n";
    
    const int size = 1048576;
    
    std::vector<std::pair<std::string, std::function<void(std::vector<int>&)>>> patterns = {
        {"Random", [](std::vector<int>& data) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, 1000000);
            for (auto& val : data) val = dis(gen);
        }},
        {"Sorted", [](std::vector<int>& data) {
            std::iota(data.begin(), data.end(), 0);
        }},
        {"Reverse Sorted", [](std::vector<int>& data) {
            std::iota(data.rbegin(), data.rend(), 0);
        }},
        {"Nearly Sorted", [](std::vector<int>& data) {
            std::iota(data.begin(), data.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> swap_dis(0, data.size() - 1);
            // Randomly swap 1% of elements
            for (int i = 0; i < data.size() / 100; ++i) {
                int idx1 = swap_dis(gen);
                int idx2 = swap_dis(gen);
                std::swap(data[idx1], data[idx2]);
            }
        }},
        {"Duplicate Heavy", [](std::vector<int>& data) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, 100);  // Many duplicates
            for (auto& val : data) val = dis(gen);
        }}
    };
    
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
    
    for (auto& [pattern_name, pattern_func] : patterns) {
        std::cout << "\nTesting " << pattern_name << " data pattern:\n";
        
        std::vector<int> h_data(size);
        pattern_func(h_data);
        
        // Test radix sort (generally most robust)
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        
        PerformanceTimer timer;
        timer.start();
        radix_sort(d_data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        float gpu_time = timer.stop();
        
        // Verify correctness
        std::vector<int> gpu_result(size);
        CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
        
        bool correct = std::is_sorted(gpu_result.begin(), gpu_result.end());
        double throughput = size / (gpu_time * 1e3);
        
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms"
                  << ", Throughput: " << std::setprecision(1) << throughput/1e6 << "M elem/s"
                  << " (" << (correct ? "PASS" : "FAIL") << ")\n";
    }
    
    cudaFree(d_data);
}

int main() {
    std::cout << "CUDA Advanced Sorting Algorithms - Comprehensive Implementation\n";
    std::cout << "==============================================================\n";
    
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
        run_sorting_benchmarks();
        test_different_data_patterns();
        
        std::cout << "\n=== Sorting Algorithm Analysis Summary ===\n";
        std::cout << "1. Bitonic Sort: O(N log² N) work, predictable performance, best for small arrays\n";
        std::cout << "2. Radix Sort: O(d·N) work, excellent for integers, data-independent performance\n";
        std::cout << "3. Merge Sort: O(N log N) work, cache-friendly, good for general comparison sorting\n";
        std::cout << "4. Thrust Sort: Highly optimized library implementation, best overall performance\n";
        std::cout << "5. Performance varies significantly with data patterns and GPU architecture\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}