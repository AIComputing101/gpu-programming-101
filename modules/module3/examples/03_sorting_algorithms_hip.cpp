#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include "rocm7_utils.h"

// Bitonic sorting network optimized for AMD GPUs
__global__ void bitonicSortHIP(float *data, int n, int k, int j) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j; // XOR operation for bitonic network
    
    if (ixj > tid && tid < n && ixj < n) {
        // Determine sort direction based on position in sequence
        bool ascending = ((tid & k) == 0);
        
        if ((data[tid] > data[ixj]) == ascending) {
            // Swap elements using explicit temporary variable
            float temp = data[tid];
            data[tid] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// AMD-optimized radix sort using wavefront operations
__global__ void radixSortHIP(unsigned int *input, unsigned int *output, 
                            unsigned int *histogram, int n, int bit, int pass) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64; // AMD wavefront size
    int warp_id = threadIdx.x / 64;
    
    __shared__ unsigned int local_hist[2]; // 0s and 1s count
    __shared__ unsigned int warp_offsets[16]; // Up to 16 warps per block
    
    if (threadIdx.x < 2) {
        local_hist[threadIdx.x] = 0;
    }
    
    __syncthreads();
    
    // Count bits in this block
    if (tid < n) {
        unsigned int value = input[tid];
        int bin = (value >> bit) & 1;
        atomicAdd(&local_hist[bin], 1);
    }
    
    __syncthreads();
    
    // Compute offsets for each warp
    if (threadIdx.x == 0) {
        warp_offsets[0] = 0;
        for (int i = 1; i < blockDim.x / 64; i++) {
            warp_offsets[i] = warp_offsets[i-1] + 64;
        }
    }
    
    __syncthreads();
    
    // Scatter elements to output
    if (tid < n) {
        unsigned int value = input[tid];
        int bin = (value >> bit) & 1;
        
        // Calculate position based on prefix sum and local offset
        int local_offset = __popc(__ballot(lane < (threadIdx.x % 64) && 
                                          ((input[blockIdx.x * blockDim.x + (threadIdx.x / 64) * 64 + lane] >> bit) & 1) == bin));
        
        int global_pos;
        if (bin == 0) {
            global_pos = histogram[blockIdx.x] + local_offset;
        } else {
            global_pos = histogram[blockIdx.x] + local_hist[0] + local_offset;
        }
        
        if (global_pos < n) {
            output[global_pos] = value;
        }
    }
}

// Optimized merge sort for HIP
__global__ void mergeSortHIP(float *input, float *output, float *temp, 
                            int n, int segment_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int segment_id = tid / segment_size;
    int local_id = tid % segment_size;
    
    int start = segment_id * segment_size * 2;
    int mid = start + segment_size - 1;
    int end = min(start + segment_size * 2 - 1, n - 1);
    
    if (start >= n) return;
    
    // Simple merge for demonstration
    if (local_id == 0) {
        int i = start;
        int j = mid + 1;
        int k = start;
        
        while (i <= mid && j <= end && k < n) {
            if (input[i] <= input[j]) {
                temp[k++] = input[i++];
            } else {
                temp[k++] = input[j++];
            }
        }
        
        while (i <= mid && k < n) {
            temp[k++] = input[i++];
        }
        
        while (j <= end && k < n) {
            temp[k++] = input[j++];
        }
        
        // Copy back to output
        for (int idx = start; idx < k; idx++) {
            output[idx] = temp[idx];
        }
    }
}

// Quick sort partitioning kernel
__global__ void quickSortPartition(float *data, int *pivot_pos, int left, int right, 
                                  float pivot_value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x + left;
    
    if (tid >= right) return;
    
    __shared__ int local_less[256];
    __shared__ int local_greater[256];
    
    int ltid = threadIdx.x;
    local_less[ltid] = 0;
    local_greater[ltid] = 0;
    
    if (tid < right) {
        if (data[tid] < pivot_value) {
            local_less[ltid] = 1;
        } else if (data[tid] > pivot_value) {
            local_greater[ltid] = 1;
        }
    }
    
    __syncthreads();
    
    // Prefix sum to find positions
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp_less = local_less[ltid];
        int temp_greater = local_greater[ltid];
        
        if (ltid >= stride) {
            temp_less += local_less[ltid - stride];
            temp_greater += local_greater[ltid - stride];
        }
        
        __syncthreads();
        local_less[ltid] = temp_less;
        local_greater[ltid] = temp_greater;
        __syncthreads();
    }
    
    // Store partitioning result (simplified)
    if (threadIdx.x == blockDim.x - 1) {
        *pivot_pos = left + local_less[ltid];
    }
}

// Odd-even sort for small arrays
__global__ void oddEvenSortHIP(float *data, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int idx;
    if (phase % 2 == 0) {
        // Even phase: compare (0,1), (2,3), (4,5), ...
        idx = 2 * tid;
    } else {
        // Odd phase: compare (1,2), (3,4), (5,6), ...
        idx = 2 * tid + 1;
    }
    
    if (idx + 1 < n) {
        if (data[idx] > data[idx + 1]) {
            // Swap
            float temp = data[idx];
            data[idx] = data[idx + 1];
            data[idx + 1] = temp;
        }
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

void launchBitonicSortHIP(float *d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Find next power of 2
    int n_pow2 = 1;
    while (n_pow2 < n) n_pow2 <<= 1;
    
    // Bitonic sort phases
    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortHIP<<<blocks, threads>>>(d_data, n, k, j);
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
}

void launchOddEvenSort(float *d_data, int n) {
    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    
    // Odd-even sort requires n phases
    for (int phase = 0; phase < n; phase++) {
        oddEvenSortHIP<<<blocks, threads>>>(d_data, n, phase);
        HIP_CHECK(hipDeviceSynchronize());
    }
}

bool verifySorted(float *data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}

void printArray(float *arr, int n, const char *name, int max_print = 10) {
    printf("%s: ", name);
    int print_count = (n < max_print) ? n : max_print;
    for (int i = 0; i < print_count; i++) {
        printf("%.1f ", arr[i]);
    }
    if (n > max_print) printf("...");
    printf("\n");
}

int main() {
    printf("HIP Sorting Algorithms Demonstration\n");
    printf("====================================\n\n");
    
    // Check HIP device
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("No HIP-compatible devices found!\n");
        return 1;
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Wavefront size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);
    
    const int N = 1024; // Power of 2 for bitonic sort
    const int bytes = N * sizeof(float);
    
    // Host arrays
    float *h_data = (float*)malloc(bytes);
    float *h_sorted = (float*)malloc(bytes);
    float *h_reference = (float*)malloc(bytes);
    
    // Initialize random data
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 1000);
        h_reference[i] = h_data[i]; // Copy for CPU reference
    }
    
    printf("Array size: %d elements\n", N);
    printArray(h_data, N, "Original data");
    
    // Device memory
    float *d_data, *d_temp;
    HIP_CHECK(hipMalloc(&d_data, bytes));
    HIP_CHECK(hipMalloc(&d_temp, bytes));
    
    // 1. Bitonic Sort
    printf("\n1. Bitonic Sort:\n");
    HIP_CHECK(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    launchBitonicSortHIP(d_data, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    double bitonic_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    HIP_CHECK(hipMemcpy(h_sorted, d_data, bytes, hipMemcpyDeviceToHost));
    
    bool is_sorted = verifySorted(h_sorted, N);
    printf("   Time: %.3f ms\n", bitonic_time);
    printf("   Result: %s\n", is_sorted ? "SORTED" : "NOT SORTED");
    printf("   Complexity: O(n log²n)\n");
    printf("   Works best with power-of-2 array sizes\n");
    printArray(h_sorted, N, "   Sorted data");
    
    // 2. Odd-Even Sort (for small arrays)
    if (N <= 512) { // Only for smaller arrays due to O(n²) complexity
        printf("\n2. Odd-Even Sort:\n");
        HIP_CHECK(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
        
        start = std::chrono::high_resolution_clock::now();
        launchOddEvenSort(d_data, N);
        end = std::chrono::high_resolution_clock::now();
        
        double oddeven_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        HIP_CHECK(hipMemcpy(h_sorted, d_data, bytes, hipMemcpyDeviceToHost));
        
        is_sorted = verifySorted(h_sorted, N);
        printf("   Time: %.3f ms\n", oddeven_time);
        printf("   Result: %s\n", is_sorted ? "SORTED" : "NOT SORTED");
        printf("   Complexity: O(n²) time, but good parallelism\n");
        printf("   Suitable for smaller arrays\n");
        printArray(h_sorted, N, "   Sorted data");
    }
    
    // 3. CPU std::sort for comparison
    printf("\n3. CPU std::sort (reference):\n");
    
    start = std::chrono::high_resolution_clock::now();
    std::sort(h_reference, h_reference + N);
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("   Time: %.3f ms\n", cpu_time);
    printf("   Complexity: O(n log n) average case\n");
    printArray(h_reference, N, "   CPU sorted");
    
    // 4. Performance analysis for different array sizes
    printf("\n4. Performance Analysis (Different Array Sizes):\n");
    
    int test_sizes[] = {128, 256, 512, 1024, 2048};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("   Array Size | Bitonic Sort | CPU Sort | Speedup\n");
    printf("   -----------+--------------+----------+--------\n");
    
    for (int t = 0; t < num_tests; t++) {
        int test_n = test_sizes[t];
        if (test_n > prop.maxThreadsPerBlock * 4) continue; // Skip if too large
        
        int test_bytes = test_n * sizeof(float);
        
        float *h_test = (float*)malloc(test_bytes);
        float *d_test;
        HIP_CHECK(hipMalloc(&d_test, test_bytes));
        
        // Initialize test data
        for (int i = 0; i < test_n; i++) {
            h_test[i] = (float)(rand() % 1000);
        }
        
        // GPU bitonic sort
        HIP_CHECK(hipMemcpy(d_test, h_test, test_bytes, hipMemcpyHostToDevice));
        
        start = std::chrono::high_resolution_clock::now();
        launchBitonicSortHIP(d_test, test_n);
        end = std::chrono::high_resolution_clock::now();
        
        double gpu_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // CPU sort
        start = std::chrono::high_resolution_clock::now();
        std::sort(h_test, h_test + test_n);
        end = std::chrono::high_resolution_clock::now();
        
        double cpu_time_test = std::chrono::duration<double, std::milli>(end - start).count();
        
        double speedup = cpu_time_test / gpu_time;
        
        printf("   %8d   | %9.3f ms | %7.3f ms | %6.2fx\n", 
               test_n, gpu_time, cpu_time_test, speedup);
        
        free(h_test);
        HIP_CHECK(hipFree(d_test));
    }
    
    // 5. Memory bandwidth analysis
    printf("\n5. Memory Bandwidth Analysis:\n");
    
    // For sorting, we typically read and write each element multiple times
    double data_moved = (double)(N * sizeof(float)) * log2(N) * 2; // Rough estimate
    double bandwidth = data_moved / (bitonic_time * 1e6); // GB/s
    
    printf("   Data moved (estimated): %.2f MB\n", data_moved / (1024 * 1024));
    printf("   Effective bandwidth: %.2f GB/s\n", bandwidth);
    printf("   Theoretical peak bandwidth: ~1000+ GB/s (depends on GPU)\n");
    printf("   Efficiency: %.1f%%\n", (bandwidth / 1000.0) * 100);
    
    // Summary
    printf("\nAlgorithm Characteristics:\n");
    printf("  Bitonic Sort:\n");
    printf("    + Excellent parallelism\n");
    printf("    + Predictable memory access patterns\n");
    printf("    + Works well on GPUs\n");
    printf("    - Requires power-of-2 array sizes (or padding)\n");
    printf("    - O(n log²n) complexity (not optimal)\n\n");
    
    printf("  Odd-Even Sort:\n");
    printf("    + Simple to implement\n");
    printf("    + Good for small arrays\n");
    printf("    + Natural parallelism\n");
    printf("    - O(n²) time complexity\n");
    printf("    - Not suitable for large arrays\n\n");
    
    printf("Performance Summary:\n");
    printf("  Array size: %d elements\n", N);
    printf("  Bitonic sort: %.3f ms\n", bitonic_time);
    printf("  CPU std::sort: %.3f ms\n", cpu_time);
    printf("  GPU speedup: %.2fx\n", cpu_time / bitonic_time);
    
    // Cleanup
    free(h_data); free(h_sorted); free(h_reference);
    HIP_CHECK(hipFree(d_data)); HIP_CHECK(hipFree(d_temp));
    
    printf("\nHIP sorting algorithms demonstration completed!\n");
    return 0;
}