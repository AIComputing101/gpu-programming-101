#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

// Bitonic sorting network - always sorts 2^k elements
__global__ void bitonicSort(float *data, int n, int k, int j) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j; // XOR operation
    
    if (ixj > tid && tid < n && ixj < n) {
        // Determine sort direction based on position in bitonic sequence
        bool ascending = ((tid & k) == 0);
        
        if ((data[tid] > data[ixj]) == ascending) {
            // Swap elements
            float temp = data[tid];
            data[tid] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// Parallel radix sort - single pass
__global__ void radixSortPass(unsigned int *input, unsigned int *output, 
                             unsigned int *prefix_sums, int n, int bit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned int value = input[tid];
        int bin = (value >> bit) & 1; // Extract bit
        
        // Use prefix sum to determine output position
        int pos = tid - prefix_sums[tid] + prefix_sums[n-1] * bin;
        output[pos] = value;
    }
}

// Merge sort - merge two sorted arrays
__global__ void merge(float *input, float *output, int *indices,
                     int left, int mid, int right) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < right - left + 1) {
        int i = left;
        int j = mid + 1;
        int k = left + tid;
        
        // Simple merge logic (can be optimized)
        while (i <= mid && j <= right && k <= right) {
            if (input[i] <= input[j]) {
                output[k] = input[i];
                indices[k] = i;
                i++;
            } else {
                output[k] = input[j];
                indices[k] = j;
                j++;
            }
            k++;
        }
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

void launchBitonicSort(float *d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Find next power of 2
    int n_pow2 = 1;
    while (n_pow2 < n) n_pow2 <<= 1;
    
    // Bitonic sort phases
    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSort<<<blocks, threads>>>(d_data, n, k, j);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

int main() {
    printf("CUDA Sorting Algorithms Demonstration\n");
    printf("====================================\n");
    
    const int N = 1024; // Power of 2 for bitonic sort
    const int bytes = N * sizeof(float);
    
    // Host data
    float *h_data = (float*)malloc(bytes);
    float *h_sorted = (float*)malloc(bytes);
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;
    }
    
    printf("Array size: %d elements\n", N);
    printf("First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_data[i]);
    }
    printf("\n");
    
    // Device data
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Time the sorting
    auto start = std::chrono::high_resolution_clock::now();
    launchBitonicSort(d_data, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_sorted, d_data, bytes, cudaMemcpyDeviceToHost));
    
    // Verification
    bool is_sorted = true;
    for (int i = 1; i < N; i++) {
        if (h_sorted[i] < h_sorted[i-1]) {
            is_sorted = false;
            break;
        }
    }
    
    printf("\nBitonic sort time: %.3f ms\n", time);
    printf("Result: %s\n", is_sorted ? "SORTED" : "NOT SORTED");
    printf("First 10 sorted: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_sorted[i]);
    }
    printf("\n");
    
    // Compare with CPU sort
    std::sort(h_data, h_data + N);
    start = std::chrono::high_resolution_clock::now();
    std::sort(h_data, h_data + N);
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("CPU std::sort time: %.3f ms\n", cpu_time);
    printf("GPU speedup: %.2fx\n", cpu_time / time);
    
    free(h_data);
    free(h_sorted);
    cudaFree(d_data);
    
    printf("\nSorting algorithms demonstration completed!\n");
    return 0;
}