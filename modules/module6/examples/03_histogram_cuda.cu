/**
 * Module 6: Fundamental Parallel Algorithms
 * Example 3: Histogram Operations (CUDA Implementation)
 * 
 * This example demonstrates various histogram computation implementations:
 * 1. Naive atomic histogram
 * 2. Privatized histogram with per-block bins
 * 3. Coarsened histogram with thread coarsening
 * 4. Warp-level aggregated histogram
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cassert>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
const int NUM_BINS = 256;
const int BLOCK_SIZE = 256;
const int COARSENING_FACTOR = 4;

// Performance measurement utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

// =============================================================================
// Histogram Kernels
// =============================================================================

/**
 * Naive atomic histogram
 */
__global__ void histogram_naive_atomic(unsigned char *input, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = input[idx];
        atomicAdd(&histogram[bin], 1);
    }
}

/**
 * Privatized histogram with per-block private bins
 */
__global__ void histogram_privatized(unsigned char *input, int *histogram, int n) {
    extern __shared__ int private_hist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize private histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();
    
    // Build private histogram
    if (idx < n) {
        int bin = input[idx];
        atomicAdd(&private_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge private histogram to global histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        if (private_hist[bin] > 0) {
            atomicAdd(&histogram[bin], private_hist[bin]);
        }
    }
}

/**
 * Coarsened histogram with thread coarsening
 */
__global__ void histogram_coarsened(unsigned char *input, int *histogram, int n) {
    extern __shared__ int private_hist[];
    
    int tid = threadIdx.x;
    int base_idx = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;
    
    // Initialize private histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();
    
    // Build private histogram with coarsening
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        int idx = base_idx + i * blockDim.x;
        if (idx < n) {
            int bin = input[idx];
            atomicAdd(&private_hist[bin], 1);
        }
    }
    __syncthreads();
    
    // Merge private histogram to global histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        if (private_hist[bin] > 0) {
            atomicAdd(&histogram[bin], private_hist[bin]);
        }
    }
}

/**
 * Warp-aggregated histogram with intra-warp reduction
 */
__global__ void histogram_warp_aggregated(unsigned char *input, int *histogram, int n) {
    extern __shared__ int private_hist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    // int warp_id = threadIdx.x / 32; // Unused, commented out
    
    // Initialize private histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();
    
    // Process input with warp aggregation
    if (idx < n) {
        int bin = input[idx];
        
        // Count occurrences of this bin within the warp
        int warp_count = 0;
        for (int offset = 0; offset < 32; offset++) {
            int other_bin = __shfl_sync(0xffffffff, bin, offset);
            if (other_bin == bin) {
                warp_count++;
            }
        }
        
        // Only first thread with this bin value updates the histogram
        bool first_thread = true;
        for (int offset = 0; offset < lane_id; offset++) {
            int other_bin = __shfl_sync(0xffffffff, bin, offset);
            if (other_bin == bin) {
                first_thread = false;
                break;
            }
        }
        
        if (first_thread) {
            atomicAdd(&private_hist[bin], warp_count);
        }
    }
    __syncthreads();
    
    // Merge private histogram to global histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        if (private_hist[bin] > 0) {
            atomicAdd(&histogram[bin], private_hist[bin]);
        }
    }
}

/**
 * Optimized warp-aggregated histogram using ballot and popc
 */
__global__ void histogram_warp_optimized(unsigned char *input, int *histogram, int n) {
    extern __shared__ int private_hist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    
    // Initialize private histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();
    
    // Process input with optimized warp aggregation
    if (idx < n) {
        int bin = input[idx];
        
        // Use ballot to find threads with same bin value
        for (int target_bin = 0; target_bin < NUM_BINS; target_bin++) {
            unsigned int ballot = __ballot_sync(0xffffffff, bin == target_bin);
            int count = __popc(ballot);
            
            if (count > 0 && lane_id == __ffs(ballot) - 1) {
                atomicAdd(&private_hist[target_bin], count);
            }
        }
    }
    __syncthreads();
    
    // Merge private histogram to global histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        if (private_hist[bin] > 0) {
            atomicAdd(&histogram[bin], private_hist[bin]);
        }
    }
}

/**
 * Multi-pass histogram for very large datasets
 */
__global__ void histogram_multi_pass(unsigned char *input, int *histogram, int n, int pass_size, int pass_offset) {
    extern __shared__ int private_hist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x + pass_offset;
    
    // Initialize private histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        private_hist[bin] = 0;
    }
    __syncthreads();
    
    // Build private histogram for this pass
    if (idx < n && idx < pass_offset + pass_size) {
        int bin = input[idx];
        atomicAdd(&private_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge private histogram to global histogram
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        if (private_hist[bin] > 0) {
            atomicAdd(&histogram[bin], private_hist[bin]);
        }
    }
}

// =============================================================================
// Host Functions
// =============================================================================

/**
 * Initialize test data with different distributions
 */
void initialize_uniform_data(unsigned char *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() % NUM_BINS;
    }
}

void initialize_gaussian_data(unsigned char *data, int n, float mean = 128.0f, float stddev = 32.0f) {
    for (int i = 0; i < n; i++) {
        // Simple Box-Muller transform for Gaussian distribution
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
            has_spare = false;
            float val = mean + stddev * spare;
            data[i] = (unsigned char)fmax(0, fmin(255, val));
        } else {
            has_spare = true;
            static float u, v, mag;
            do {
                u = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
                v = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
                mag = u * u + v * v;
            } while (mag >= 1.0f || mag == 0.0f);
            
            mag = stddev * sqrt(-2.0f * log(mag) / mag);
            spare = v * mag;
            float val = mean + u * mag;
            data[i] = (unsigned char)fmax(0, fmin(255, val));
        }
    }
}

void initialize_skewed_data(unsigned char *data, int n) {
    // Create a power-law distribution (more low values)
    for (int i = 0; i < n; i++) {
        float u = (float)rand() / RAND_MAX;
        // Power law with exponent -2 (heavily skewed towards low values)
        float val = 255.0f * (1.0f - pow(u, 0.3f));
        data[i] = (unsigned char)fmax(0, fmin(255, val));
    }
}

// Wrapper functions for benchmark compatibility
void initialize_gaussian_data_default(unsigned char *data, int n) {
    initialize_gaussian_data(data, n); // Use default parameters
}

/**
 * Verify histogram results
 */
bool verify_histogram(int *gpu_hist, int *cpu_hist, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (gpu_hist[i] != cpu_hist[i]) {
            printf("Verification failed at bin %d: GPU=%d, CPU=%d\n",
                   i, gpu_hist[i], cpu_hist[i]);
            return false;
        }
    }
    return true;
}

/**
 * CPU reference histogram
 */
void histogram_cpu_reference(unsigned char *input, int *histogram, int n) {
    // Initialize histogram
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }
    
    // Count occurrences
    for (int i = 0; i < n; i++) {
        histogram[input[i]]++;
    }
}

/**
 * Print histogram statistics
 */
void print_histogram_stats(int *histogram, int n, const char* distribution_name) {
    printf("%s Distribution Statistics:\n", distribution_name);
    
    // Find min, max, and total
    int min_count = histogram[0], max_count = histogram[0];
    int total = 0, non_zero_bins = 0;
    
    for (int i = 0; i < NUM_BINS; i++) {
        if (histogram[i] > 0) {
            non_zero_bins++;
            min_count = min(min_count, histogram[i]);
            max_count = max(max_count, histogram[i]);
        }
        total += histogram[i];
    }
    
    printf("  Total elements: %d\n", total);
    printf("  Non-zero bins: %d/%d\n", non_zero_bins, NUM_BINS);
    printf("  Min count: %d, Max count: %d\n", min_count, max_count);
    printf("  Load balance ratio: %.2f\n", (float)max_count / min_count);
    printf("\n");
}

/**
 * Benchmark histogram algorithms with different distributions
 */
void benchmark_histogram(const char* distribution_name, 
                        void (*init_func)(unsigned char*, int)) {
    printf("=== %s Distribution Histogram Benchmark ===\n", distribution_name);
    
    const int n = 16 * 1024 * 1024; // 16M elements
    const int num_iterations = 100;
    
    // Allocate host memory
    unsigned char *h_input = new unsigned char[n];
    int *h_hist_naive = new int[NUM_BINS];
    int *h_hist_privatized = new int[NUM_BINS];
    int *h_hist_coarsened = new int[NUM_BINS];
    int *h_hist_warp = new int[NUM_BINS];
    int *h_hist_cpu = new int[NUM_BINS];
    
    // Initialize data
    init_func(h_input, n);
    
    // Allocate device memory
    unsigned char *d_input;
    int *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive atomic histogram
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        histogram_naive_atomic<<<grid_size, block_size>>>(d_input, d_histogram, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_hist_naive, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Benchmark privatized histogram
    int shared_mem_size = NUM_BINS * sizeof(int);
    
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        histogram_privatized<<<grid_size, block_size, shared_mem_size>>>(d_input, d_histogram, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double privatized_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_hist_privatized, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Benchmark coarsened histogram
    dim3 coarse_grid((n + block_size.x * COARSENING_FACTOR - 1) / (block_size.x * COARSENING_FACTOR));
    
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        histogram_coarsened<<<coarse_grid, block_size, shared_mem_size>>>(d_input, d_histogram, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double coarsened_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_hist_coarsened, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Benchmark warp-aggregated histogram
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        histogram_warp_aggregated<<<grid_size, block_size, shared_mem_size>>>(d_input, d_histogram, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double warp_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_hist_warp, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    
    // CPU reference
    timer.start();
    histogram_cpu_reference(h_input, h_hist_cpu, n);
    double cpu_time = timer.stop();
    
    // Verify results
    bool naive_correct = verify_histogram(h_hist_naive, h_hist_cpu, NUM_BINS);
    bool privatized_correct = verify_histogram(h_hist_privatized, h_hist_cpu, NUM_BINS);
    bool coarsened_correct = verify_histogram(h_hist_coarsened, h_hist_cpu, NUM_BINS);
    bool warp_correct = verify_histogram(h_hist_warp, h_hist_cpu, NUM_BINS);
    
    // Print results
    printf("Data size: %d elements, Bins: %d\n", n, NUM_BINS);
    printf("Naive Atomic:   %.3f ms (%s)\n", naive_time, naive_correct ? "PASS" : "FAIL");
    printf("Privatized:     %.3f ms (%s), Speedup: %.2fx\n", privatized_time, 
           privatized_correct ? "PASS" : "FAIL", naive_time / privatized_time);
    printf("Coarsened:      %.3f ms (%s), Speedup: %.2fx\n", coarsened_time, 
           coarsened_correct ? "PASS" : "FAIL", naive_time / coarsened_time);
    printf("Warp Aggregated: %.3f ms (%s), Speedup: %.2fx\n", warp_time, 
           warp_correct ? "PASS" : "FAIL", naive_time / warp_time);
    printf("CPU Reference:  %.3f ms, GPU Speedup: %.2fx (best)\n", 
           cpu_time, cpu_time / fmin(fmin(privatized_time, coarsened_time), warp_time));
    
    // Print distribution statistics
    print_histogram_stats(h_hist_cpu, n, distribution_name);
    
    // Cleanup
    delete[] h_input;
    delete[] h_hist_naive;
    delete[] h_hist_privatized;
    delete[] h_hist_coarsened;
    delete[] h_hist_warp;
    delete[] h_hist_cpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));
}

/**
 * Main function
 */
int main() {
    printf("Module 6: Histogram Operations (CUDA Implementation)\n");
    printf("====================================================\n\n");
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device information
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Atomic Operations: Supported\n");
    printf("\n");
    
    // Seed random number generator
    srand(12345);
    
    // Run benchmarks with different data distributions
    benchmark_histogram("Uniform", initialize_uniform_data);
    benchmark_histogram("Gaussian", initialize_gaussian_data_default);
    benchmark_histogram("Skewed", initialize_skewed_data);
    
    printf("Histogram operation benchmarks completed successfully!\n");
    printf("\nKey Insights:\n");
    printf("- Privatization reduces global atomic contention\n");
    printf("- Thread coarsening improves memory bandwidth utilization\n");
    printf("- Warp aggregation minimizes atomic operations\n");
    printf("- Performance varies significantly with data distribution\n");
    
    return 0;
}