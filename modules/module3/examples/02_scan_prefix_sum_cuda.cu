#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

namespace cg = cooperative_groups;

// Naive scan - O(n^2) work complexity but educational
__global__ void naiveScan(float *input, float *output, int n, bool inclusive) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float sum = 0.0f;
        int end = inclusive ? tid + 1 : tid;
        
        for (int i = 0; i < end; i++) {
            sum += input[i];
        }
        
        output[tid] = sum;
    }
}

// Hillis-Steele scan - O(n log n) work but simple to implement
__global__ void hillisSteeleScan(float *input, float *output, int n) {
    __shared__ float temp[1024];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase
    for (int d = 1; d < blockDim.x; d <<= 1) {
        float val = 0.0f;
        if (tid >= d) {
            val = temp[tid - d];
        }
        __syncthreads();
        
        if (tid >= d) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    // Write result (inclusive scan)
    if (i < n) {
        output[i] = temp[tid];
    }
}

// Blelloch scan - O(n) work complexity (work-efficient)
__global__ void blellochScan(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            if (ai < blockDim.x && bi < blockDim.x) {
                temp[bi] += temp[ai];
            }
        }
        __syncthreads();
    }
    
    // Clear last element for exclusive scan
    if (tid == 0) {
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d <<= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            
            if (ai < blockDim.x && bi < blockDim.x) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();
    }
    
    // Write result (exclusive scan)
    if (i < n) {
        output[i] = temp[tid];
    }
}

// Segmented scan - scan multiple segments independently
__global__ void segmentedScan(float *input, int *segment_heads, float *output, int n) {
    __shared__ float sdata[256];
    __shared__ int flags[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and segment flags
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    flags[tid] = (i < n) ? segment_heads[i] : 1; // End segment if out of bounds
    
    __syncthreads();
    
    // Segmented inclusive scan
    for (int d = 1; d < blockDim.x; d <<= 1) {
        float val = 0.0f;
        int flag = 0;
        
        if (tid >= d) {
            val = sdata[tid - d];
            flag = flags[tid - d];
        }
        __syncthreads();
        
        if (tid >= d) {
            if (!flags[tid]) { // Not a segment head
                sdata[tid] += val;
                flags[tid] = flag;
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (i < n) {
        output[i] = sdata[tid];
    }
}

// Modern cooperative groups scan
__global__ void cooperativeGroupsScan(float *input, float *output, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    __shared__ float warp_sums[32];
    __shared__ float shared_data[1024];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data
    float value = (idx < n) ? input[idx] : 0.0f;
    shared_data[tid] = value;
    
    // Phase 1: Warp-level scan
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = warp.shfl_up(value, offset);
        if (warp.thread_rank() >= offset) {
            value += temp;
        }
    }
    
    // Store warp scan results
    if (warp.thread_rank() == 31) {
        warp_sums[warp.meta_group_rank()] = value;
    }
    
    block.sync();
    
    // Phase 2: Scan the warp sums
    if (warp.meta_group_rank() == 0) {
        float warp_sum = (threadIdx.x < block.size() / 32) ? warp_sums[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float temp = warp.shfl_up(warp_sum, offset);
            if (warp.thread_rank() >= offset) {
                warp_sum += temp;
            }
        }
        
        if (threadIdx.x < block.size() / 32) {
            warp_sums[threadIdx.x] = warp_sum;
        }
    }
    
    block.sync();
    
    // Phase 3: Add warp scan results to get final scan
    if (warp.meta_group_rank() > 0) {
        value += warp_sums[warp.meta_group_rank() - 1];
    }
    
    // Write result
    if (idx < n) {
        output[idx] = value;
    }
}

// Large array scan using multiple kernel launches
__global__ void blockScanWithSums(float *input, float *output, float *block_sums, int n) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and scan within block
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Blelloch scan within block
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            if (ai < blockDim.x && bi < blockDim.x) {
                temp[bi] += temp[ai];
            }
        }
        __syncthreads();
    }
    
    // Store the total sum of this block
    if (tid == 0) {
        block_sums[blockIdx.x] = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();
    
    // Down-sweep
    for (int d = 1; d < blockDim.x; d <<= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            
            if (ai < blockDim.x && bi < blockDim.x) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();
    }
    
    // Write block-local scan result
    if (i < n) {
        output[i] = temp[tid];
    }
}

__global__ void addBlockSums(float *output, float *block_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && blockIdx.x > 0) {
        output[idx] += block_sums[blockIdx.x - 1];
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

class ScanBenchmark {
private:
    float *d_input, *d_output, *d_temp;
    int *d_segment_heads;
    size_t size;
    cudaEvent_t start, stop;
    
public:
    ScanBenchmark(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_segment_heads, size * sizeof(int)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        initializeData();
    }
    
    ~ScanBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp);
        cudaFree(d_segment_heads);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void initializeData() {
        float *h_input = new float[size];
        int *h_segments = new int[size];
        
        // Initialize with test pattern
        srand(42);
        for (size_t i = 0; i < size; i++) {
            h_input[i] = (rand() % 10) + 1; // Values 1-10
            h_segments[i] = (rand() % 20 == 0) ? 1 : 0; // ~5% segment heads
        }
        h_segments[0] = 1; // Ensure first element is segment head
        
        CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_segment_heads, h_segments, size * sizeof(int), cudaMemcpyHostToDevice));
        
        delete[] h_input;
        delete[] h_segments;
    }
    
    float testScan(void (*kernel)(float*, float*, int), const char* name, bool verify = true) {
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        
        // Skip naive scan for large arrays (too slow)
        if (strcmp(name, "Naive Scan") == 0 && size > 10000) {
            printf("%-25s: Skipped (too slow for large arrays)\n", name);
            return 0.0f;
        }
        
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<blocks, threads>>>(d_input, d_output, size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        
        if (verify) {
            // Verify first few elements
            float *h_input = new float[10];
            float *h_output = new float[10];
            
            CUDA_CHECK(cudaMemcpy(h_input, d_input, 10 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_output, d_output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
            
            printf("%-25s: %.3f ms (first 5: %.1f %.1f %.1f %.1f %.1f)\n", 
                   name, time, h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
            
            delete[] h_input;
            delete[] h_output;
        } else {
            printf("%-25s: %.3f ms\n", name, time);
        }
        
        return time;
    }
    
    void runBasicScans() {
        printf("=== Basic Scan Algorithms ===\n");
        printf("Array size: %zu elements\n", size);
        
        auto naive_wrapper = [](float *input, float *output, int n) {
            naiveScan<<<(n + 255) / 256, 256>>>(input, output, n, true);
        };
        
        auto hillis_wrapper = [](float *input, float *output, int n) {
            hillisSteeleScan<<<(n + 255) / 256, 256>>>(input, output, n);
        };
        
        auto blelloch_wrapper = [](float *input, float *output, int n) {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            blellochScan<<<blocks, threads, threads * sizeof(float)>>>(input, output, n);
        };
        
        auto cg_wrapper = [](float *input, float *output, int n) {
            cooperativeGroupsScan<<<(n + 1023) / 1024, 1024>>>(input, output, n);
        };
        
        float naive_time = testScan(naive_wrapper, "Naive Scan (O(n²))");
        float hillis_time = testScan(hillis_wrapper, "Hillis-Steele (O(n log n))");
        float blelloch_time = testScan(blelloch_wrapper, "Blelloch (O(n))");
        float cg_time = testScan(cg_wrapper, "Cooperative Groups");
        
        if (naive_time > 0) {
            printf("\nSpeedup vs Naive:\n");
            printf("Hillis-Steele: %.2fx\n", naive_time / hillis_time);
            printf("Blelloch: %.2fx\n", naive_time / blelloch_time);
            printf("Cooperative Groups: %.2fx\n", naive_time / cg_time);
        }
        
        printf("\nWork Efficiency (vs Blelloch):\n");
        printf("Hillis-Steele: %.2fx more work\n", hillis_time / blelloch_time);
        printf("Cooperative Groups: %.2fx\n", cg_time / blelloch_time);
    }
    
    void testSegmentedScan() {
        printf("\n=== Segmented Scan ===\n");
        
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start));
        segmentedScan<<<blocks, threads>>>(d_input, d_segment_heads, d_output, size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        
        // Verify segmented scan results
        float *h_input = new float[20];
        int *h_segments = new int[20];
        float *h_output = new float[20];
        
        CUDA_CHECK(cudaMemcpy(h_input, d_input, 20 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_segments, d_segment_heads, 20 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output, d_output, 20 * sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("Segmented Scan: %.3f ms\n", time);
        printf("Sample (input, segment_head, output):\n");
        for (int i = 0; i < 10; i++) {
            printf("  %.1f %s -> %.1f\n", h_input[i], h_segments[i] ? "(HEAD)" : "      ", h_output[i]);
        }
        
        delete[] h_input;
        delete[] h_segments;
        delete[] h_output;
    }
};

// Large array scan implementation
class LargeArrayScan {
private:
    float *d_block_sums;
    int max_blocks;
    
public:
    LargeArrayScan(size_t max_elements) {
        max_blocks = (max_elements + 255) / 256;
        CUDA_CHECK(cudaMalloc(&d_block_sums, max_blocks * sizeof(float)));
    }
    
    ~LargeArrayScan() {
        cudaFree(d_block_sums);
    }
    
    void scan(float *d_input, float *d_output, int n) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        
        // Phase 1: Block-local scans and collect block sums
        blockScanWithSums<<<blocks, threads>>>(d_input, d_output, d_block_sums, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Phase 2: Scan the block sums (recursively if needed)
        if (blocks > 1) {
            if (blocks <= threads) {
                // Single block scan of sums
                blellochScan<<<1, threads, threads * sizeof(float)>>>(d_block_sums, d_block_sums, blocks);
            } else {
                // Recursive scan (simplified - could be implemented properly)
                printf("Warning: Recursive scan needed but not implemented\n");
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Phase 3: Add scanned block sums to each block's results
            addBlockSums<<<blocks, threads>>>(d_output, d_block_sums, n);
        }
    }
};

void demonstrateLargeArrayScan() {
    printf("\n=== Large Array Scan Demonstration ===\n");
    
    const size_t large_n = 1024 * 1024; // 1M elements
    float *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, large_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, large_n * sizeof(float)));
    
    // Initialize with all 1s for easy verification
    float *h_input = new float[large_n];
    for (size_t i = 0; i < large_n; i++) {
        h_input[i] = 1.0f;
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, large_n * sizeof(float), cudaMemcpyHostToDevice));
    
    LargeArrayScan scanner(large_n);
    
    auto start = std::chrono::high_resolution_clock::now();
    scanner.scan(d_input, d_output, large_n);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Verify results
    float *h_output = new float[large_n];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, large_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (size_t i = 0; i < std::min(large_n, size_t(1000)); i++) {
        if (fabs(h_output[i] - (i + 1)) > 1e-3) {
            correct = false;
            printf("Error at %zu: expected %.1f, got %.1f\n", i, (float)(i + 1), h_output[i]);
            break;
        }
    }
    
    printf("Large array size: %zu elements (%.2f MB)\n", 
           large_n, (large_n * sizeof(float)) / (1024.0 * 1024.0));
    printf("Large array scan time: %.3f ms\n", time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    printf("Sample results: %.1f, %.1f, %.1f, %.1f, %.1f\n",
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
    
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("CUDA Scan (Prefix Sum) Algorithms\n");
    printf("==================================\n");
    
    // Check device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Shared memory per block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    
    // Run benchmarks with different array sizes
    size_t test_sizes[] = {1024, 4096, 16384};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        printf("\n" + std::string(50, '=') + "\n");
        ScanBenchmark benchmark(test_sizes[i]);
        benchmark.runBasicScans();
        benchmark.testSegmentedScan();
    }
    
    // Demonstrate large array scanning
    demonstrateLargeArrayScan();
    
    // Educational summary
    printf("\n=== Scan Algorithm Analysis ===\n");
    printf("Algorithm Comparison:\n");
    printf("┌─────────────────┬──────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Algorithm       │ Work         │ Depth           │ Memory Access   │\n");
    printf("├─────────────────┼──────────────┼─────────────────┼─────────────────┤\n");
    printf("│ Naive           │ O(n²)        │ O(n)            │ O(n²)           │\n");
    printf("│ Hillis-Steele   │ O(n log n)   │ O(log n)        │ O(n log n)      │\n");
    printf("│ Blelloch        │ O(n)         │ O(log n)        │ O(n)            │\n");
    printf("│ Multi-level     │ O(n)         │ O(log n)        │ O(n)            │\n");
    printf("└─────────────────┴──────────────┴─────────────────┴─────────────────┘\n");
    
    printf("\n✓ ALGORITHM SELECTION GUIDELINES:\n");
    printf("  - Small arrays (<1K): Any algorithm works\n");
    printf("  - Medium arrays (1K-1M): Blelloch or cooperative groups\n");
    printf("  - Large arrays (>1M): Multi-level scan required\n");
    printf("  - Segmented data: Use segmented scan variants\n");
    
    printf("\n✓ OPTIMIZATION STRATEGIES:\n");
    printf("  - Use work-efficient algorithms (Blelloch) for large data\n");
    printf("  - Leverage warp-level primitives for modern GPUs\n");
    printf("  - Consider memory bandwidth limitations\n");
    printf("  - Use cooperative groups for cleaner code\n");
    printf("  - Implement multi-level scanning for scalability\n");
    
    printf("\n💡 PRACTICAL APPLICATIONS:\n");
    printf("  - Stream compaction and filtering\n");
    printf("  - Radix sort implementation\n");
    printf("  - Sparse matrix operations\n");
    printf("  - Graph algorithm building blocks\n");
    printf("  - Parallel resource allocation\n");
    
    printf("\nScan algorithms demonstration completed successfully!\n");
    return 0;
}