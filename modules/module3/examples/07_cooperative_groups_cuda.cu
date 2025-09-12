#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

namespace cg = cooperative_groups;

// Modern reduction using cooperative groups
__global__ void cooperativeReduction(float *input, float *output, int n) {
    // Create cooperative group objects
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data with bounds checking
    float value = (tid < n) ? input[tid] : 0.0f;
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        value += warp.shfl_down(value, offset);
    }
    
    // Store warp results to shared memory
    __shared__ float warp_results[32]; // Max 1024/32 = 32 warps per block
    
    if (warp.thread_rank() == 0) {
        warp_results[warp.meta_group_rank()] = value;
    }
    
    block.sync();
    
    // Final reduction within block using first warp
    if (warp.meta_group_rank() == 0) {
        value = (warp.thread_rank() < (blockDim.x + 31) / 32) ? 
                warp_results[warp.thread_rank()] : 0.0f;
        
        #pragma unroll
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            value += warp.shfl_down(value, offset);
        }
        
        if (warp.thread_rank() == 0) {
            output[blockIdx.x] = value;
        }
    }
}

// Matrix multiplication using thread block tiles
__global__ void cooperativeMatMul(float *A, float *B, float *C, int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<256>(block); // 16x16 tile
    
    int row = blockIdx.y * 16 + (threadIdx.x / 16);
    int col = blockIdx.x * 16 + (threadIdx.x % 16);
    
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + 15) / 16; t++) {
        // Cooperative loading of tiles
        int tileRow = threadIdx.x / 16;
        int tileCol = threadIdx.x % 16;
        
        if (row < N && t * 16 + tileCol < N) {
            tileA[tileRow][tileCol] = A[row * N + t * 16 + tileCol];
        } else {
            tileA[tileRow][tileCol] = 0.0f;
        }
        
        if (t * 16 + tileRow < N && col < N) {
            tileB[tileRow][tileCol] = B[(t * 16 + tileRow) * N + col];
        } else {
            tileB[tileRow][tileCol] = 0.0f;
        }
        
        block.sync();
        
        // Compute partial product
        for (int k = 0; k < 16; k++) {
            sum += tileA[threadIdx.x / 16][k] * tileB[k][threadIdx.x % 16];
        }
        
        block.sync();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Histogram computation using cooperative groups
__global__ void cooperativeHistogram(int *input, int *histogram, int n, int num_bins) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    // Shared memory for local histogram
    extern __shared__ int local_hist[];
    
    // Initialize local histogram cooperatively
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    
    block.sync();
    
    // Process input elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        int bin = input[i] % num_bins;
        atomicAdd(&local_hist[bin], 1);
    }
    
    block.sync();
    
    // Cooperatively write local histogram to global
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}

// Advanced warp-level primitives demonstration
__global__ void warpPrimitivesDemo(int *input, int *output, int n) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int value = (tid < n) ? input[tid] : 0;
    
    // Demonstrate various warp-level operations
    int lane_id = warp.thread_rank();
    
    // 1. Shuffle operations
    int next_value = warp.shfl_down(value, 1);  // Get value from next thread
    int prev_value = warp.shfl_up(value, 1);    // Get value from previous thread
    int broadcast_value = warp.shfl(value, 0);   // Broadcast from lane 0
    
    // 2. Voting operations
    bool predicate = (value > 50);
    bool all_true = warp.all(predicate);        // All threads satisfy condition
    bool any_true = warp.any(predicate);        // Any thread satisfies condition
    unsigned int ballot = warp.ballot(predicate); // Bitmask of threads satisfying condition
    
    // 3. Matching operations (if supported)
    unsigned int match_mask = warp.match_all(value);  // Threads with same value
    
    if (tid < n) {
        // Store various results for demonstration
        output[tid] = next_value + prev_value + (all_true ? 1000 : 0) + 
                     (any_true ? 100 : 0) + __popc(ballot);
    }
}

// Multi-GPU cooperative kernel (requires special launch)
__global__ void multiGPUReduction(float *input, float *output, int n, int gpu_id) {
    auto grid = cg::this_multi_grid();
    auto block = cg::this_thread_block();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x + gpu_id * (n / grid.num_grids());
    
    float sum = 0.0f;
    
    // Local reduction
    if (tid < n) {
        sum = input[tid];
    }
    
    // Grid-level reduction using cooperative groups
    sum = cg::reduce(grid, sum, cg::plus<float>());
    
    if (grid.thread_rank() == 0) {
        atomicAdd(output, sum);
    }
}

// Parallel scan using cooperative groups
__global__ void cooperativeScan(float *input, float *output, int n) {
    auto block = cg::this_thread_block();
    
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    
    __shared__ float temp[1024];
    
    // Load input
    if (offset + tid < n) {
        temp[tid] = input[offset + tid];
    } else {
        temp[tid] = 0.0f;
    }
    
    block.sync();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            temp[idx] += temp[idx - stride];
        }
        block.sync();
    }
    
    // Clear last element
    if (tid == 0) {
        temp[blockDim.x - 1] = 0.0f;
    }
    
    block.sync();
    
    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            float tmp = temp[idx];
            temp[idx] += temp[idx - stride];
            temp[idx - stride] = tmp;
        }
        block.sync();
    }
    
    // Write output
    if (offset + tid < n) {
        output[offset + tid] = temp[tid];
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    printf("CUDA Cooperative Groups Examples\n");
    printf("================================\n\n");
    
    const int N = 1048576;  // 1M elements
    const int bytes = N * sizeof(float);
    
    // Initialize input data
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100);
    }
    
    float *d_input, *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    printf("Input array size: %d elements\n", N);
    printf("First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_input[i]);
    }
    printf("\n\n");
    
    // 1. Cooperative Reduction
    printf("1. Cooperative Groups Reduction:\n");
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // First level reduction
    cooperativeReduction<<<blocks, threads>>>(d_input, d_temp, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Final reduction if needed
    while (blocks > 1) {
        int new_blocks = (blocks + threads - 1) / threads;
        cooperativeReduction<<<new_blocks, threads>>>(d_temp, d_temp, blocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        blocks = new_blocks;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double reduction_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify with CPU
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_sum += h_input[i];
    }
    
    printf("   GPU sum: %.2f\n", gpu_sum);
    printf("   CPU sum: %.2f\n", cpu_sum);
    printf("   Difference: %.6f\n", fabs(gpu_sum - cpu_sum));
    printf("   Time: %.3f ms\n\n", reduction_time);
    
    // 2. Cooperative Histogram
    printf("2. Cooperative Histogram:\n");
    
    const int num_bins = 100;
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_histogram = (int*)calloc(num_bins, sizeof(int));
    
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % num_bins;
    }
    
    int *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
    
    start = std::chrono::high_resolution_clock::now();
    cooperativeHistogram<<<blocks, threads, num_bins * sizeof(int)>>>(
        d_data, d_histogram, N, num_bins);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double hist_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, num_bins * sizeof(int), 
                         cudaMemcpyDeviceToHost));
    
    printf("   Histogram computation time: %.3f ms\n", hist_time);
    printf("   First 10 bins: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_histogram[i]);
    }
    printf("\n\n");
    
    // 3. Warp Primitives Demo
    printf("3. Warp-Level Primitives:\n");
    
    int *d_data_int, *d_output_int;
    CUDA_CHECK(cudaMalloc(&d_data_int, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_int, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data_int, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    warpPrimitivesDemo<<<blocks, threads>>>(d_data_int, d_output_int, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double warp_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    int *h_warp_output = (int*)malloc(N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_warp_output, d_output_int, N * sizeof(int), 
                         cudaMemcpyDeviceToHost));
    
    printf("   Warp primitives demo time: %.3f ms\n", warp_time);
    printf("   Sample outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_warp_output[i]);
    }
    printf("\n\n");
    
    // 4. Cooperative Scan
    printf("4. Cooperative Scan (Prefix Sum):\n");
    
    const int scan_size = 1024; // Single block size
    float *h_scan_input = (float*)malloc(scan_size * sizeof(float));
    float *h_scan_output = (float*)malloc(scan_size * sizeof(float));
    
    for (int i = 0; i < scan_size; i++) {
        h_scan_input[i] = 1.0f; // All ones for easy verification
    }
    
    float *d_scan_input, *d_scan_output;
    CUDA_CHECK(cudaMalloc(&d_scan_input, scan_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_output, scan_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scan_input, h_scan_input, scan_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    cooperativeScan<<<1, scan_size>>>(d_scan_input, d_scan_output, scan_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double scan_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(h_scan_output, d_scan_output, scan_size * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    printf("   Scan computation time: %.3f ms\n", scan_time);
    printf("   First 10 scan results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_scan_output[i]);
    }
    printf("\n");
    printf("   Last 10 scan results: ");
    for (int i = scan_size - 10; i < scan_size; i++) {
        printf("%.0f ", h_scan_output[i]);
    }
    printf("\n\n");
    
    // Performance Summary
    printf("Performance Summary:\n");
    printf("  Array size: %d elements\n", N);
    printf("  Cooperative reduction: %.3f ms\n", reduction_time);
    printf("  Cooperative histogram: %.3f ms\n", hist_time);
    printf("  Warp primitives demo: %.3f ms\n", warp_time);
    printf("  Cooperative scan: %.3f ms\n", scan_time);
    
    // Cleanup
    free(h_input); free(h_output); free(h_data); free(h_histogram);
    free(h_warp_output); free(h_scan_input); free(h_scan_output);
    
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_temp);
    cudaFree(d_data); cudaFree(d_histogram);
    cudaFree(d_data_int); cudaFree(d_output_int);
    cudaFree(d_scan_input); cudaFree(d_scan_output);
    
    printf("\nCooperative groups examples completed!\n");
    return 0;
}