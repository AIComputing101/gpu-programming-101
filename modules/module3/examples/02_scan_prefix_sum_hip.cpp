#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "rocm7_utils.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 64  // AMD wavefront size

// Naive inclusive scan implementation
__global__ void naiveScan(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i <= tid; i++) {
            sum += input[i];
        }
        output[tid] = sum;
    }
}

// Hillis-Steele scan - O(n log n) work, O(log n) depth
__global__ void hillisSteeleScan(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (gid < n) {
        temp[tid] = input[gid];
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Up-sweep (build sum tree)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (gid < n) {
        output[gid] = temp[tid];
    }
}

// Blelloch scan - O(n) work, O(log n) depth (work-efficient)
__global__ void blellochScan(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;
    
    // Load input
    if (gid < n) {
        temp[2 * tid] = input[2 * gid];
        temp[2 * tid + 1] = (2 * gid + 1 < n) ? input[2 * gid + 1] : 0.0f;
    } else {
        temp[2 * tid] = temp[2 * tid + 1] = 0.0f;
    }
    
    int n_items = 2 * blockDim.x;
    
    // Up-sweep (reduce) phase
    for (int d = n_items >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (ai < n_items && bi < n_items) {
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        if (n_items - 1 < n_items) {
            temp[n_items - 1] = 0;
        }
    }
    
    // Down-sweep phase
    for (int d = 1; d < n_items; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (ai < n_items && bi < n_items) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    
    __syncthreads();
    
    // Write results
    if (gid < n) {
        output[2 * gid] = temp[2 * tid];
        if (2 * gid + 1 < n) {
            output[2 * gid + 1] = temp[2 * tid + 1];
        }
    }
}

// AMD-optimized wavefront scan using __shfl operations
__global__ void wavefrontScan(float *input, float *output, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    float value = (gid < n) ? input[gid] : 0.0f;
    
    // Intra-wavefront scan using shuffle
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float temp = __shfl_up(value, offset);
        if (lane >= offset) {
            value += temp;
        }
    }
    
    // Store warp scan result
    if (lane == WARP_SIZE - 1) {
        warp_sums[warp_id] = value;
    }
    
    __syncthreads();
    
    // Scan the warp sums
    if (warp_id == 0) {
        float warp_sum = (lane < blockDim.x / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            float temp = __shfl_up(warp_sum, offset);
            if (lane >= offset) {
                warp_sum += temp;
            }
        }
        
        warp_sums[lane] = warp_sum;
    }
    
    __syncthreads();
    
    // Add warp sum offset to get final result
    if (warp_id > 0) {
        value += warp_sums[warp_id - 1];
    }
    
    if (gid < n) {
        output[gid] = value;
    }
}

// Segmented scan for multiple independent sequences
__global__ void segmentedScan(float *input, int *flags, float *output, int n) {
    extern __shared__ float temp[];
    int *flag_temp = (int*)&temp[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input and flags
    if (gid < n) {
        temp[tid] = input[gid];
        flag_temp[tid] = flags[gid];
    } else {
        temp[tid] = 0.0f;
        flag_temp[tid] = 0;
    }
    
    __syncthreads();
    
    // Segmented scan using flags
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = temp[tid];
        int flag = flag_temp[tid];
        
        if (tid >= stride) {
            if (!flag_temp[tid - stride]) {
                val += temp[tid - stride];
            }
        }
        
        __syncthreads();
        temp[tid] = val;
        __syncthreads();
    }
    
    if (gid < n) {
        output[gid] = temp[tid];
    }
}

// Large array scan using multiple passes
void largeScan(float *input, float *output, int n) {
    const int block_size = BLOCK_SIZE;
    const int blocks_per_grid = (n + block_size - 1) / block_size;
    
    if (blocks_per_grid == 1) {
        // Single block - use Blelloch scan
        dim3 block(block_size / 2);
        blellochScan<<<1, block, block_size * sizeof(float)>>>(input, output, n);
        return;
    }
    
    // Multi-block approach
    float *block_sums, *block_scan_sums;
    HIP_CHECK(hipMalloc(&block_sums, blocks_per_grid * sizeof(float)));
    HIP_CHECK(hipMalloc(&block_scan_sums, blocks_per_grid * sizeof(float)));
    
    // Phase 1: Scan each block independently and collect block sums
    // (This would require a modified kernel to collect sums - simplified here)
    dim3 grid(blocks_per_grid);
    dim3 block(block_size);
    wavefrontScan<<<grid, block>>>(input, output, n);
    
    // Phase 2: Scan the block sums (recursive call for simplicity)
    // Phase 3: Add scanned block sums to each block's results
    
    HIP_CHECK(hipFree(block_sums));
    HIP_CHECK(hipFree(block_scan_sums));
}

void printArray(float *arr, int n, const char *name, int max_print = 10) {
    printf("%s: ", name);
    int print_count = (n < max_print) ? n : max_print;
    for (int i = 0; i < print_count; i++) {
        printf("%.1f ", arr[i]);
    }
    if (n > max_print) printf("... ");
    printf("\n");
}

bool verifyScan(float *input, float *output, int n, bool exclusive = false) {
    float expected = exclusive ? 0.0f : input[0];
    
    if (!exclusive && fabs(output[0] - expected) > 1e-5) return false;
    if (exclusive && fabs(output[0] - 0.0f) > 1e-5) return false;
    
    for (int i = 1; i < n; i++) {
        expected += input[exclusive ? i - 1 : i];
        if (fabs(output[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: expected %.2f, got %.2f\n", 
                   i, expected, output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("HIP Scan (Prefix Sum) Algorithms\n");
    printf("================================\n\n");
    
    // Check HIP device info
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
    printf("Wavefront size: %d\n\n", prop.warpSize);
    
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    // Host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *h_reference = (float*)malloc(bytes);
    int *h_flags = (int*)malloc(N * sizeof(int));
    
    // Initialize test data
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10 + 1); // Values 1-10
        h_flags[i] = (i % 64 == 0) ? 1 : 0;    // Segment boundaries
    }
    
    printArray(h_input, N, "Input");
    
    // Device memory
    float *d_input, *d_output;
    int *d_flags;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMalloc(&d_flags, N * sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_flags, h_flags, N * sizeof(int), hipMemcpyHostToDevice));
    
    // Kernel launch parameters
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Timing variables
    double naive_time = 0.0, hillis_time, blelloch_time;
    
    // 1. Naive Scan (for small arrays only)
    if (N <= 1024) {
        printf("1. Naive Scan:\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        naiveScan<<<grid, block>>>(d_input, d_output, N);
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        double naive_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
        
        bool correct = verifyScan(h_input, h_output, N);
        printf("   Time: %.3f ms\n", naive_time);
        printf("   Result: %s\n", correct ? "CORRECT" : "INCORRECT");
        printArray(h_output, N, "   Output");
        printf("\n");
    }
    
    // 2. Hillis-Steele Scan
    printf("2. Hillis-Steele Scan (step-efficient):\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    hillisSteeleScan<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double hillis_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    bool correct = verifyScan(h_input, h_output, N);
    printf("   Time: %.3f ms\n", hillis_time);
    printf("   Work complexity: O(n log n)\n");
    printf("   Depth complexity: O(log n)\n");
    printf("   Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printArray(h_output, N, "   Output");
    printf("\n");
    
    // 3. Blelloch Scan (work-efficient)
    printf("3. Blelloch Scan (work-efficient):\n");
    
    start = std::chrono::high_resolution_clock::now();
    dim3 blelloch_block(BLOCK_SIZE / 2);
    blellochScan<<<grid, blelloch_block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double blelloch_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    printf("   Time: %.3f ms\n", blelloch_time);
    printf("   Work complexity: O(n)\n");
    printf("   Depth complexity: O(log n)\n");
    printArray(h_output, N, "   Output");
    printf("\n");
    
    // 4. AMD Wavefront-Optimized Scan
    printf("4. Wavefront-Optimized Scan:\n");
    
    start = std::chrono::high_resolution_clock::now();
    wavefrontScan<<<grid, block>>>(d_input, d_output, N);
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double wavefront_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    correct = verifyScan(h_input, h_output, N);
    printf("   Time: %.3f ms\n", wavefront_time);
    printf("   Optimized for AMD wavefront size: %d\n", WARP_SIZE);
    printf("   Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printArray(h_output, N, "   Output");
    printf("\n");
    
    // 5. Segmented Scan
    printf("5. Segmented Scan:\n");
    
    start = std::chrono::high_resolution_clock::now();
    segmentedScan<<<grid, block, (BLOCK_SIZE * sizeof(float)) + (BLOCK_SIZE * sizeof(int))>>>(
        d_input, d_flags, d_output, N);
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double segmented_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    printf("   Time: %.3f ms\n", segmented_time);
    printf("   Processes multiple independent sequences\n");
    printf("   Segment boundaries every 64 elements\n");
    printArray(h_output, N, "   Output");
    printf("\n");
    
    // CPU reference for performance comparison
    printf("6. CPU Reference:\n");
    
    start = std::chrono::high_resolution_clock::now();
    h_reference[0] = h_input[0];
    for (int i = 1; i < N; i++) {
        h_reference[i] = h_reference[i-1] + h_input[i];
    }
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("   CPU time: %.3f ms\n", cpu_time);
    
    // Performance comparison
    printf("\nPerformance Comparison:\n");
    printf("  Array size: %d elements\n", N);
    if (N <= 1024) {
        printf("  Naive scan: %.3f ms (%.2fx slower than CPU)\n", 
               naive_time, naive_time / cpu_time);
    }
    printf("  Hillis-Steele: %.3f ms (%.2fx vs CPU)\n", 
           hillis_time, hillis_time / cpu_time);
    printf("  Blelloch: %.3f ms (%.2fx vs CPU)\n", 
           blelloch_time, blelloch_time / cpu_time);
    printf("  Wavefront-optimized: %.3f ms (%.2fx vs CPU)\n", 
           wavefront_time, wavefront_time / cpu_time);
    printf("  Segmented scan: %.3f ms\n", segmented_time);
    printf("  CPU reference: %.3f ms\n", cpu_time);
    
    printf("\nAlgorithm Analysis:\n");
    printf("  - Hillis-Steele: More steps but simpler, good for small arrays\n");
    printf("  - Blelloch: Work-efficient, better for large arrays\n");
    printf("  - Wavefront: Optimized for AMD GPU architecture\n");
    printf("  - Segmented: Handles multiple independent sequences\n");
    
    // Cleanup
    free(h_input); free(h_output); free(h_reference); free(h_flags);
    HIP_CHECK(hipFree(d_input)); HIP_CHECK(hipFree(d_output)); HIP_CHECK(hipFree(d_flags));
    
    printf("\nHIP scan algorithms completed successfully!\n");
    return 0;
}