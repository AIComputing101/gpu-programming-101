#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define ARRAY_SIZE (16 * 1024 * 1024)  // 16M elements
#define BLOCK_SIZE 256
#define NUM_ITERATIONS 10

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timer class for performance measurement
class OptimizationTimer {
private:
    cudaEvent_t start, stop;
    
public:
    OptimizationTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~OptimizationTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float endTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// 1. Occupancy Optimization Examples
//===================================

// Poor occupancy - high register usage
__global__ void highRegisterUsage(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Excessive local variables consuming registers
        float val1 = data[idx];
        float val2 = val1 * 2.0f;
        float val3 = sinf(val2);
        float val4 = cosf(val3);
        float val5 = expf(val4);
        float val6 = logf(fabsf(val5) + 1.0f);
        float val7 = powf(val6, 2.0f);
        float val8 = sqrtf(fabsf(val7));
        float val9 = tanf(val8);
        float val10 = atanf(val9);
        
        // Many intermediate calculations
        float result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9 + val10;
        
        data[idx] = result;
    }
}

// Optimized occupancy - controlled register usage
__global__ void __launch_bounds__(256, 4) // 256 threads/block, min 4 blocks/SM
optimizedRegisterUsage(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Reuse variables to reduce register pressure
        val = val * 2.0f;
        val = sinf(val);
        val = cosf(val);
        val = expf(val);
        val = logf(fabsf(val) + 1.0f);
        
        data[idx] = val;
    }
}

// 2. Warp Efficiency Optimization
//===============================

// Poor warp efficiency - high divergence
__global__ void divergentExecution(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // High divergence - threads take very different paths
        if (value > 0.8f) {
            for (int i = 0; i < 200; i++) {
                value = sinf(value + i * 0.01f);
            }
        } else if (value > 0.6f) {
            for (int i = 0; i < 150; i++) {
                value = cosf(value + i * 0.01f);
            }
        } else if (value > 0.4f) {
            for (int i = 0; i < 100; i++) {
                value = tanf(value + i * 0.01f);
            }
        } else if (value > 0.2f) {
            for (int i = 0; i < 50; i++) {
                value = expf(value * 0.1f + i * 0.001f);
            }
        } else {
            for (int i = 0; i < 25; i++) {
                value = logf(fabsf(value) + i * 0.1f + 1.0f);
            }
        }
        
        data[idx] = value;
    }
}

// Optimized warp efficiency - minimal divergence
__global__ void convergentExecution(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // All threads execute same operations, results selected by predicates
        float sin_result = value;
        float cos_result = value;
        float tan_result = value;
        float exp_result = value;
        float log_result = value;
        
        // Fixed iteration count for all threads
        for (int i = 0; i < 200; i++) {
            sin_result = sinf(sin_result + i * 0.01f);
            if (i < 150) cos_result = cosf(cos_result + i * 0.01f);
            if (i < 100) tan_result = tanf(tan_result + i * 0.01f);
            if (i < 50) exp_result = expf(exp_result * 0.1f + i * 0.001f);
            if (i < 25) log_result = logf(fabsf(log_result) + i * 0.1f + 1.0f);
        }
        
        // Predicated selection instead of branching
        float result;
        if (value > 0.8f) {
            result = sin_result;
        } else if (value > 0.6f) {
            result = cos_result;
        } else if (value > 0.4f) {
            result = tan_result;
        } else if (value > 0.2f) {
            result = exp_result;
        } else {
            result = log_result;
        }
        
        data[idx] = result;
    }
}

// Cooperative groups optimization
__global__ void cooperativeGroupsOptimized(float *data, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Warp-level reduction using cooperative groups
        value = sinf(value * 2.0f);
        
        // Efficient warp-level sum reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            value += warp.shfl_down(value, offset);
        }
        
        // Only one thread per warp writes back
        if (warp.thread_rank() == 0) {
            data[blockIdx.x * blockDim.x / 32 + warp.meta_group_rank()] = value;
        }
    }
}

// 3. Instruction Optimization
//===========================

// Slow math functions
__global__ void slowMathKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Use slow, precise math functions
        value = sin(value * 2.0);          // Slow double precision
        value = cos(value * 1.5);          // Slow double precision
        value = exp(value * 0.1);          // Slow double precision
        value = log(fabs(value) + 1.0);    // Slow double precision
        value = sqrt(fabs(value));         // Slow double precision
        
        data[idx] = (float)value;
    }
}

// Fast math functions
__global__ void fastMathKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Use fast, single precision intrinsics
        value = __sinf(value * 2.0f);      // Fast single precision
        value = __cosf(value * 1.5f);      // Fast single precision
        value = __expf(value * 0.1f);      // Fast single precision
        value = __logf(fabsf(value) + 1.0f); // Fast single precision
        value = __fsqrt_rn(fabsf(value));  // Fast single precision square root
        
        data[idx] = value;
    }
}

// Vectorized memory operations
__global__ void scalarMemoryOps(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Scalar operations - one float at a time
        float value = input[idx];
        value = value * 2.0f + 1.0f;
        output[idx] = value;
    }
}

__global__ void vectorizedMemoryOps(float4 *input, float4 *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Vectorized operations - four floats at a time
        float4 value = input[idx];
        value.x = value.x * 2.0f + 1.0f;
        value.y = value.y * 2.0f + 1.0f;
        value.z = value.z * 2.0f + 1.0f;
        value.w = value.w * 2.0f + 1.0f;
        output[idx] = value;
    }
}

// 4. Loop Optimization
//=====================

// Unoptimized loop
__global__ void unoptimizedLoop(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        
        // Loop with dependencies and inefficient access
        for (int i = 0; i < 64; i++) {
            sum += sinf(data[idx] + i * 0.1f);
        }
        
        data[idx] = sum;
    }
}

// Loop unrolling optimization
__global__ void unrolledLoop(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float base_val = data[idx];
        float sum = 0.0f;
        
        // Manual loop unrolling (4x)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            sum += sinf(base_val + (i*4) * 0.1f);
            sum += sinf(base_val + (i*4+1) * 0.1f);
            sum += sinf(base_val + (i*4+2) * 0.1f);
            sum += sinf(base_val + (i*4+3) * 0.1f);
        }
        
        data[idx] = sum;
    }
}

// Thread coarsening - each thread processes multiple elements
__global__ void threadCoarsening(float *data, int n, int coarsening_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements
    for (int i = 0; i < coarsening_factor; i++) {
        int global_idx = idx * coarsening_factor + i;
        if (global_idx < n) {
            float value = data[global_idx];
            value = sinf(value * 2.0f) + cosf(value * 1.5f);
            data[global_idx] = value;
        }
    }
}

// 5. Shared Memory Bank Conflict Optimization
//===========================================

// Bank conflicts example
__global__ void bankConflictsKernel(float *input, float *output, int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Accessing with conflicts - stride that causes bank conflicts
    int conflict_idx = (tid * 17) % BLOCK_SIZE; // Creates bank conflicts
    float value = shared_data[conflict_idx];
    
    // Simple computation
    value = value * 2.0f + 1.0f;
    
    __syncthreads();
    shared_data[tid] = value;
    __syncthreads();
    
    if (idx < n) {
        output[idx] = shared_data[tid];
    }
}

// Bank conflict free version
__global__ void noBankConflictsKernel(float *input, float *output, int n) {
    __shared__ float shared_data[BLOCK_SIZE + 32]; // Padding to avoid conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Sequential access - no bank conflicts
    float value = shared_data[tid];
    
    // Simple computation
    value = value * 2.0f + 1.0f;
    
    __syncthreads();
    shared_data[tid] = value;
    __syncthreads();
    
    if (idx < n) {
        output[idx] = shared_data[tid];
    }
}

// Performance testing functions
void testOccupancyOptimization() {
    printf("=== Occupancy Optimization Test ===\n");
    
    float *d_data;
    size_t size = ARRAY_SIZE * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Initialize test data
    std::vector<float> h_data(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    
    OptimizationTimer timer;
    
    // Test high register usage kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        highRegisterUsage<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float high_reg_time = total_time / NUM_ITERATIONS;
    
    // Test optimized register usage kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        optimizedRegisterUsage<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float opt_reg_time = total_time / NUM_ITERATIONS;
    
    printf("High register usage:  %.3f ms\n", high_reg_time);
    printf("Optimized registers:  %.3f ms [%.1fx %s]\n", 
           opt_reg_time, fabs(high_reg_time / opt_reg_time),
           opt_reg_time < high_reg_time ? "faster" : "slower");
    printf("Register optimization speedup: %.1fx\n", high_reg_time / opt_reg_time);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

void testWarpEfficiency() {
    printf("=== Warp Efficiency Test ===\n");
    
    float *d_data;
    size_t size = ARRAY_SIZE * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Initialize test data with values that will cause divergence
    std::vector<float> h_data(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    
    OptimizationTimer timer;
    
    // Test divergent execution
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        divergentExecution<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float divergent_time = total_time / NUM_ITERATIONS;
    
    // Test convergent execution
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        convergentExecution<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float convergent_time = total_time / NUM_ITERATIONS;
    
    // Test cooperative groups optimization
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        cooperativeGroupsOptimized<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float coop_time = total_time / NUM_ITERATIONS;
    
    printf("Divergent execution:     %.3f ms\n", divergent_time);
    printf("Convergent execution:    %.3f ms [%.1fx %s]\n", 
           convergent_time, fabs(divergent_time / convergent_time),
           convergent_time < divergent_time ? "faster" : "slower");
    printf("Cooperative groups:      %.3f ms [%.1fx %s]\n", 
           coop_time, fabs(divergent_time / coop_time),
           coop_time < divergent_time ? "faster" : "slower");
    printf("Best divergence optimization speedup: %.1fx\n", 
           divergent_time / fmin(convergent_time, coop_time));
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

void testInstructionOptimization() {
    printf("=== Instruction Optimization Test ===\n");
    
    float *d_data;
    float4 *d_data_vec;
    size_t size = ARRAY_SIZE * sizeof(float);
    size_t size_vec = (ARRAY_SIZE / 4) * sizeof(float4);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_data_vec, size_vec));
    
    // Initialize test data
    std::vector<float> h_data(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 0.1f; // Small values for math functions
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    dim3 grid_vec((ARRAY_SIZE / 4 + block.x - 1) / block.x);
    
    OptimizationTimer timer;
    
    // Test slow math functions
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        slowMathKernel<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float slow_math_time = total_time / NUM_ITERATIONS;
    
    // Test fast math functions
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        fastMathKernel<<<grid, block>>>(d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float fast_math_time = total_time / NUM_ITERATIONS;
    
    printf("Slow math functions:  %.3f ms\n", slow_math_time);
    printf("Fast math functions:  %.3f ms [%.1fx %s]\n", 
           fast_math_time, fabs(slow_math_time / fast_math_time),
           fast_math_time < slow_math_time ? "faster" : "slower");
    printf("Fast math speedup: %.1fx\n", slow_math_time / fast_math_time);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_data_vec));
}

void testLoopOptimization() {
    printf("=== Loop Optimization Test ===\n");
    
    float *d_data;
    size_t size = ARRAY_SIZE * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Initialize test data
    std::vector<float> h_data(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 0.1f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    
    OptimizationTimer timer;
    
    // Test unoptimized loop
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    timer.startTimer();
    unoptimizedLoop<<<grid, block>>>(d_data, ARRAY_SIZE);
    cudaDeviceSynchronize();
    float unopt_time = timer.endTimer();
    
    // Test unrolled loop
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    timer.startTimer();
    unrolledLoop<<<grid, block>>>(d_data, ARRAY_SIZE);
    cudaDeviceSynchronize();
    float unrolled_time = timer.endTimer();
    
    // Test thread coarsening
    int coarsening_factor = 4;
    dim3 coarse_grid((ARRAY_SIZE / coarsening_factor + block.x - 1) / block.x);
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    timer.startTimer();
    threadCoarsening<<<coarse_grid, block>>>(d_data, ARRAY_SIZE, coarsening_factor);
    cudaDeviceSynchronize();
    float coarsening_time = timer.endTimer();
    
    printf("Unoptimized loop:     %.3f ms\n", unopt_time);
    printf("Unrolled loop:        %.3f ms [%.1fx %s]\n", 
           unrolled_time, fabs(unopt_time / unrolled_time),
           unrolled_time < unopt_time ? "faster" : "slower");
    printf("Thread coarsening:    %.3f ms [%.1fx %s]\n", 
           coarsening_time, fabs(unopt_time / coarsening_time),
           coarsening_time < unopt_time ? "faster" : "slower");
    printf("Best loop optimization speedup: %.1fx\n", 
           unopt_time / fmin(unrolled_time, coarsening_time));
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

void printOptimizationGuidance() {
    printf("=== Kernel Optimization Guidance ===\n");
    printf("Key optimization strategies:\n\n");
    
    printf("1. Occupancy Optimization:\n");
    printf("   - Use __launch_bounds__ to guide compiler\n");
    printf("   - Balance register usage vs parallelism\n");
    printf("   - Monitor occupancy with profiling tools\n");
    printf("   - Consider shared memory usage impact\n\n");
    
    printf("2. Warp Efficiency:\n");
    printf("   - Minimize thread divergence\n");
    printf("   - Use predicated execution when possible\n");
    printf("   - Leverage cooperative groups for modern optimization\n");
    printf("   - Structure data to reduce branching\n\n");
    
    printf("3. Instruction Optimization:\n");
    printf("   - Use fast math intrinsics (__sinf, __cosf, etc.)\n");
    printf("   - Prefer single precision over double precision\n");
    printf("   - Use vectorized memory operations (float4)\n");
    printf("   - Avoid expensive operations in inner loops\n\n");
    
    printf("4. Loop Optimization:\n");
    printf("   - Apply loop unrolling judiciously\n");
    printf("   - Consider thread coarsening for better ILP\n");
    printf("   - Use #pragma unroll compiler hints\n");
    printf("   - Balance unrolling vs register pressure\n\n");
    
    printf("5. Memory Pattern Optimization:\n");
    printf("   - Ensure coalesced memory access\n");
    printf("   - Avoid shared memory bank conflicts\n");
    printf("   - Use appropriate data layouts (SoA vs AoS)\n");
    printf("   - Leverage constant memory for read-only data\n\n");
}

int main() {
    printf("=== CUDA Kernel Optimization Comprehensive Test ===\n\n");
    
    // Run all optimization tests
    testOccupancyOptimization();
    testWarpEfficiency();
    testInstructionOptimization();
    testLoopOptimization();
    
    // Print optimization guidance
    printOptimizationGuidance();
    
    printf("=== Kernel Optimization Tests Complete ===\n");
    printf("Use profiling tools for detailed occupancy and efficiency analysis!\n");
    
    return 0;
}