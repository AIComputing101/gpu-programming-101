/**
 * Module 6: Fundamental Parallel Algorithms
 * Example 2: Stencil Computations (HIP Implementation)
 * 
 * This example demonstrates various stencil computation implementations using HIP:
 * 1. 1D Stencil operations (heat equation simulation)
 * 2. 2D Stencil operations (Laplacian operator)
 * 3. 3D Stencil operations (physics simulations)
 * 4. Optimized implementations with LDS and thread coarsening
 */

#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cassert>

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
const int BLOCK_SIZE = 16;
const int RADIUS = 3;
const int COARSENING_FACTOR = 4;

// Stencil coefficients for different operators
__constant__ float c_stencil_1d[2 * RADIUS + 1];
__constant__ float c_laplacian_2d[9] = {0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f};
__constant__ float c_laplacian_3d[27];

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
// 1D Stencil Kernels
// =============================================================================

/**
 * Naive 1D stencil kernel (3-point stencil for heat equation)
 */
__global__ void stencil_1d_naive(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= RADIUS && idx < n - RADIUS) {
        float result = 0.0f;
        
        // Apply 3-point stencil: -1, 2, -1 (scaled by dt/dx^2)
        result = -input[idx - 1] + 2.0f * input[idx] - input[idx + 1];
        
        output[idx] = input[idx] + 0.1f * result; // dt/dx^2 = 0.1
    } else if (idx < n) {
        // Boundary conditions (Dirichlet: fixed values)
        output[idx] = input[idx];
    }
}

/**
 * Optimized 1D stencil with Local Data Store (LDS)
 */
__global__ void stencil_1d_shared(float *input, float *output, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into LDS with halo regions
    int shared_idx = tid + RADIUS;
    
    // Load main data
    if (idx < n) {
        shared_data[shared_idx] = input[idx];
    } else {
        shared_data[shared_idx] = 0.0f;
    }
    
    // Load left halo
    if (tid < RADIUS && blockIdx.x > 0) {
        shared_data[tid] = input[idx - RADIUS];
    } else if (tid < RADIUS) {
        shared_data[tid] = input[idx]; // Boundary condition
    }
    
    // Load right halo
    if (tid < RADIUS && idx + blockDim.x < n) {
        shared_data[shared_idx + blockDim.x] = input[idx + blockDim.x];
    } else if (tid < RADIUS && idx + blockDim.x >= n) {
        shared_data[shared_idx + blockDim.x] = input[n - 1]; // Boundary condition
    }
    
    __syncthreads();
    
    // Apply stencil
    if (idx >= RADIUS && idx < n - RADIUS) {
        float result = -shared_data[shared_idx - 1] + 2.0f * shared_data[shared_idx] - shared_data[shared_idx + 1];
        output[idx] = input[idx] + 0.1f * result;
    } else if (idx < n) {
        output[idx] = input[idx];
    }
}

/**
 * Thread coarsening 1D stencil with wavefront optimizations
 */
__global__ void stencil_1d_coarsened(float *input, float *output, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int base_idx = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;
    
    // Load data with coarsening (optimized for wavefront size)
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        int idx = base_idx + i * blockDim.x;
        int shared_idx = tid + i * blockDim.x + RADIUS;
        
        if (idx < n) {
            shared_data[shared_idx] = input[idx];
        } else {
            shared_data[shared_idx] = 0.0f;
        }
    }
    
    // Load halo regions with wavefront coordination
    if (tid < RADIUS) {
        // Left halo
        int left_idx = base_idx - RADIUS + tid;
        if (left_idx >= 0) {
            shared_data[tid] = input[left_idx];
        } else {
            shared_data[tid] = input[0];
        }
        
        // Right halo
        int right_idx = base_idx + COARSENING_FACTOR * blockDim.x + tid;
        int right_shared_idx = COARSENING_FACTOR * blockDim.x + RADIUS + tid;
        if (right_idx < n) {
            shared_data[right_shared_idx] = input[right_idx];
        } else {
            shared_data[right_shared_idx] = input[n - 1];
        }
    }
    
    __syncthreads();
    
    // Apply stencil with coarsening
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        int idx = base_idx + i * blockDim.x;
        int shared_idx = tid + i * blockDim.x + RADIUS;
        
        if (idx >= RADIUS && idx < n - RADIUS) {
            float result = -shared_data[shared_idx - 1] + 2.0f * shared_data[shared_idx] - shared_data[shared_idx + 1];
            output[idx] = input[idx] + 0.1f * result;
        } else if (idx < n) {
            output[idx] = input[idx];
        }
    }
}

// =============================================================================
// 2D Stencil Kernels
// =============================================================================

/**
 * Naive 2D Laplacian stencil
 */
__global__ void stencil_2d_naive(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;
    
    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        float center = input[idx];
        float north = input[idx - width];
        float south = input[idx + width];
        float east = input[idx + 1];
        float west = input[idx - 1];
        
        // 5-point Laplacian stencil
        output[idx] = center + 0.1f * (-4.0f * center + north + south + east + west);
    } else if (col < width && row < height) {
        // Boundary conditions
        output[idx] = input[idx];
    }
}

/**
 * Optimized 2D stencil with Local Data Store (LDS)
 */
__global__ void stencil_2d_shared(float *input, float *output, int width, int height) {
    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    int shared_width = blockDim.x + 2 * RADIUS;
    int shared_height = blockDim.y + 2 * RADIUS;
    
    // Load data into LDS with halo - optimized for AMD memory hierarchy
    for (int dy = 0; dy < shared_height; dy += blockDim.y) {
        for (int dx = 0; dx < shared_width; dx += blockDim.x) {
            int shared_row = ty + dy;
            int shared_col = tx + dx;
            
            if (shared_row < shared_height && shared_col < shared_width) {
                int input_row = blockIdx.y * blockDim.y + shared_row - RADIUS;
                int input_col = blockIdx.x * blockDim.x + shared_col - RADIUS;
                
                int shared_idx = shared_row * shared_width + shared_col;
                
                // Handle boundaries - optimized for AMD memory channels
                input_row = max(0, min(input_row, height - 1));
                input_col = max(0, min(input_col, width - 1));
                
                int input_idx = input_row * width + input_col;
                shared_data[shared_idx] = input[input_idx];
            }
        }
    }
    
    __syncthreads();
    
    // Apply stencil
    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        int shared_idx = (ty + RADIUS) * shared_width + (tx + RADIUS);
        
        float center = shared_data[shared_idx];
        float north = shared_data[shared_idx - shared_width];
        float south = shared_data[shared_idx + shared_width];
        float east = shared_data[shared_idx + 1];
        float west = shared_data[shared_idx - 1];
        
        int output_idx = row * width + col;
        output[output_idx] = center + 0.1f * (-4.0f * center + north + south + east + west);
    } else if (col < width && row < height) {
        int output_idx = row * width + col;
        output[output_idx] = input[output_idx];
    }
}

/**
 * 2D stencil with register blocking - AMD wavefront optimized
 */
__global__ void stencil_2d_register_blocked(float *input, float *output, int width, int height) {
    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * (blockDim.x - 2 * RADIUS) + tx;
    int row = blockIdx.y * (blockDim.y - 2 * RADIUS) + ty;
    
    // Each thread loads one element - consider wavefront size for optimal access
    if (col < width && row < height) {
        int shared_idx = ty * blockDim.x + tx;
        int input_idx = row * width + col;
        shared_data[shared_idx] = input[input_idx];
    } else {
        int shared_idx = ty * blockDim.x + tx;
        shared_data[shared_idx] = 0.0f;
    }
    
    __syncthreads();
    
    // Apply stencil only for interior threads
    if (tx >= RADIUS && tx < blockDim.x - RADIUS && 
        ty >= RADIUS && ty < blockDim.y - RADIUS &&
        col < width && row < height) {
        
        int shared_idx = ty * blockDim.x + tx;
        
        float center = shared_data[shared_idx];
        float north = shared_data[shared_idx - blockDim.x];
        float south = shared_data[shared_idx + blockDim.x];
        float east = shared_data[shared_idx + 1];
        float west = shared_data[shared_idx - 1];
        
        // Only update interior points
        if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
            int output_idx = row * width + col;
            output[output_idx] = center + 0.1f * (-4.0f * center + north + south + east + west);
        }
    }
}

// =============================================================================
// 3D Stencil Kernels
// =============================================================================

/**
 * Naive 3D Laplacian stencil (7-point)
 */
__global__ void stencil_3d_naive(float *input, float *output, int width, int height, int depth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int slice = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (col > 0 && col < width - 1 && 
        row > 0 && row < height - 1 && 
        slice > 0 && slice < depth - 1) {
        
        int idx = slice * width * height + row * width + col;
        int slice_size = width * height;
        
        float center = input[idx];
        float north = input[idx - width];
        float south = input[idx + width];
        float east = input[idx + 1];
        float west = input[idx - 1];
        float up = input[idx + slice_size];
        float down = input[idx - slice_size];
        
        // 7-point Laplacian stencil
        output[idx] = center + 0.05f * (-6.0f * center + north + south + east + west + up + down);
    } else if (col < width && row < height && slice < depth) {
        int idx = slice * width * height + row * width + col;
        output[idx] = input[idx];
    }
}

/**
 * Wavefront-optimized reduction for shared memory coordination
 */
__device__ float wavefront_reduce_sum(float val) {
    #ifdef __HIP_PLATFORM_AMD__
    // AMD wavefront is typically 64 threads
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
    }
    #else
    // NVIDIA warp is 32 threads
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 32);
    }
    #endif
    return val;
}

// =============================================================================
// Host Functions
// =============================================================================

/**
 * Print GPU information
 */
void print_gpu_info() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Wavefront Size: %d\n", prop.warpSize);
    
    // Print platform information
    #ifdef __HIP_PLATFORM_AMD__
        printf("Platform: AMD ROCm\n");
        printf("Memory Channels: Optimized for HBM\n");
    #elif defined(__HIP_PLATFORM_NVIDIA__)
        printf("Platform: NVIDIA CUDA\n");
        printf("Memory: GDDR optimizations\n");
    #else
        printf("Platform: Unknown\n");
    #endif
    printf("\n");
}

/**
 * Initialize test data
 */
void initialize_1d_data(float *data, int n) {
    for (int i = 0; i < n; i++) {
        // Heat source in the middle
        if (i > n/3 && i < 2*n/3) {
            data[i] = 1.0f;
        } else {
            data[i] = 0.0f;
        }
    }
}

void initialize_2d_data(float *data, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            // Heat source in the center
            if (abs(row - height/2) < height/8 && abs(col - width/2) < width/8) {
                data[idx] = 1.0f;
            } else {
                data[idx] = 0.0f;
            }
        }
    }
}

void initialize_3d_data(float *data, int width, int height, int depth) {
    for (int slice = 0; slice < depth; slice++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = slice * width * height + row * width + col;
                // Heat source in the center
                if (abs(slice - depth/2) < depth/8 && 
                    abs(row - height/2) < height/8 && 
                    abs(col - width/2) < width/8) {
                    data[idx] = 1.0f;
                } else {
                    data[idx] = 0.0f;
                }
            }
        }
    }
}

/**
 * Verify results
 */
bool verify_result(float *gpu_result, float *cpu_reference, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(gpu_result[i] - cpu_reference[i]) > tolerance) {
            printf("Verification failed at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                   i, gpu_result[i], cpu_reference[i], 
                   fabsf(gpu_result[i] - cpu_reference[i]));
            return false;
        }
    }
    return true;
}

/**
 * CPU reference for 1D stencil
 */
void stencil_1d_cpu_reference(float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        if (i >= RADIUS && i < n - RADIUS) {
            float result = -input[i - 1] + 2.0f * input[i] - input[i + 1];
            output[i] = input[i] + 0.1f * result;
        } else {
            output[i] = input[i];
        }
    }
}

/**
 * CPU reference for 2D stencil
 */
void stencil_2d_cpu_reference(float *input, float *output, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            
            if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
                float center = input[idx];
                float north = input[idx - width];
                float south = input[idx + width];
                float east = input[idx + 1];
                float west = input[idx - 1];
                
                output[idx] = center + 0.1f * (-4.0f * center + north + south + east + west);
            } else {
                output[idx] = input[idx];
            }
        }
    }
}

/**
 * Benchmark 1D stencil algorithms
 */
void benchmark_stencil_1d() {
    printf("=== 1D Stencil Benchmark ===\n");
    
    const int n = 1024 * 1024;
    const int num_iterations = 1000;
    
    // Allocate host memory
    float *h_input = new float[n];
    float *h_output_naive = new float[n];
    float *h_output_shared = new float[n];
    float *h_output_coarsened = new float[n];
    float *h_output_cpu = new float[n];
    
    // Initialize data
    initialize_1d_data(h_input, n);
    
    // Allocate device memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive 1D stencil
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_1d_naive, grid_size, block_size, 0, 0, d_input, d_output, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_naive, d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // Benchmark shared memory 1D stencil
    int shared_mem_size = (block_size.x + 2 * RADIUS) * sizeof(float);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_1d_shared, grid_size, block_size, shared_mem_size, 0, d_input, d_output, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double shared_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_shared, d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // Benchmark coarsened 1D stencil
    dim3 coarse_grid((n + block_size.x * COARSENING_FACTOR - 1) / (block_size.x * COARSENING_FACTOR));
    int coarse_shared_size = (block_size.x * COARSENING_FACTOR + 2 * RADIUS) * sizeof(float);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_1d_coarsened, coarse_grid, block_size, coarse_shared_size, 0, d_input, d_output, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double coarsened_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_coarsened, d_output, n * sizeof(float), hipMemcpyDeviceToHost));
    
    // CPU reference
    timer.start();
    stencil_1d_cpu_reference(h_input, h_output_cpu, n);
    double cpu_time = timer.stop();
    
    // Verify results
    bool naive_correct = verify_result(h_output_naive, h_output_cpu, n);
    bool shared_correct = verify_result(h_output_shared, h_output_cpu, n);
    bool coarsened_correct = verify_result(h_output_coarsened, h_output_cpu, n);
    
    // Print results
    printf("Array size: %d\n", n);
    printf("Naive GPU:      %.3f ms (%s)\n", naive_time, naive_correct ? "PASS" : "FAIL");
    printf("Shared GPU:     %.3f ms (%s), Speedup: %.2fx\n", shared_time, 
           shared_correct ? "PASS" : "FAIL", naive_time / shared_time);
    printf("Coarsened GPU:  %.3f ms (%s), Speedup: %.2fx\n", coarsened_time, 
           coarsened_correct ? "PASS" : "FAIL", naive_time / coarsened_time);
    printf("CPU Reference:  %.3f ms, GPU Speedup: %.2fx (best)\n", 
           cpu_time, cpu_time / fmin(fmin(naive_time, shared_time), coarsened_time));
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_shared;
    delete[] h_output_coarsened;
    delete[] h_output_cpu;
    HIP_CHECK(HIP_CHECK(hipFree(d_input));
    HIP_CHECK(HIP_CHECK(hipFree(d_output));
}

/**
 * Benchmark 2D stencil algorithms
 */
void benchmark_stencil_2d() {
    printf("=== 2D Stencil Benchmark ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    const int num_iterations = 100;
    
    // Allocate host memory
    float *h_input = new float[size];
    float *h_output_naive = new float[size];
    float *h_output_shared = new float[size];
    float *h_output_register = new float[size];
    float *h_output_cpu = new float[size];
    
    // Initialize data
    initialize_2d_data(h_input, width, height);
    
    // Allocate device memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_input, h_input, size * sizeof(float), hipMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive 2D stencil
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_2d_naive, grid_size, block_size, 0, 0, d_input, d_output, width, height);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_naive, d_output, size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Benchmark shared memory 2D stencil
    int shared_mem_size = (block_size.x + 2 * RADIUS) * (block_size.y + 2 * RADIUS) * sizeof(float);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_2d_shared, grid_size, block_size, shared_mem_size, 0, d_input, d_output, width, height);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double shared_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_shared, d_output, size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Benchmark register blocked 2D stencil
    dim3 reg_block_size(BLOCK_SIZE + 2 * RADIUS, BLOCK_SIZE + 2 * RADIUS);
    dim3 reg_grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int reg_shared_size = reg_block_size.x * reg_block_size.y * sizeof(float);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_2d_register_blocked, reg_grid_size, reg_block_size, reg_shared_size, 0, 
                          d_input, d_output, width, height);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double register_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_register, d_output, size * sizeof(float), hipMemcpyDeviceToHost));
    
    // CPU reference
    timer.start();
    stencil_2d_cpu_reference(h_input, h_output_cpu, width, height);
    double cpu_time = timer.stop();
    
    // Verify results
    bool naive_correct = verify_result(h_output_naive, h_output_cpu, size);
    bool shared_correct = verify_result(h_output_shared, h_output_cpu, size);
    bool register_correct = verify_result(h_output_register, h_output_cpu, size);
    
    // Print results
    printf("Grid size: %dx%d\n", width, height);
    printf("Naive GPU:      %.3f ms (%s)\n", naive_time, naive_correct ? "PASS" : "FAIL");
    printf("Shared GPU:     %.3f ms (%s), Speedup: %.2fx\n", shared_time, 
           shared_correct ? "PASS" : "FAIL", naive_time / shared_time);
    printf("Register GPU:   %.3f ms (%s), Speedup: %.2fx\n", register_time, 
           register_correct ? "PASS" : "FAIL", naive_time / register_time);
    printf("CPU Reference:  %.3f ms, GPU Speedup: %.2fx (best)\n", 
           cpu_time, cpu_time / fmin(fmin(naive_time, shared_time), register_time));
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_shared;
    delete[] h_output_register;
    delete[] h_output_cpu;
    HIP_CHECK(HIP_CHECK(hipFree(d_input));
    HIP_CHECK(HIP_CHECK(hipFree(d_output));
}

/**
 * Benchmark 3D stencil algorithms
 */
void benchmark_stencil_3d() {
    printf("=== 3D Stencil Benchmark ===\n");
    
    const int width = 128;
    const int height = 128;
    const int depth = 128;
    const int size = width * height * depth;
    const int num_iterations = 10;
    
    // Allocate host memory
    float *h_input = new float[size];
    float *h_output_naive = new float[size];
    
    // Initialize data
    initialize_3d_data(h_input, width, height, depth);
    
    // Allocate device memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_input, h_input, size * sizeof(float), hipMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive 3D stencil
    dim3 block_size(8, 8, 8);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y,
                   (depth + block_size.z - 1) / block_size.z);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(stencil_3d_naive, grid_size, block_size, 0, 0, d_input, d_output, width, height, depth);
    }
    HIP_CHECK(hipDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    HIP_CHECK(hipMemcpy(h_output_naive, d_output, size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Print results (verification skipped for 3D due to complexity)
    printf("Grid size: %dx%dx%d\n", width, height, depth);
    printf("Naive GPU:      %.3f ms\n", naive_time);
    printf("Note: 3D stencils are memory bandwidth limited\n");
    printf("AMD GPUs: Optimized for HBM memory hierarchy\n");
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    HIP_CHECK(HIP_CHECK(hipFree(d_input));
    HIP_CHECK(HIP_CHECK(hipFree(d_output));
}

/**
 * Main function
 */
int main() {
    printf("Module 6: Stencil Computations (HIP Implementation)\n");
    printf("===================================================\n\n");
    
    // Initialize HIP
    HIP_CHECK(hipSetDevice(0));
    
    // Print device information
    print_gpu_info();
    
    // Run benchmarks
    benchmark_stencil_1d();
    benchmark_stencil_2d();
    benchmark_stencil_3d();
    
    printf("Stencil computation benchmarks completed successfully!\n");
    return 0;
}