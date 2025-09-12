/**
 * Module 6: Fundamental Parallel Algorithms
 * Example 1: Convolution Algorithms (CUDA Implementation)
 * 
 * This example demonstrates various convolution implementations:
 * 1. 1D Convolution for signal processing
 * 2. 2D Convolution for image processing
 * 3. Separable Convolution optimization
 * 4. Shared memory optimization techniques
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
const int BLOCK_SIZE = 16;
const int TILE_SIZE = 16;
const int MAX_KERNEL_SIZE = 31;

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
// 1D Convolution Kernels
// =============================================================================

/**
 * Naive 1D convolution kernel
 */
__global__ void conv1d_naive(float *input, float *kernel, float *output, 
                           int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = idx + k - half_kernel;
            
            // Handle boundary conditions with zero padding
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * kernel[k];
            }
        }
        
        output[idx] = sum;
    }
}

/**
 * Optimized 1D convolution with shared memory
 */
__global__ void conv1d_shared(float *input, float *kernel, float *output,
                            int input_size, int kernel_size) {
    extern __shared__ float shared_mem[];
    float *shared_input = shared_mem;
    float *shared_kernel = shared_input + blockDim.x + kernel_size - 1;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_kernel = kernel_size / 2;
    
    // Load kernel into shared memory (first few threads)
    if (tid < kernel_size) {
        shared_kernel[tid] = kernel[tid];
    }
    
    // Load input data into shared memory with halo regions
    int shared_idx = tid + half_kernel;
    
    // Load center data
    if (idx < input_size) {
        shared_input[shared_idx] = input[idx];
    } else {
        shared_input[shared_idx] = 0.0f;
    }
    
    // Load left halo
    if (tid < half_kernel) {
        int left_idx = blockIdx.x * blockDim.x - half_kernel + tid;
        shared_input[tid] = (left_idx >= 0) ? input[left_idx] : 0.0f;
    }
    
    // Load right halo
    if (tid < half_kernel) {
        int right_idx = (blockIdx.x + 1) * blockDim.x + tid;
        int shared_right_idx = blockDim.x + half_kernel + tid;
        shared_input[shared_right_idx] = (right_idx < input_size) ? input[right_idx] : 0.0f;
    }
    
    __syncthreads();
    
    // Perform convolution
    if (idx < input_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += shared_input[shared_idx - half_kernel + k] * shared_kernel[k];
        }
        output[idx] = sum;
    }
}

// =============================================================================
// 2D Convolution Kernels
// =============================================================================

/**
 * Naive 2D convolution kernel
 */
__global__ void conv2d_naive(float *input, float *kernel, float *output,
                           int width, int height, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int input_row = row + ky - half_kernel;
                int input_col = col + kx - half_kernel;
                
                // Handle boundary conditions with zero padding
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width) {
                    int input_idx = input_row * width + input_col;
                    int kernel_idx = ky * kernel_size + kx;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        
        int output_idx = row * width + col;
        output[output_idx] = sum;
    }
}

/**
 * Optimized 2D convolution with shared memory tiling
 */
__global__ void conv2d_shared(float *input, float *kernel, float *output,
                            int width, int height, int kernel_size) {
    extern __shared__ float shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    int half_kernel = kernel_size / 2;
    int shared_width = TILE_SIZE + kernel_size - 1;
    
    // Load input data into shared memory with halo
    for (int dy = 0; dy < shared_width; dy += TILE_SIZE) {
        for (int dx = 0; dx < shared_width; dx += TILE_SIZE) {
            int shared_row = ty + dy;
            int shared_col = tx + dx;
            
            if (shared_row < shared_width && shared_col < shared_width) {
                int input_row = blockIdx.y * TILE_SIZE + shared_row - half_kernel;
                int input_col = blockIdx.x * TILE_SIZE + shared_col - half_kernel;
                
                int shared_idx = shared_row * shared_width + shared_col;
                
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width) {
                    int input_idx = input_row * width + input_col;
                    shared_input[shared_idx] = input[input_idx];
                } else {
                    shared_input[shared_idx] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Perform convolution
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int shared_row = ty + half_kernel + ky - half_kernel;
                int shared_col = tx + half_kernel + kx - half_kernel;
                int shared_idx = shared_row * shared_width + shared_col;
                int kernel_idx = ky * kernel_size + kx;
                
                sum += shared_input[shared_idx] * kernel[kernel_idx];
            }
        }
        
        int output_idx = row * width + col;
        output[output_idx] = sum;
    }
}

// =============================================================================
// Separable Convolution Kernels
// =============================================================================

/**
 * Separable convolution - horizontal pass
 */
__global__ void conv_separable_horizontal(float *input, float *kernel, float *output,
                                        int width, int height, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int kx = 0; kx < kernel_size; kx++) {
            int input_col = col + kx - half_kernel;
            
            if (input_col >= 0 && input_col < width) {
                int input_idx = row * width + input_col;
                sum += input[input_idx] * kernel[kx];
            }
        }
        
        int output_idx = row * width + col;
        output[output_idx] = sum;
    }
}

/**
 * Separable convolution - vertical pass
 */
__global__ void conv_separable_vertical(float *input, float *kernel, float *output,
                                       int width, int height, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            int input_row = row + ky - half_kernel;
            
            if (input_row >= 0 && input_row < height) {
                int input_idx = input_row * width + col;
                sum += input[input_idx] * kernel[ky];
            }
        }
        
        int output_idx = row * width + col;
        output[output_idx] = sum;
    }
}

// =============================================================================
// Host Functions
// =============================================================================

/**
 * Initialize test data
 */
void initialize_data(float *data, int size, bool random = true) {
    for (int i = 0; i < size; i++) {
        if (random) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        } else {
            data[i] = 1.0f; // Unit impulse for testing
        }
    }
}

/**
 * Create Gaussian kernel
 */
void create_gaussian_kernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        int x = i - half;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

/**
 * Verify convolution results
 */
bool verify_result(float *gpu_result, float *cpu_reference, int size, float tolerance = 1e-5f) {
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
 * CPU reference implementation for 1D convolution
 */
void conv1d_cpu_reference(float *input, float *kernel, float *output,
                         int input_size, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    for (int i = 0; i < input_size; i++) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = i + k - half_kernel;
            
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * kernel[k];
            }
        }
        
        output[i] = sum;
    }
}

/**
 * CPU reference implementation for 2D convolution
 */
void conv2d_cpu_reference(float *input, float *kernel, float *output,
                         int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int input_row = row + ky - half_kernel;
                    int input_col = col + kx - half_kernel;
                    
                    if (input_row >= 0 && input_row < height && 
                        input_col >= 0 && input_col < width) {
                        int input_idx = input_row * width + input_col;
                        int kernel_idx = ky * kernel_size + kx;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
            
            int output_idx = row * width + col;
            output[output_idx] = sum;
        }
    }
}

/**
 * Benchmark 1D convolution algorithms
 */
void benchmark_conv1d() {
    printf("=== 1D Convolution Benchmark ===\n");
    
    const int input_size = 1024 * 1024;
    const int kernel_size = 15;
    const int num_iterations = 100;
    
    // Allocate host memory
    float *h_input = new float[input_size];
    float *h_kernel = new float[kernel_size];
    float *h_output_naive = new float[input_size];
    float *h_output_shared = new float[input_size];
    float *h_output_cpu = new float[input_size];
    
    // Initialize data
    initialize_data(h_input, input_size);
    create_gaussian_kernel(h_kernel, kernel_size, 2.0f);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive 1D convolution
    dim3 block_size(256);
    dim3 grid_size((input_size + block_size.x - 1) / block_size.x);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        conv1d_naive<<<grid_size, block_size>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Benchmark shared memory 1D convolution
    int shared_mem_size = (block_size.x + kernel_size - 1 + kernel_size) * sizeof(float);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        conv1d_shared<<<grid_size, block_size, shared_mem_size>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double shared_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference
    timer.start();
    conv1d_cpu_reference(h_input, h_kernel, h_output_cpu, input_size, kernel_size);
    double cpu_time = timer.stop();
    
    // Verify results
    bool naive_correct = verify_result(h_output_naive, h_output_cpu, input_size);
    bool shared_correct = verify_result(h_output_shared, h_output_cpu, input_size);
    
    // Print results
    printf("Input size: %d, Kernel size: %d\n", input_size, kernel_size);
    printf("Naive GPU:     %.3f ms (%s)\n", naive_time, naive_correct ? "PASS" : "FAIL");
    printf("Shared GPU:    %.3f ms (%s), Speedup: %.2fx\n", shared_time, 
           shared_correct ? "PASS" : "FAIL", naive_time / shared_time);
    printf("CPU Reference: %.3f ms, GPU Speedup: %.2fx (naive), %.2fx (shared)\n", 
           cpu_time, cpu_time / naive_time, cpu_time / shared_time);
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output_naive;
    delete[] h_output_shared;
    delete[] h_output_cpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}

/**
 * Benchmark 2D convolution algorithms
 */
void benchmark_conv2d() {
    printf("=== 2D Convolution Benchmark ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int kernel_size = 15;
    const int num_iterations = 10;
    const int image_size = width * height;
    const int kernel_area = kernel_size * kernel_size;
    
    // Allocate host memory
    float *h_input = new float[image_size];
    float *h_kernel = new float[kernel_area];
    float *h_output_naive = new float[image_size];
    float *h_output_shared = new float[image_size];
    float *h_output_cpu = new float[image_size];
    
    // Initialize data
    initialize_data(h_input, image_size);
    initialize_data(h_kernel, kernel_area);
    
    // Normalize kernel
    float kernel_sum = 0.0f;
    for (int i = 0; i < kernel_area; i++) {
        kernel_sum += h_kernel[i];
    }
    for (int i = 0; i < kernel_area; i++) {
        h_kernel[i] /= kernel_sum;
    }
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_area * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, image_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_area * sizeof(float), cudaMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark naive 2D convolution
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        conv2d_naive<<<grid_size, block_size>>>(d_input, d_kernel, d_output, width, height, kernel_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double naive_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Benchmark shared memory 2D convolution
    int shared_size = TILE_SIZE + kernel_size - 1;
    int shared_mem_size = shared_size * shared_size * sizeof(float);
    
    dim3 tile_grid((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        conv2d_shared<<<tile_grid, block_size, shared_mem_size>>>(d_input, d_kernel, d_output, width, height, kernel_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double shared_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference
    timer.start();
    conv2d_cpu_reference(h_input, h_kernel, h_output_cpu, width, height, kernel_size);
    double cpu_time = timer.stop();
    
    // Verify results
    bool naive_correct = verify_result(h_output_naive, h_output_cpu, image_size);
    bool shared_correct = verify_result(h_output_shared, h_output_cpu, image_size);
    
    // Print results
    printf("Image size: %dx%d, Kernel size: %dx%d\n", width, height, kernel_size, kernel_size);
    printf("Naive GPU:     %.3f ms (%s)\n", naive_time, naive_correct ? "PASS" : "FAIL");
    printf("Shared GPU:    %.3f ms (%s), Speedup: %.2fx\n", shared_time, 
           shared_correct ? "PASS" : "FAIL", naive_time / shared_time);
    printf("CPU Reference: %.3f ms, GPU Speedup: %.2fx (naive), %.2fx (shared)\n", 
           cpu_time, cpu_time / naive_time, cpu_time / shared_time);
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output_naive;
    delete[] h_output_shared;
    delete[] h_output_cpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}

/**
 * Benchmark separable convolution
 */
void benchmark_separable_conv() {
    printf("=== Separable Convolution Benchmark ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int kernel_size = 15;
    const int num_iterations = 10;
    const int image_size = width * height;
    
    // Allocate host memory
    float *h_input = new float[image_size];
    float *h_kernel_1d = new float[kernel_size];
    float *h_output_separable = new float[image_size];
    float *h_output_cpu = new float[image_size];
    float *h_temp = new float[image_size];
    
    // Initialize data
    initialize_data(h_input, image_size);
    create_gaussian_kernel(h_kernel_1d, kernel_size, 2.0f);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, image_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel_1d, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    Timer timer;
    
    // Benchmark separable convolution
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        // Horizontal pass
        conv_separable_horizontal<<<grid_size, block_size>>>(d_input, d_kernel, d_temp, width, height, kernel_size);
        // Vertical pass
        conv_separable_vertical<<<grid_size, block_size>>>(d_temp, d_kernel, d_output, width, height, kernel_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double separable_time = timer.stop() / num_iterations;
    
    CUDA_CHECK(cudaMemcpy(h_output_separable, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference for separable convolution
    timer.start();
    // Horizontal pass
    for (int row = 0; row < height; row++) {
        conv1d_cpu_reference(&h_input[row * width], h_kernel_1d, &h_temp[row * width], width, kernel_size);
    }
    // Vertical pass
    for (int col = 0; col < width; col++) {
        float col_data[height], col_result[height];
        for (int row = 0; row < height; row++) {
            col_data[row] = h_temp[row * width + col];
        }
        conv1d_cpu_reference(col_data, h_kernel_1d, col_result, height, kernel_size);
        for (int row = 0; row < height; row++) {
            h_output_cpu[row * width + col] = col_result[row];
        }
    }
    double cpu_time = timer.stop();
    
    // Verify results
    bool separable_correct = verify_result(h_output_separable, h_output_cpu, image_size, 1e-4f);
    
    // Print results
    printf("Image size: %dx%d, Kernel size: %d (separable)\n", width, height, kernel_size);
    printf("Separable GPU: %.3f ms (%s)\n", separable_time, separable_correct ? "PASS" : "FAIL");
    printf("CPU Reference: %.3f ms, GPU Speedup: %.2fx\n", 
           cpu_time, cpu_time / separable_time);
    printf("\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_kernel_1d;
    delete[] h_output_separable;
    delete[] h_output_cpu;
    delete[] h_temp;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
}

/**
 * Main function
 */
int main() {
    printf("Module 6: Convolution Algorithms (CUDA Implementation)\n");
    printf("=====================================================\n\n");
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device information
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
    
    // Run benchmarks
    benchmark_conv1d();
    benchmark_conv2d();
    benchmark_separable_conv();
    
    printf("Convolution benchmarks completed successfully!\n");
    return 0;
}