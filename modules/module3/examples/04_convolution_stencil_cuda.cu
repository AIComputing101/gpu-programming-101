#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RADIUS 3
#define BLOCK_SIZE 16

// 1D Stencil with shared memory
__global__ void stencil1D(float *input, float *output, int n) {
    __shared__ float shared[BLOCK_SIZE + 2 * RADIUS];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load main data
    int shared_idx = tid + RADIUS;
    if (gid < n) {
        shared[shared_idx] = input[gid];
    } else {
        shared[shared_idx] = 0.0f;
    }
    
    // Load left halo
    if (tid < RADIUS) {
        int left_idx = gid - RADIUS;
        shared[tid] = (left_idx >= 0) ? input[left_idx] : 0.0f;
    }
    
    // Load right halo
    if (tid < RADIUS) {
        int right_idx = gid + BLOCK_SIZE;
        shared[shared_idx + BLOCK_SIZE] = (right_idx < n) ? input[right_idx] : 0.0f;
    }
    
    __syncthreads();
    
    // Apply stencil (simple averaging)
    if (gid < n) {
        float result = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++) {
            result += shared[shared_idx + i];
        }
        output[gid] = result / (2 * RADIUS + 1);
    }
}

// 2D Convolution
__global__ void convolution2D(float *input, float *output, float *kernel,
                             int width, int height, int kernel_size) {
    __shared__ float shared_input[BLOCK_SIZE + 2*RADIUS][BLOCK_SIZE + 2*RADIUS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int shared_x = tx + RADIUS;
    int shared_y = ty + RADIUS;
    
    // Load main data
    if (row < height && col < width) {
        shared_input[shared_y][shared_x] = input[row * width + col];
    } else {
        shared_input[shared_y][shared_x] = 0.0f;
    }
    
    // Load halos
    if (tx < RADIUS) {
        // Left halo
        int left_col = col - RADIUS;
        shared_input[shared_y][tx] = (left_col >= 0 && row < height) ? 
                                    input[row * width + left_col] : 0.0f;
        
        // Right halo
        int right_col = col + BLOCK_SIZE;
        shared_input[shared_y][shared_x + BLOCK_SIZE] = 
            (right_col < width && row < height) ? input[row * width + right_col] : 0.0f;
    }
    
    if (ty < RADIUS) {
        // Top halo
        int top_row = row - RADIUS;
        shared_input[ty][shared_x] = (top_row >= 0 && col < width) ? 
                                    input[top_row * width + col] : 0.0f;
        
        // Bottom halo
        int bottom_row = row + BLOCK_SIZE;
        shared_input[shared_y + BLOCK_SIZE][shared_x] = 
            (bottom_row < height && col < width) ? input[bottom_row * width + col] : 0.0f;
    }
    
    __syncthreads();
    
    // Apply convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                sum += shared_input[shared_y + ky][shared_x + kx] * kernel[kernel_idx];
            }
        }
        
        output[row * width + col] = sum;
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

int main() {
    printf("CUDA Convolution and Stencil Examples\n");
    printf("====================================\n");
    
    // 1D Stencil test
    const int N = 1024;
    float *h_input1d = (float*)malloc(N * sizeof(float));
    float *h_output1d = (float*)malloc(N * sizeof(float));
    
    // Initialize 1D data
    for (int i = 0; i < N; i++) {
        h_input1d[i] = sin(i * 0.1f);
    }
    
    float *d_input1d, *d_output1d;
    CUDA_CHECK(cudaMalloc(&d_input1d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output1d, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input1d, h_input1d, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block1d(BLOCK_SIZE);
    dim3 grid1d((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    stencil1D<<<grid1d, block1d>>>(d_input1d, d_output1d, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output1d, d_output1d, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("1D Stencil completed\n");
    printf("Sample input:  %.3f %.3f %.3f %.3f %.3f\n", 
           h_input1d[10], h_input1d[11], h_input1d[12], h_input1d[13], h_input1d[14]);
    printf("Sample output: %.3f %.3f %.3f %.3f %.3f\n", 
           h_output1d[10], h_output1d[11], h_output1d[12], h_output1d[13], h_output1d[14]);
    
    // 2D Convolution test
    const int width = 512, height = 512;
    const int size2d = width * height * sizeof(float);
    const int kernel_size = 3;
    
    float *h_input2d = (float*)malloc(size2d);
    float *h_output2d = (float*)malloc(size2d);
    float h_kernel[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; // Edge detection
    
    // Initialize 2D data
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input2d[i * width + j] = (i + j) % 2; // Checkerboard pattern
        }
    }
    
    float *d_input2d, *d_output2d, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input2d, size2d));
    CUDA_CHECK(cudaMalloc(&d_output2d, size2d));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input2d, h_input2d, size2d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    dim3 block2d(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid2d((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    convolution2D<<<grid2d, block2d>>>(d_input2d, d_output2d, d_kernel, width, height, kernel_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n2D Convolution completed\n");
    printf("Image size: %dx%d\n", width, height);
    printf("Kernel: Edge detection (3x3)\n");
    
    // Cleanup
    free(h_input1d); free(h_output1d);
    free(h_input2d); free(h_output2d);
    cudaFree(d_input1d); cudaFree(d_output1d);
    cudaFree(d_input2d); cudaFree(d_output2d); cudaFree(d_kernel);
    
    printf("\nConvolution and stencil examples completed!\n");
    return 0;
}