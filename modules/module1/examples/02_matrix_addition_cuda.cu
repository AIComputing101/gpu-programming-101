#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {
    // 2D thread indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (row < height && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
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
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Define block and grid sizes
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Matrix dimensions: %d x %d\n", width, height);
    printf("Grid size: (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block size: (%d, %d)\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify result
    bool success = true;
    for (int i = 0; i < width * height; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            printf("Error at element %d: expected 3.0, got %f\n", i, h_C[i]);
            break;
        }
    }
    
    printf("Matrix addition %s\n", success ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}