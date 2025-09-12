#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel - runs on GPU
__global__ void addVectors(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Boundary check to prevent out-of-bounds access
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Error checking macro
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
    const int N = 1024;  // Vector size
    const int bytes = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel: 4 blocks of 256 threads each
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
    
    printf("Launching kernel with %d blocks of %d threads each\n", gridSize, blockSize);
    
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify result (first few elements)
    printf("Verification (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.3f + %.3f = %.3f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Clean up memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    printf("Vector addition completed successfully!\n");
    return 0;
}