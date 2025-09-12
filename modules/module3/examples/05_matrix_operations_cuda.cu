#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Matrix multiplication with shared memory tiling
__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;
    
    float Cvalue = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (Row < N && t * TILE_SIZE + tx < N)
            tileA[ty][tx] = A[Row * N + t * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0.0f;
            
        if (t * TILE_SIZE + ty < N && Col < N)
            tileB[ty][tx] = B[(t * TILE_SIZE + ty) * N + Col];
        else
            tileB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// Matrix transpose with shared memory to avoid bank conflicts
__global__ void transposeSharedMem(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int xIndex = blockIdx.x * TILE_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = input[yIndex * width + xIndex];
    }
    
    __syncthreads();
    
    // Write transposed data to output
    xIndex = blockIdx.y * TILE_SIZE + threadIdx.x;
    yIndex = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (xIndex < height && yIndex < width) {
        output[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

// Matrix-vector multiplication using cooperative groups
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void matrixVectorMul(float *matrix, float *vector, float *result, int N) {
    auto block = cg::this_thread_block();
    __shared__ float shared_vector[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    
    // Process matrix row in chunks
    for (int chunk = 0; chunk < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++chunk) {
        int idx = chunk * BLOCK_SIZE + tid;
        
        // Load vector chunk into shared memory
        if (idx < N) {
            shared_vector[tid] = vector[idx];
        } else {
            shared_vector[tid] = 0.0f;
        }
        
        block.sync();
        
        // Compute partial dot product
        for (int i = 0; i < BLOCK_SIZE && chunk * BLOCK_SIZE + i < N; ++i) {
            if (row < N) {
                sum += matrix[row * N + chunk * BLOCK_SIZE + i] * shared_vector[i];
            }
        }
        
        block.sync();
    }
    
    if (tid == 0 && row < N) {
        result[row] = sum;
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void initMatrix(float *matrix, int rows, int cols, bool random = true) {
    for (int i = 0; i < rows * cols; i++) {
        if (random) {
            matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        } else {
            matrix[i] = (float)i / (rows * cols);
        }
    }
}

void printMatrix(float *matrix, int rows, int cols, const char* name, int maxShow = 4) {
    printf("%s (%dx%d):\n", name, rows, cols);
    int showRows = (rows < maxShow) ? rows : maxShow;
    int showCols = (cols < maxShow) ? cols : maxShow;
    
    for (int i = 0; i < showRows; i++) {
        for (int j = 0; j < showCols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        if (cols > maxShow) printf("...");
        printf("\n");
    }
    if (rows > maxShow) printf("...\n");
    printf("\n");
}

int main() {
    printf("CUDA Advanced Matrix Operations\n");
    printf("==============================\n\n");
    
    const int N = 512;  // Matrix size NxN
    const int bytes = N * N * sizeof(float);
    
    // Host memory allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_At = (float*)malloc(bytes);
    float *h_vector = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(N * sizeof(float));
    
    // Initialize matrices
    srand(42);
    initMatrix(h_A, N, N);
    initMatrix(h_B, N, N);
    initMatrix(h_vector, N, 1);
    
    printMatrix(h_A, N, N, "Matrix A (sample)");
    printMatrix(h_B, N, N, "Matrix B (sample)");
    
    // Device memory allocation
    float *d_A, *d_B, *d_C, *d_At, *d_vector, *d_result;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_At, bytes));
    CUDA_CHECK(cudaMalloc(&d_vector, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector, h_vector, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Matrix Multiplication
    printf("1. Matrix Multiplication (Tiled)\n");
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double matmul_time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("   Tiled matrix multiplication time: %.3f ms\n", matmul_time);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (matmul_time * 1e6);
    printf("   Performance: %.2f GFLOPS\n", gflops);
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printMatrix(h_C, N, N, "Result C = A * B (sample)");
    
    // Matrix Transpose
    printf("2. Matrix Transpose (Shared Memory)\n");
    start = std::chrono::high_resolution_clock::now();
    transposeSharedMem<<<dimGrid, dimBlock>>>(d_A, d_At, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double transpose_time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("   Transpose time: %.3f ms\n", transpose_time);
    
    double bandwidth = (2.0 * bytes) / (transpose_time * 1e6); // GB/s
    printf("   Bandwidth: %.2f GB/s\n", bandwidth);
    
    CUDA_CHECK(cudaMemcpy(h_At, d_At, bytes, cudaMemcpyDeviceToHost));
    printMatrix(h_At, N, N, "Transposed A (sample)");
    
    // Matrix-Vector Multiplication
    printf("3. Matrix-Vector Multiplication\n");
    dim3 mvGrid(N);
    dim3 mvBlock(BLOCK_SIZE);
    
    start = std::chrono::high_resolution_clock::now();
    matrixVectorMul<<<mvGrid, mvBlock>>>(d_A, d_vector, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double matvec_time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("   Matrix-vector multiplication time: %.3f ms\n", matvec_time);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("   Vector result (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_result[i]);
    }
    printf("\n\n");
    
    // Verify matrix multiplication result (sample check)
    printf("4. Verification (sample)\n");
    float cpu_result = 0.0f;
    for (int k = 0; k < N; k++) {
        cpu_result += h_A[0 * N + k] * h_B[k * N + 0];
    }
    printf("   CPU C[0,0]: %.6f\n", cpu_result);
    printf("   GPU C[0,0]: %.6f\n", h_C[0]);
    printf("   Difference: %.6f\n", fabs(cpu_result - h_C[0]));
    
    // Performance summary
    printf("Performance Summary:\n");
    printf("  Matrix size: %dx%d\n", N, N);
    printf("  Matrix multiplication: %.2f GFLOPS\n", gflops);
    printf("  Transpose bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Total operations completed successfully!\n");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_At);
    free(h_vector); free(h_result);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_At);
    cudaFree(d_vector); cudaFree(d_result);
    
    printf("\nMatrix operations examples completed!\n");
    return 0;
}