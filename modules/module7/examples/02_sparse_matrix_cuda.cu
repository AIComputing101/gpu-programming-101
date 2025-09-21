#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<float> values;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    
    CSRMatrix(int r, int c, int nz) : rows(r), cols(c), nnz(nz) {
        values.resize(nnz);
        row_ptr.resize(rows + 1);
        col_idx.resize(nnz);
    }
};

CSRMatrix generateRandomSparseMatrix(int rows, int cols, double sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    int target_nnz = static_cast<int>(rows * cols * sparsity);
    CSRMatrix matrix(rows, cols, target_nnz);
    
    int current_nnz = 0;
    matrix.row_ptr[0] = 0;
    
    for (int i = 0; i < rows && current_nnz < target_nnz; ++i) {
        int nnz_this_row = std::min(target_nnz - current_nnz, 
                                   static_cast<int>((target_nnz / rows) * (1.0 + val_dist(gen) * 0.5)));
        
        std::vector<int> cols_in_row;
        for (int j = 0; j < nnz_this_row; ++j) {
            int col;
            do {
                col = col_dist(gen);
            } while (std::find(cols_in_row.begin(), cols_in_row.end(), col) != cols_in_row.end());
            cols_in_row.push_back(col);
        }
        
        std::sort(cols_in_row.begin(), cols_in_row.end());
        
        for (int col : cols_in_row) {
            matrix.values[current_nnz] = val_dist(gen);
            matrix.col_idx[current_nnz] = col;
            current_nnz++;
        }
        
        matrix.row_ptr[i + 1] = current_nnz;
    }
    
    // Fill remaining rows
    for (int i = rows; i >= 0; --i) {
        if (matrix.row_ptr[i] == 0) {
            matrix.row_ptr[i] = current_nnz;
        } else {
            break;
        }
    }
    
    matrix.nnz = current_nnz;
    matrix.values.resize(current_nnz);
    matrix.col_idx.resize(current_nnz);
    
    return matrix;
}

__global__ void spmv_csr_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                const float* x, float* y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        
        for (int j = start; j < end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        
        y[row] = sum;
    }
}

__global__ void spmv_csr_warp_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                     const float* x, float* y, int rows) {
    int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int total_warps = gridDim.x * blockDim.x / 32;
    
    for (int row = warp_id; row < rows; row += total_warps) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        // int nnz_in_row = end - start; // Unused, commented out
        
        float sum = 0.0f;
        
        // Process elements in chunks of 32
        for (int j = start + lane_id; j < end; j += 32) {
            sum += values[j] * x[col_idx[j]];
        }
        
        // Warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void spmv_csr_vector_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                       const float* x, float* y, int rows) {
    __shared__ float sdata[256];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    // int nnz_in_row = end - start; // Unused, commented out
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        sum += values[j] * x[col_idx[j]];
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}

class SparseMatrixMultiplier {
private:
    cusparseHandle_t cusparse_handle;
    cudaStream_t stream;
    
public:
    SparseMatrixMultiplier() {
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUSPARSE(cusparseSetStream(cusparse_handle, stream));
    }
    
    ~SparseMatrixMultiplier() {
        cusparseDestroy(cusparse_handle);
        cudaStreamDestroy(stream);
    }
    
    void spmv_cusparse(const CSRMatrix& matrix, const float* d_x, float* d_y) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        
        // Create sparse matrix descriptor
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, matrix.rows, matrix.cols, matrix.nnz,
                                        (void*)matrix.row_ptr.data(), (void*)matrix.col_idx.data(),
                                        (void*)matrix.values.data(),
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        
        // Create dense vector descriptors
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, matrix.cols, (void*)d_x, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, matrix.rows, (void*)d_y, CUDA_R_32F));
        
        // Compute buffer size
        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                              CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        
        void* dBuffer = nullptr;
        if (bufferSize > 0) {
            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
        }
        
        // Perform SpMV
        CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                   CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Cleanup
        if (dBuffer) cudaFree(dBuffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    }
    
    void spmv_custom(const CSRMatrix& matrix, const float* d_x, float* d_y, int kernel_type = 0) {
        dim3 block_size(256);
        dim3 grid_size;
        
        switch (kernel_type) {
            case 0: // Thread-per-row
                grid_size = dim3((matrix.rows + block_size.x - 1) / block_size.x);
                spmv_csr_kernel<<<grid_size, block_size, 0, stream>>>(
                    matrix.values.data(), matrix.row_ptr.data(), matrix.col_idx.data(),
                    d_x, d_y, matrix.rows);
                break;
                
            case 1: // Warp-per-row
                grid_size = dim3((matrix.rows * 32 + block_size.x - 1) / block_size.x);
                spmv_csr_warp_kernel<<<grid_size, block_size, 0, stream>>>(
                    matrix.values.data(), matrix.row_ptr.data(), matrix.col_idx.data(),
                    d_x, d_y, matrix.rows);
                break;
                
            case 2: // Block-per-row
                grid_size = dim3(matrix.rows);
                spmv_csr_vector_kernel<<<grid_size, block_size, 0, stream>>>(
                    matrix.values.data(), matrix.row_ptr.data(), matrix.col_idx.data(),
                    d_x, d_y, matrix.rows);
                break;
        }
        
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
};

__global__ void sparse_matrix_add_kernel(const float* A_vals, const int* A_row_ptr, const int* A_col_idx,
                                          const float* B_vals, const int* B_row_ptr, const int* B_col_idx,
                                          float* C_vals, int* C_row_ptr, int* C_col_idx,
                                          int rows, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows) return;
    
    int c_start = C_row_ptr[row];
    int a_start = A_row_ptr[row];
    int a_end = A_row_ptr[row + 1];
    int b_start = B_row_ptr[row];
    int b_end = B_row_ptr[row + 1];
    
    int c_idx = c_start;
    int a_idx = a_start;
    int b_idx = b_start;
    
    while (a_idx < a_end && b_idx < b_end) {
        int a_col = A_col_idx[a_idx];
        int b_col = B_col_idx[b_idx];
        
        if (a_col < b_col) {
            C_col_idx[c_idx] = a_col;
            C_vals[c_idx] = alpha * A_vals[a_idx];
            a_idx++;
        } else if (b_col < a_col) {
            C_col_idx[c_idx] = b_col;
            C_vals[c_idx] = beta * B_vals[b_idx];
            b_idx++;
        } else {
            C_col_idx[c_idx] = a_col;
            C_vals[c_idx] = alpha * A_vals[a_idx] + beta * B_vals[b_idx];
            a_idx++;
            b_idx++;
        }
        c_idx++;
    }
    
    while (a_idx < a_end) {
        C_col_idx[c_idx] = A_col_idx[a_idx];
        C_vals[c_idx] = alpha * A_vals[a_idx];
        a_idx++;
        c_idx++;
    }
    
    while (b_idx < b_end) {
        C_col_idx[c_idx] = B_col_idx[b_idx];
        C_vals[c_idx] = beta * B_vals[b_idx];
        b_idx++;
        c_idx++;
    }
}

void demonstrateSparseOperations() {
    std::cout << "\n=== CUDA Sparse Matrix Operations Demo ===" << std::endl;
    
    const int rows = 4096;
    const int cols = 4096;
    const double sparsity = 0.01; // 1% non-zero elements
    
    std::cout << "Generating sparse matrix (" << rows << "x" << cols 
              << ", sparsity=" << sparsity << ")..." << std::endl;
    
    CSRMatrix matrix = generateRandomSparseMatrix(rows, cols, sparsity);
    std::cout << "Generated matrix with " << matrix.nnz << " non-zero elements" << std::endl;
    
    // Allocate device memory
    float *d_values, *d_x, *d_y1, *d_y2;
    int *d_row_ptr, *d_col_idx;
    
    CHECK_CUDA(cudaMalloc(&d_values, matrix.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (matrix.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, matrix.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, matrix.cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y1, matrix.rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2, matrix.rows * sizeof(float)));
    
    // Copy matrix to device
    CHECK_CUDA(cudaMemcpy(d_values, matrix.values.data(), matrix.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, matrix.row_ptr.data(), (matrix.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, matrix.col_idx.data(), matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize input vector
    std::vector<float> x(matrix.cols, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), matrix.cols * sizeof(float), cudaMemcpyHostToDevice));
    
    Timer timer;
    SparseMatrixMultiplier multiplier;
    
    // Test cuSPARSE SpMV
    std::cout << "\n--- cuSPARSE SpMV ---" << std::endl;
    timer.start();
    multiplier.spmv_cusparse(matrix, d_x, d_y1);
    double cusparse_time = timer.elapsed();
    std::cout << "cuSPARSE time: " << cusparse_time << " ms" << std::endl;
    
    // Test custom kernels
    const char* kernel_names[] = {"Thread-per-row", "Warp-per-row", "Block-per-row"};
    
    for (int kernel_type = 0; kernel_type < 3; ++kernel_type) {
        std::cout << "\n--- Custom " << kernel_names[kernel_type] << " ---" << std::endl;
        
        CHECK_CUDA(cudaMemset(d_y2, 0, matrix.rows * sizeof(float)));
        
        timer.start();
        multiplier.spmv_custom(matrix, d_x, d_y2, kernel_type);
        double custom_time = timer.elapsed();
        
        std::cout << "Custom kernel time: " << custom_time << " ms" << std::endl;
        std::cout << "Speedup vs cuSPARSE: " << cusparse_time / custom_time << "x" << std::endl;
        
        // Verify correctness
        std::vector<float> y1(matrix.rows), y2(matrix.rows);
        CHECK_CUDA(cudaMemcpy(y1.data(), d_y1, matrix.rows * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y2.data(), d_y2, matrix.rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        double max_error = 0.0;
        for (int i = 0; i < matrix.rows; ++i) {
            double error = std::abs(y1[i] - y2[i]);
            max_error = std::max(max_error, error);
        }
        std::cout << "Max error vs cuSPARSE: " << max_error << std::endl;
    }
    
    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    double flops = 2.0 * matrix.nnz; // One multiply and one add per non-zero
    double gflops_cusparse = flops / (cusparse_time * 1e6);
    std::cout << "cuSPARSE GFLOPS: " << gflops_cusparse << std::endl;
    
    double bandwidth_cusparse = (matrix.nnz * sizeof(float) + matrix.nnz * sizeof(int) + 
                                matrix.cols * sizeof(float) + matrix.rows * sizeof(float)) / 
                               (cusparse_time * 1e6);
    std::cout << "cuSPARSE Bandwidth: " << bandwidth_cusparse << " GB/s" << std::endl;
    
    // Cleanup
    cudaFree(d_values);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_x);
    cudaFree(d_y1);
    cudaFree(d_y2);
}

void demonstrateAdvancedSparseOperations() {
    std::cout << "\n=== Advanced Sparse Operations ===" << std::endl;
    
    // Demonstrate sparse matrix-matrix multiplication using cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    
    const int rows = 1024;
    const int cols = 1024;
    
    CSRMatrix A = generateRandomSparseMatrix(rows, cols, 0.05);
    CSRMatrix B = generateRandomSparseMatrix(rows, cols, 0.05);
    
    std::cout << "Matrix A: " << A.rows << "x" << A.cols << " with " << A.nnz << " non-zeros" << std::endl;
    std::cout << "Matrix B: " << B.rows << "x" << B.cols << " with " << B.nnz << " non-zeros" << std::endl;
    
    // Allocate device memory for matrices
    float *d_A_vals, *d_B_vals;
    int *d_A_row_ptr, *d_A_col_idx, *d_B_row_ptr, *d_B_col_idx;
    
    CHECK_CUDA(cudaMalloc(&d_A_vals, A.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_row_ptr, (A.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_col_idx, A.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_vals, B.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_row_ptr, (B.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_col_idx, B.nnz * sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A_vals, A.values.data(), A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_col_idx, A.col_idx.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_vals, B.values.data(), B.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_row_ptr, B.row_ptr.data(), (B.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_col_idx, B.col_idx.data(), B.nnz * sizeof(int), cudaMemcpyHostToDevice));
    
    Timer timer;
    timer.start();
    
    // Create matrix descriptors
    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                    d_A_row_ptr, d_A_col_idx, d_A_vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz,
                                    d_B_row_ptr, d_B_col_idx, d_B_vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    // Compute C = A * B
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute buffer sizes
    size_t bufferSize1 = 0; // bufferSize2 unused, removed
    
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, matB, &beta, matC,
                                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize1, nullptr));
    
    void* dBuffer1 = nullptr;
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, matB, &beta, matC,
                                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize1, dBuffer1));
    
    double spgemm_time = timer.elapsed();
    std::cout << "Sparse matrix-matrix multiplication setup time: " << spgemm_time << " ms" << std::endl;
    
    // Cleanup
    if (dBuffer1) cudaFree(dBuffer1);
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroy(handle);
    
    cudaFree(d_A_vals);
    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_idx);
    cudaFree(d_B_vals);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_idx);
}

int main() {
    std::cout << "CUDA Sparse Matrix Operations Demo" << std::endl;
    std::cout << "==================================" << std::endl;
    
    demonstrateSparseOperations();
    demonstrateAdvancedSparseOperations();
    
    return 0;
}