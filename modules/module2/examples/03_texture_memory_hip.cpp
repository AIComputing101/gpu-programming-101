#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// HIP texture object approach
__global__ void textureFilterKernel(hipTextureObject_t texObj, float *output, 
                                   int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        // Apply filter using texture memory
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                // Normalize coordinates to [0,1] range
                float u = (float)(x + fx + 0.5f) / width;
                float v = (float)(y + fy + 0.5f) / height;
                
                // Texture automatically handles boundary conditions and interpolation
                float value = hipTex2D<float>(texObj, u, v);
                sum += value;
            }
        }
        
        output[y * width + x] = sum / (filter_size * filter_size);
    }
}

// Texture-based matrix transpose with spatial locality
__global__ void textureTranspose(hipTextureObject_t texObj, float *output, 
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Normalized coordinates
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        // Fetch using texture cache
        float value = hipTex2D<float>(texObj, u, v);
        
        // Write transposed
        if (y < width && x < height) {
            output[x * height + y] = value;
        }
    }
}

// Bilinear interpolation example
__global__ void bilinearInterpolation(hipTextureObject_t texObj, float *output, 
                                     int out_width, int out_height, 
                                     int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        // Map output coordinates to input coordinates
        float scale_x = (float)in_width / out_width;
        float scale_y = (float)in_height / out_height;
        
        float src_x = (x + 0.5f) * scale_x;
        float src_y = (y + 0.5f) * scale_y;
        
        // Normalize coordinates
        float u = src_x / in_width;
        float v = src_y / in_height;
        
        // Hardware bilinear interpolation
        float interpolated = hipTex2D<float>(texObj, u, v);
        
        output[y * out_width + x] = interpolated;
    }
}

// Manual bilinear interpolation for platforms without hardware support
__global__ void manualBilinearInterpolation(const float *input, float *output,
                                           int out_width, int out_height,
                                           int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        float scale_x = (float)in_width / out_width;
        float scale_y = (float)in_height / out_height;
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        int x1 = (int)floorf(src_x);
        int y1 = (int)floorf(src_y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        
        // Clamp coordinates
        x1 = max(0, min(x1, in_width - 1));
        y1 = max(0, min(y1, in_height - 1));
        x2 = max(0, min(x2, in_width - 1));
        y2 = max(0, min(y2, in_height - 1));
        
        float fx = src_x - x1;
        float fy = src_y - y1;
        
        // Bilinear interpolation
        float v1 = input[y1 * in_width + x1] * (1.0f - fx) + input[y1 * in_width + x2] * fx;
        float v2 = input[y2 * in_width + x1] * (1.0f - fx) + input[y2 * in_width + x2] * fx;
        float result = v1 * (1.0f - fy) + v2 * fy;
        
        output[y * out_width + x] = result;
    }
}

// AMD GPU optimized texture access pattern
__global__ void amdOptimizedTextureAccess(hipTextureObject_t texObj, float *output,
                                         int width, int height) {
    // AMD wavefront-aware texture access
    int wavefront_id = blockIdx.x * blockDim.x / 64 + threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    
    int total_wavefronts = gridDim.x * blockDim.x / 64;
    int pixels_per_wavefront = (width * height + total_wavefronts - 1) / total_wavefronts;
    
    for (int i = 0; i < pixels_per_wavefront; i++) {
        int pixel_id = wavefront_id * pixels_per_wavefront + i;
        if (pixel_id < width * height) {
            int x = pixel_id % width;
            int y = pixel_id / width;
            
            // Coalesced texture access within wavefront
            float u = (x + 0.5f) / width;
            float v = (y + 0.5f) / height;
            
            float value = hipTex2D<float>(texObj, u, v);
            output[pixel_id] = value;
        }
    }
}

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

hipTextureObject_t createTextureObject(float *d_data, int width, int height) {
    // Create resource descriptor
    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = hipResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);
    
    // Create texture descriptor
    hipTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.filterMode = hipFilterModeLinear;
    texDesc.readMode = hipReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    // Create texture object
    hipTextureObject_t texObj;
    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    
    return texObj;
}

void demonstrateTextureMemory() {
    printf("=== HIP Texture Memory Demo ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    const int filter_size = 3;
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output_texture = (float*)malloc(size);
    float *h_output_manual = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < width * height; i++) {
        h_input[i] = sinf(i * 0.01f) + cosf(i * 0.02f);
    }
    
    // Allocate device memory
    float *d_input, *d_output_texture, *d_output_manual;
    HIP_CHECK(hipMalloc(&d_input, size));
    HIP_CHECK(hipMalloc(&d_output_texture, size));
    HIP_CHECK(hipMalloc(&d_output_manual, size));
    
    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
    
    // Create texture object
    hipTextureObject_t texObj = createTextureObject(d_input, width, height);
    
    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Test 1: Texture-based filtering
    printf("Testing texture-based filtering...\n");
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(textureFilterKernel, gridSize, blockSize, 0, 0,
                       texObj, d_output_texture, width, height, filter_size);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float texture_time;
    HIP_CHECK(hipEventElapsedTime(&texture_time, start, stop));
    printf("Texture filtering time: %.3f ms\n", texture_time);
    
    // Test 2: Manual bilinear interpolation
    printf("Testing manual interpolation...\n");
    
    int out_width = 512, out_height = 512;
    float *d_resized;
    HIP_CHECK(hipMalloc(&d_resized, out_width * out_height * sizeof(float)));
    
    dim3 resizeBlockSize(16, 16);
    dim3 resizeGridSize((out_width + resizeBlockSize.x - 1) / resizeBlockSize.x,
                        (out_height + resizeBlockSize.y - 1) / resizeBlockSize.y);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(bilinearInterpolation, resizeGridSize, resizeBlockSize, 0, 0,
                       texObj, d_resized, out_width, out_height, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float resize_time;
    HIP_CHECK(hipEventElapsedTime(&resize_time, start, stop));
    printf("Texture resize time: %.3f ms\n", resize_time);
    
    // Test 3: AMD optimized access pattern
    printf("Testing AMD optimized texture access...\n");
    
    dim3 amdBlockSize(256);
    dim3 amdGridSize((width * height + amdBlockSize.x - 1) / amdBlockSize.x);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(amdOptimizedTextureAccess, amdGridSize, amdBlockSize, 0, 0,
                       texObj, d_output_manual, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float amd_time;
    HIP_CHECK(hipEventElapsedTime(&amd_time, start, stop));
    printf("AMD optimized access time: %.3f ms\n", amd_time);
    
    // Verify results
    HIP_CHECK(hipMemcpy(h_output_texture, d_output_texture, size, hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    float bandwidth_gb_s = (2.0f * size) / (texture_time * 1e6); // Read + Write
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    // Texture cache hit rate analysis
    printf("\n=== Texture Memory Analysis ===\n");
    printf("Texture memory provides:\n");
    printf("- Automatic boundary handling\n");
    printf("- Hardware interpolation\n");
    printf("- Cached access for spatial locality\n");
    printf("- Normalized coordinate addressing\n");
    
#ifdef __HIP_PLATFORM_AMD__
    printf("\nAMD GPU specific optimizations:\n");
    printf("- Wavefront-aware texture access patterns\n");
    printf("- Optimized for 64-thread wavefronts\n");
    printf("- Memory coalescing for texture cache\n");
#endif
    
    // Cleanup
    HIP_CHECK(hipDestroyTextureObject(texObj));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output_texture));
    HIP_CHECK(hipFree(d_output_manual));
    HIP_CHECK(hipFree(d_resized));
    
    free(h_input);
    free(h_output_texture);
    free(h_output_manual);
}

int main() {
    printf("HIP Texture Memory Example\n");
    printf("=========================\n");
    
    demonstrateTextureMemory();
    
    return 0;
}