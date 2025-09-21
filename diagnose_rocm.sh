#!/bin/bash

echo "=== GPU Programming 101 - ROCm Diagnosis ==="
echo "Date: $(date)"
echo ""

echo "=== System Information ==="
uname -a
echo ""

echo "=== ROCm Installation ==="
if command -v rocminfo >/dev/null 2>&1; then
    echo "✓ rocminfo available"
    rocminfo | head -20
else
    echo "✗ rocminfo not found"
fi
echo ""

echo "=== GPU Devices ==="
ls -la /dev/kfd /dev/dri/render* 2>/dev/null || echo "No GPU devices found"
echo ""

echo "=== ROCm SMI ==="
if command -v rocm-smi >/dev/null 2>&1; then
    echo "✓ rocm-smi available"
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null || echo "rocm-smi failed"
else
    echo "✗ rocm-smi not found"
fi
echo ""

echo "=== HIP Compiler ==="
hipcc --version
echo ""

echo "=== Environment Variables ==="
echo "ROCM_PATH: $ROCM_PATH"
echo "HIP_PATH: $HIP_PATH" 
echo "HIP_PLATFORM: $HIP_PLATFORM"
echo ""

echo "=== GPU Architecture Detection ==="
if command -v rocminfo >/dev/null 2>&1; then
    GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9]*' | head -1)
    if [ -n "$GPU_ARCH" ]; then
        echo "Detected GPU architecture: $GPU_ARCH"
    else
        echo "Warning: Could not detect GPU architecture from rocminfo"
    fi
else
    echo "Warning: rocminfo not available for arch detection"
fi
echo ""

echo "=== Container GPU Access Check ==="
if [ -c /dev/kfd ]; then
    echo "✓ /dev/kfd device available"
    ls -la /dev/kfd
else
    echo "✗ /dev/kfd device not accessible"
fi

if ls /dev/dri/render* >/dev/null 2>&1; then
    echo "✓ DRI render devices available:"
    ls -la /dev/dri/render*
else
    echo "✗ DRI render devices not accessible"
fi
echo ""

echo "=== Simple HIP Test ==="
cd /tmp
cat > test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    printf("Starting HIP device test...\n");
    
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        printf("HIP Error getting device count: %s\n", hipGetErrorString(error));
        return 1;
    }
    
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No HIP devices available\n");
        return 1;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        error = hipGetDeviceProperties(&props, i);
        if (error != hipSuccess) {
            printf("Error getting properties for device %d: %s\n", i, hipGetErrorString(error));
            continue;
        }
        printf("Device %d: %s\n", i, props.name);
        printf("  Compute capability: %d.%d\n", props.major, props.minor);
        printf("  Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", props.multiProcessorCount);
    }
    
    // Test simple memory allocation
    printf("\nTesting memory allocation...\n");
    void* ptr;
    error = hipMalloc(&ptr, 1024);
    if (error != hipSuccess) {
        printf("Memory allocation failed: %s\n", hipGetErrorString(error));
        return 1;
    }
    
    printf("Memory allocation successful\n");
    hipFree(ptr);
    
    return 0;
}
EOF

echo "Compiling HIP test..."
# Use detected GPU architecture and proper ROCm path
ROCM_PATH_VAL="${ROCM_PATH:-/opt/rocm}"
if hipcc --rocm-path="$ROCM_PATH_VAL" -I"$ROCM_PATH_VAL/include" --offload-arch="$GPU_ARCH" -o test_hip test_hip.cpp 2>&1; then
    echo "✓ Compilation successful"
    echo "Running HIP test..."
    if timeout 10s ./test_hip 2>&1; then
        echo "✓ HIP test completed successfully"
    else
        echo "✗ HIP test failed or timed out"
    fi
else
    echo "✗ Compilation failed"
fi

rm -f test_hip test_hip.cpp
echo ""
echo "=== Diagnosis completed ==="