# GPU Programming 101 - ROCm Development Container
# Based on AMD's official ROCm 6.4.3 development image (latest stable as of 2025)

FROM rocm/dev-ubuntu-22.04:6.4.3

# Metadata
LABEL maintainer="GPU Programming 101"
LABEL description="ROCm/HIP development environment for GPU programming course"
LABEL version="2.0"
LABEL rocm.version="6.4.3"
LABEL ubuntu.version="22.04"

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Update and install essential development tools
RUN apt-get update && apt-get install -y \
    # Basic development tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    # Python development
    python3 \
    python3-pip \
    python3-dev \
    # Additional utilities
    pkg-config \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    # Debugging and profiling tools
    gdb \
    valgrind \
    strace \
    # Network tools
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install ROCm development packages
RUN apt-get update && apt-get install -y \
    # Core ROCm packages
    rocm-dev \
    rocm-libs \
    hip-dev \
    hip-samples \
    hipblas-dev \
    hipfft-dev \
    hipsparse-dev \
    # ROCm profiling and debugging tools
    rocprofiler-dev \
    roctracer-dev \
    roctx \
    # Additional ROCm libraries
    rocrand-dev \
    rocthrust-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for data analysis and visualization
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    jupyterlab \
    plotly \
    scipy

# Set up ROCm environment variables
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm/hip
ENV PATH=${ROCM_PATH}/bin:${HIP_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${HIP_PATH}/lib:${LD_LIBRARY_PATH}
ENV HIP_PLATFORM=amd
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV ROCM_VERSION=6.4.3

# Add ROCm binaries to PATH
ENV PATH=/opt/rocm/bin:/opt/rocm/hip/bin:${PATH}

# Verify ROCm installation
RUN hipcc --version && rocminfo > /dev/null 2>&1 || echo "ROCm info check completed (may fail without GPU)"

# Create development workspace
WORKDIR /workspace
RUN mkdir -p /workspace/{projects,samples,output}

# Copy course materials (will be mounted as volume in practice)
COPY . /workspace/gpu-programming-101/

# Set up convenient aliases and environment
RUN echo 'alias ll="ls -alF"' >> /root/.bashrc && \
    echo 'alias la="ls -A"' >> /root/.bashrc && \
    echo 'alias l="ls -CF"' >> /root/.bashrc && \
    echo 'alias cls="clear"' >> /root/.bashrc && \
    echo 'alias gpu-info="rocm-smi"' >> /root/.bashrc && \
    echo 'alias hip-info="hipcc --version"' >> /root/.bashrc && \
    echo 'alias rocm-info="rocminfo"' >> /root/.bashrc && \
    echo 'export PS1="\[\e[1;34m\][ROCm-DEV]\[\e[0m\] \w $ "' >> /root/.bashrc

# Create a comprehensive GPU test script
RUN cat > /workspace/test-gpu.sh << 'EOF'
#!/bin/bash
echo "=== GPU Programming 101 - ROCm Environment Test ==="
echo "Date: $(date)"
echo ""

echo "=== HIP Compiler ==="
hipcc --version
echo ""

echo "=== ROCm Version ==="
if command -v rocminfo > /dev/null 2>&1; then
    rocminfo | head -20
else
    echo "rocminfo command not available"
fi
echo ""

echo "=== GPU Information ==="
if command -v rocm-smi > /dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram || echo "No AMD GPU detected or accessible"
else
    echo "rocm-smi not available"
fi
echo ""

echo "=== Environment Variables ==="
echo "ROCM_PATH: $ROCM_PATH"
echo "HIP_PATH: $HIP_PATH"
echo "HIP_PLATFORM: $HIP_PLATFORM"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "=== HIP Platform Detection ==="
cat > /tmp/platform_test.cpp << 'HIP_EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cout << "HIP Error: " << hipGetErrorString(error) << std::endl;
        std::cout << "This may be normal if no GPU is available" << std::endl;
    } else {
        std::cout << "Number of HIP devices: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, i);
            std::cout << "Device " << i << ": " << props.name << std::endl;
        }
    }
    
    return 0;
}
HIP_EOF

echo "Compiling platform detection test..."
if hipcc -o /tmp/platform_test /tmp/platform_test.cpp; then
    echo "✓ Compilation successful"
    echo "Running platform test:"
    /tmp/platform_test
else
    echo "✗ Platform test compilation failed"
fi

echo ""
echo "=== Build Test ==="
cd /tmp
cat > test.hip.cpp << 'HIP_EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello() {
    printf("Hello from HIP thread %d!\n", hipThreadIdx_x);
}

int main() {
    printf("HIP Test Program\n");
    
    // Check for HIP devices
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error == hipSuccess && deviceCount > 0) {
        printf("Found %d HIP device(s)\n", deviceCount);
        hello<<<1, 5>>>();
        hipDeviceSynchronize();
        printf("GPU kernel completed!\n");
    } else {
        printf("No HIP devices found or error: %s\n", hipGetErrorString(error));
        printf("This is normal when running without GPU access\n");
    }
    
    return 0;
}
HIP_EOF

echo "Compiling test HIP program..."
if hipcc -o test test.hip.cpp; then
    echo "✓ Compilation successful"
    echo "Running test program:"
    ./test
    echo "✓ HIP environment is working correctly!"
else
    echo "✗ Compilation failed"
    exit 1
fi

rm -f test test.hip.cpp platform_test platform_test.cpp
echo ""
echo "=== All tests completed ==="
EOF

RUN chmod +x /workspace/test-gpu.sh

# Install HIP samples
RUN cd /workspace && \
    if [ -d "/opt/rocm/hip/samples" ]; then \
        cp -r /opt/rocm/hip/samples ./hip-samples; \
    else \
        git clone https://github.com/ROCm-Developer-Tools/HIP-Examples.git hip-examples; \
    fi

# Create jupyter kernel for HIP (for notebooks)
RUN python3 -m ipykernel install --name hip-kernel --display-name "HIP Python"

# Set up HIP for both AMD and NVIDIA compatibility
RUN cat > /workspace/setup-hip-nvidia.sh << 'EOF'
#!/bin/bash
# Switch HIP to NVIDIA backend (for systems with NVIDIA GPUs)
export HIP_PLATFORM=nvidia
export HIP_COMPILER=nvcc
echo "HIP configured for NVIDIA backend"
echo "HIP_PLATFORM=$HIP_PLATFORM"
EOF

RUN cat > /workspace/setup-hip-amd.sh << 'EOF'  
#!/bin/bash
# Switch HIP to AMD backend (default)
export HIP_PLATFORM=amd
unset HIP_COMPILER
echo "HIP configured for AMD backend"  
echo "HIP_PLATFORM=$HIP_PLATFORM"
EOF

RUN chmod +x /workspace/setup-hip-*.sh

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/bin/bash"]

# Health check to verify ROCm access
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD rocminfo > /dev/null 2>&1 || hipcc --version > /dev/null 2>&1 || exit 1