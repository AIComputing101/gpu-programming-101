# GPU Programming 101 - CUDA Development Container
# Based on NVIDIA's official CUDA 12.9.1 development image (latest stable as of 2025)

FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Metadata
LABEL maintainer="GPU Programming 101"
LABEL description="CUDA development environment for GPU programming course"
LABEL version="2.0"
LABEL cuda.version="12.9.1"
LABEL ubuntu.version="22.04"

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install essential development tools
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
    # GPU monitoring tools
    nvidia-utils-535 \
    # Debugging and profiling tools
    gdb \
    valgrind \
    strace \
    # Network tools for downloading samples
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA profiling tools (Nsight Systems, Compute) - Latest 2025 versions
RUN apt-get update && apt-get install -y \
    nsight-systems-2025.1.1 \
    nsight-compute-2025.1.1 \
    && rm -rf /var/lib/apt/lists/* || \
    # Fallback to 2024 versions if 2025 not available yet
    (apt-get update && apt-get install -y \
    nsight-systems-2024.6.1 \
    nsight-compute-2024.3.1 \
    && rm -rf /var/lib/apt/lists/*)

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

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_VERSION=12.9.1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Verify CUDA installation
RUN nvcc --version && nvidia-smi

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
    echo 'alias gpu-info="nvidia-smi"' >> /root/.bashrc && \
    echo 'alias cuda-info="nvcc --version"' >> /root/.bashrc && \
    echo 'export PS1="\[\e[1;32m\][CUDA-DEV]\[\e[0m\] \w $ "' >> /root/.bashrc

# Create a simple GPU test script
RUN cat > /workspace/test-gpu.sh << 'EOF'
#!/bin/bash
echo "=== GPU Programming 101 - CUDA Environment Test ==="
echo "Date: $(date)"
echo ""

echo "=== CUDA Compiler ==="
nvcc --version
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
echo ""

echo "=== CUDA Samples Test ==="
if [ -d "/usr/local/cuda/samples" ]; then
    echo "CUDA samples directory found"
else
    echo "CUDA samples not found - this is normal for newer CUDA versions"
fi

echo "=== Environment Variables ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "=== Build Test ==="
cd /tmp
cat > test.cu << 'CUDA_EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("CUDA Test Program\n");
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    printf("GPU kernel completed!\n");
    return 0;
}
CUDA_EOF

echo "Compiling test CUDA program..."
if nvcc -o test test.cu; then
    echo "✓ Compilation successful"
    echo "Running test program:"
    ./test
    echo "✓ CUDA environment is working correctly!"
else
    echo "✗ Compilation failed"
    exit 1
fi

rm -f test test.cu
echo ""
echo "=== All tests completed ==="
EOF

RUN chmod +x /workspace/test-gpu.sh

# Install additional CUDA samples and utilities
RUN cd /workspace && \
    git clone https://github.com/NVIDIA/cuda-samples.git && \
    cd cuda-samples && \
    git checkout v12.9

# Create jupyter kernel for CUDA (for notebooks)
RUN python3 -m ipykernel install --name cuda-kernel --display-name "CUDA Python"

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/bin/bash"]

# Health check to verify GPU access
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi > /dev/null 2>&1 || exit 1