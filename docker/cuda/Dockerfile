# GPU Programming 101 - CUDA Development Container
# Based on NVIDIA's official CUDA 13.0.1 cuDNN development image (Ubuntu 24.04)

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Metadata
LABEL maintainer="GPU Programming 101"
LABEL description="CUDA development environment for GPU programming course"
LABEL version="2.0"
LABEL cuda.version="13.0.1"
LABEL ubuntu.version="24.04"

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install essential development tools for GPU programming
RUN apt-get update && apt-get install -y \
    # Core development tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    # Minimal Python for basic scripting (not data science)
    python3 \
    python3-pip \
    python3-dev \
    # Additional utilities
    pkg-config \
    software-properties-common \
    # Debugging and profiling tools
    gdb \
    valgrind \
    strace \
    # Network tools
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install optional CUDA tools if available
RUN apt-get update && \
    (apt-get install -y cuda-tools-13-0 || apt-get install -y cuda-tools || true) && \
    rm -rf /var/lib/apt/lists/*

# Install minimal Python packages for basic development (no heavy data science libs)
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_VERSION=13.0.1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Verify CUDA compiler installation (skip nvidia-smi as no GPU during build)
RUN nvcc --version

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
RUN printf '#!/bin/bash\n\
echo "=== GPU Programming 101 - CUDA Environment Test ==="\n\
echo "Date: $(date)"\n\
echo ""\n\
\n\
echo "=== CUDA Compiler ==="\n\
nvcc --version\n\
echo ""\n\
\n\
echo "=== GPU Information ==="\n\
if nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv 2>/dev/null; then\n\
    echo "GPU detected successfully"\n\
else\n\
    echo "No GPU detected or nvidia-smi not available"\n\
fi\n\
echo ""\n\
\n\
echo "=== Environment Variables ==="\n\
echo "CUDA_HOME: $CUDA_HOME"\n\
echo "PATH: $PATH"\n\
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"\n\
echo ""\n\
\n\
echo "=== Build Test ==="\n\
cd /tmp\n\
cat > test.cu << '"'"'CUDA_EOF'"'"'\n\
#include <cuda_runtime.h>\n\
#include <stdio.h>\n\
\n\
__global__ void hello() {\n\
    printf("Hello from GPU thread %%d!\\n", threadIdx.x);\n\
}\n\
\n\
int main() {\n\
    printf("CUDA Test Program\\n");\n\
    \n\
    int deviceCount;\n\
    cudaError_t error = cudaGetDeviceCount(&deviceCount);\n\
    \n\
    if (error != cudaSuccess) {\n\
        printf("CUDA Error: %%s\\n", cudaGetErrorString(error));\n\
        printf("No CUDA-capable devices found\\n");\n\
        return 0;\n\
    }\n\
    \n\
    printf("Found %%d CUDA device(s)\\n", deviceCount);\n\
    hello<<<1, 5>>>();\n\
    cudaDeviceSynchronize();\n\
    printf("GPU kernel completed!\\n");\n\
    return 0;\n\
}\n\
CUDA_EOF\n\
\n\
echo "Compiling test CUDA program..."\n\
if nvcc -o test test.cu; then\n\
    echo "✓ Compilation successful"\n\
    echo "Running test program:"\n\
    ./test\n\
    echo "✓ CUDA environment is working correctly!"\n\
else\n\
    echo "✗ Compilation failed"\n\
    exit 1\n\
fi\n\
\n\
rm -f test test.cu\n\
echo ""\n\
echo "=== All tests completed ==="\n' > /workspace/test-gpu.sh

RUN chmod +x /workspace/test-gpu.sh

# Install CUDA samples for learning and reference
RUN cd /workspace && \
    git clone https://github.com/NVIDIA/cuda-samples.git && \
    cd cuda-samples && \
    git checkout v13.0

# Default command
CMD ["/bin/bash"]

# Health check to verify GPU access (will only work when GPU is available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvcc --version > /dev/null 2>&1 || exit 1