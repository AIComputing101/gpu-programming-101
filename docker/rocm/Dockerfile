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

# Install core ROCm development packages (keep minimal)
RUN apt-get update && apt-get install -y \
    # Core ROCm packages for GPU programming
    hip-dev \
    hip-samples \
    hipblas-dev \
    # ROCm profiling tools (essential for performance work)
    rocprofiler-dev \
    roctracer-dev \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python packages for basic development (no heavy data science libs)
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib

# Set up ROCm environment variables
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm/hip
ENV PATH=${ROCM_PATH}/bin:${HIP_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${HIP_PATH}/lib:${LD_LIBRARY_PATH}
ENV HIP_PLATFORM=amd
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV ROCM_VERSION=6.4.3

# Verify HIP compiler installation (skip rocminfo as no GPU during build)
RUN hipcc --version

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

# Create a simple GPU test script
RUN printf '#!/bin/bash\n\
echo "=== GPU Programming 101 - ROCm Environment Test ==="\n\
echo "Date: $(date)"\n\
echo ""\n\
\n\
echo "=== HIP Compiler ==="\n\
hipcc --version\n\
echo ""\n\
\n\
echo "=== GPU Information ==="\n\
if rocm-smi --showproductname --showmeminfo vram 2>/dev/null; then\n\
    echo "AMD GPU detected successfully"\n\
else\n\
    echo "No AMD GPU detected or rocm-smi not available"\n\
fi\n\
echo ""\n\
\n\
echo "=== Environment Variables ==="\n\
echo "ROCM_PATH: $ROCM_PATH"\n\
echo "HIP_PATH: $HIP_PATH"\n\
echo "HIP_PLATFORM: $HIP_PLATFORM"\n\
echo "PATH: $PATH"\n\
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"\n\
echo ""\n\
\n\
echo "=== Build Test ==="\n\
cd /tmp\n\
cat > test.hip.cpp << '"'"'HIP_EOF'"'"'\n\
#include <hip/hip_runtime.h>\n\
#include <stdio.h>\n\
\n\
__global__ void hello() {\n\
    printf("Hello from HIP thread %%d!\\n", hipThreadIdx_x);\n\
}\n\
\n\
int main() {\n\
    printf("HIP Test Program\\n");\n\
    \n\
    int deviceCount;\n\
    hipError_t error = hipGetDeviceCount(&deviceCount);\n\
    \n\
    if (error != hipSuccess) {\n\
        printf("HIP Error: %%s\\n", hipGetErrorString(error));\n\
        printf("No HIP-capable devices found\\n");\n\
        return 0;\n\
    }\n\
    \n\
    printf("Found %%d HIP device(s)\\n", deviceCount);\n\
    hello<<<1, 5>>>();\n\
    hipDeviceSynchronize();\n\
    printf("GPU kernel completed!\\n");\n\
    return 0;\n\
}\n\
HIP_EOF\n\
\n\
echo "Compiling test HIP program..."\n\
if hipcc -o test test.hip.cpp; then\n\
    echo "✓ Compilation successful"\n\
    echo "Running test program:"\n\
    ./test\n\
    echo "✓ HIP environment is working correctly!"\n\
else\n\
    echo "✗ Compilation failed"\n\
    exit 1\n\
fi\n\
\n\
rm -f test test.hip.cpp\n\
echo ""\n\
echo "=== All tests completed ==="\n' > /workspace/test-gpu.sh

RUN chmod +x /workspace/test-gpu.sh

# Install HIP samples for learning and reference
RUN cd /workspace && \
    if [ -d "/opt/rocm/hip/samples" ]; then \
        cp -r /opt/rocm/hip/samples ./hip-samples; \
    else \
        git clone https://github.com/ROCm-Developer-Tools/HIP-Examples.git hip-examples; \
    fi

# Default command
CMD ["/bin/bash"]

# Health check to verify HIP compiler access (will only work when GPU is available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD hipcc --version > /dev/null 2>&1 || exit 1