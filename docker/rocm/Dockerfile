# GPU Programming 101 - ROCm Development Container
# Based on AMD's official ROCm development image - used as-is for maximum compatibility

FROM rocm/dev-ubuntu-24.04:latest

# Metadata
LABEL maintainer="GPU Programming 101"
LABEL description="ROCm/HIP development environment for GPU programming course"
LABEL version="2.0"
LABEL rocm.version="latest"
LABEL ubuntu.version="24.04"

# Create development workspace
WORKDIR /workspace
RUN mkdir -p /workspace/{projects,samples,output}

# Copy course materials (will be mounted as volume in practice)
COPY . /workspace/gpu-programming-101/

# Set up convenient aliases and environment for the course
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

# Default command
CMD ["/bin/bash"]