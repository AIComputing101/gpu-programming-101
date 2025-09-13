#!/bin/bash

# GPU Programming 101 - Docker Run Script
# Runs CUDA and ROCm development containers with proper GPU access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DOCKER_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if image exists
check_image() {
    local image_name=$1
    if ! docker image inspect "$image_name" >/dev/null 2>&1; then
        error "Image $image_name not found. Please build it first using:"
        error "  ./docker/scripts/build.sh"
        return 1
    fi
    return 0
}

# Detect GPU platform
detect_gpu() {
    local gpu_type="none"
    
    # Check for NVIDIA GPU presence (multiple methods)
    if command -v nvidia-smi &> /dev/null; then
        gpu_type="nvidia"
        log "NVIDIA GPU tools detected (nvidia-smi found)" >&2
    elif [ -c /dev/nvidia0 ] || [ -c /dev/nvidiactl ]; then
        gpu_type="nvidia"
        log "NVIDIA GPU devices detected (/dev/nvidia*)" >&2
    elif lspci 2>/dev/null | grep -i nvidia &> /dev/null; then
        gpu_type="nvidia"
        log "NVIDIA GPU detected via lspci" >&2
    # Check for AMD
    elif command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null; then
        gpu_type="amd"  
        log "AMD GPU detected" >&2
    elif [ -c /dev/kfd ] && [ -c /dev/dri/renderD128 ]; then
        gpu_type="amd"
        log "AMD GPU devices detected" >&2
    else
        warning "No GPU detected or GPU tools not available" >&2
        warning "Container will run in CPU-only mode" >&2
    fi
    
    echo "$gpu_type"
}

# Run CUDA container
run_cuda() {
    local container_name="gpu101-cuda-dev"
    local image_name="gpu-programming-101:cuda"
    local gpu_args=""
    local ports_args="-p 8888:8888"
    local extra_args=()
    local no_gpu_requested=false
    
    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --name)
                container_name="$2"
                shift 2
                ;;
            --port)
                ports_args="-p $2:8888"
                shift 2
                ;;
            --no-gpu)
                no_gpu_requested=true
                shift
                ;;
            --detach|-d)
                extra_args+=("--detach")
                shift
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done
    
    if ! check_image "$image_name"; then
        return 1
    fi
    
    # Set up GPU access for NVIDIA
    local detected_gpu=$(detect_gpu)
    log "Detected GPU type: $detected_gpu"
    log "No GPU requested: $no_gpu_requested"
    
    if [ "$detected_gpu" = "nvidia" ] && [ "$no_gpu_requested" = false ]; then
        gpu_args="--gpus all"
        log "Enabling NVIDIA GPU access"
    elif [ "$no_gpu_requested" = true ]; then
        log "GPU access explicitly disabled with --no-gpu"
    else
        log "GPU access disabled (no NVIDIA GPU detected or other reason)"
    fi
    
    # Remove existing container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^$container_name$"; then
        log "Removing existing container: $container_name"
        docker rm -f "$container_name" > /dev/null
    fi
    
    log "Starting CUDA development container..."
    log "Container name: $container_name"
    log "Image: $image_name"
    log "GPU access: ${gpu_args:-disabled}"
    
    # Build docker run command
    local cmd=(
        docker run
        --name "$container_name"
        --hostname "cuda-dev"
        -it
        -v "$PROJECT_ROOT:/workspace/gpu-programming-101:rw"
        -v "gpu101-cuda-home:/root"
        -w "/workspace/gpu-programming-101"
    )
    
    # Add port mapping
    cmd+=($ports_args)
    
    # Add GPU args if available
    if [ -n "$gpu_args" ]; then
        cmd+=($gpu_args)
    fi
    
    # Add extra arguments
    cmd+=("${extra_args[@]}")
    
    # Add image
    cmd+=("$image_name")
    
    # Add default command if running interactively
    if [[ ! "${extra_args[*]}" =~ "--detach" ]]; then
        cmd+=("/bin/bash")
    fi
    
    # Execute command
    "${cmd[@]}"
}

# Run ROCm container
run_rocm() {
    local container_name="gpu101-rocm-dev"
    local image_name="gpu-programming-101:rocm"
    local gpu_args=""
    local ports_args="-p 8889:8888"
    local extra_args=()
    local no_gpu_requested=false
    
    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --name)
                container_name="$2"
                shift 2
                ;;
            --port)
                ports_args="-p $2:8888"
                shift 2
                ;;
            --no-gpu)
                no_gpu_requested=true
                shift
                ;;
            --detach|-d)
                extra_args+=("--detach")
                shift
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done
    
    if ! check_image "$image_name"; then
        return 1
    fi
    
    # Set up GPU access for AMD
    local detected_gpu=$(detect_gpu)
    if [ "$detected_gpu" = "amd" ] && [ "$no_gpu_requested" = false ]; then
        gpu_args="--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined"
        log "Enabling AMD GPU access"
    elif [ "$no_gpu_requested" = true ]; then
        log "GPU access explicitly disabled with --no-gpu"
    fi
    
    # Remove existing container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^$container_name$"; then
        log "Removing existing container: $container_name"
        docker rm -f "$container_name" > /dev/null
    fi
    
    log "Starting ROCm development container..."
    log "Container name: $container_name"
    log "Image: $image_name"
    log "GPU access: ${gpu_args:-disabled}"
    
    # Build docker run command
    local cmd=(
        docker run
        --name "$container_name"
        --hostname "rocm-dev"
        -it
        -v "$PROJECT_ROOT:/workspace/gpu-programming-101:rw"
        -v "gpu101-rocm-home:/root"
        -w "/workspace/gpu-programming-101"
        -e HIP_VISIBLE_DEVICES=0
        -e HSA_OVERRIDE_GFX_VERSION=10.3.0
    )
    
    # Add port mapping
    cmd+=($ports_args)
    
    # Add GPU args if available
    if [ -n "$gpu_args" ]; then
        cmd+=($gpu_args)
    fi
    
    # Add extra arguments
    cmd+=("${extra_args[@]}")
    
    # Add image
    cmd+=("$image_name")
    
    # Add default command if running interactively
    if [[ ! "${extra_args[*]}" =~ "--detach" ]]; then
        cmd+=("/bin/bash")
    fi
    
    # Execute command
    "${cmd[@]}"
}

# Run with docker compose (v2) or docker-compose (v1)
run_compose() {
    local service=$1
    shift
    
    log "Starting $service using docker compose..."
    cd "$DOCKER_DIR"
    
    # Parse arguments for docker compose
    local compose_args=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --detach|-d)
                compose_args+=("-d")
                shift
                ;;
            *)
                compose_args+=("$1")
                shift
                ;;
        esac
    done
    
    # Try docker compose (v2) first, then fall back to docker-compose (v1)
    if docker compose up "${compose_args[@]}" "$service" 2>/dev/null; then
        log "Started $service using docker compose (v2)"
    elif docker-compose up "${compose_args[@]}" "$service" 2>/dev/null; then
        log "Started $service using docker-compose (v1)"
    else
        error "Failed to start $service using docker compose"
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
GPU Programming 101 - Docker Run Script

Usage: $0 [PLATFORM] [OPTIONS]

Platforms:
    cuda                Run NVIDIA CUDA container
    rocm                Run AMD ROCm container
    compose SERVICE     Run using docker compose

Options:
    -h, --help          Show this help message
    --name NAME         Set custom container name
    --port PORT         Map port to host (default: 8888 for CUDA, 8889 for ROCm)
    --no-gpu            Disable GPU access (CPU-only mode)
    --detach, -d        Run in background (detached mode)
    --auto              Auto-detect GPU and run appropriate container

Examples:
    $0 cuda             Run CUDA container interactively
    $0 rocm             Run ROCm container interactively
    $0 cuda --detach    Run CUDA container in background
    $0 rocm --no-gpu    Run ROCm container in CPU-only mode
    $0 --auto           Auto-detect GPU type and run appropriate container
    $0 compose cuda-dev Run using docker compose

Container Management:
    List containers:    docker ps -a
    Stop container:     docker stop gpu101-cuda-dev
    Remove container:   docker rm gpu101-cuda-dev
    Container logs:     docker logs gpu101-cuda-dev
    Enter container:    docker exec -it gpu101-cuda-dev bash

GPU Programming Setup:
    Inside container:   /workspace/test-gpu.sh    # Test GPU environment
    Build examples:     cd modules/module1/examples && make
    CUDA samples:       cd /workspace/cuda-samples
    HIP samples:        cd /workspace/hip-examples

EOF
}

# Auto-detect and run appropriate container
auto_run() {
    local gpu_type=$(detect_gpu)
    
    case $gpu_type in
        nvidia)
            log "Auto-detected NVIDIA GPU, running CUDA container"
            run_cuda "$@"
            ;;
        amd)
            log "Auto-detected AMD GPU, running ROCm container"
            run_rocm "$@"
            ;;
        *)
            warning "No GPU detected, defaulting to CUDA container in CPU-only mode"
            run_cuda --no-gpu "$@"
            ;;
    esac
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        cuda)
            shift
            run_cuda "$@"
            ;;
        rocm)
            shift
            run_rocm "$@"
            ;;
        compose)
            shift
            run_compose "$@"
            ;;
        --auto)
            shift
            auto_run "$@"
            ;;
        *)
            error "Unknown platform: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"