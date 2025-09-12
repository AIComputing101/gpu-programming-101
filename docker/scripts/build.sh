#!/bin/bash

# GPU Programming 101 - Docker Build Script
# Builds CUDA and ROCm development containers

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

# Check requirements
check_requirements() {
    log "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "Requirements check passed"
}

# Build specific platform
build_platform() {
    local platform=$1
    local dockerfile_path="$DOCKER_DIR/$platform/Dockerfile"
    local image_tag="gpu-programming-101:$platform"
    
    if [ ! -f "$dockerfile_path" ]; then
        error "Dockerfile not found: $dockerfile_path"
        return 1
    fi
    
    log "Building $platform container..."
    log "Context: $PROJECT_ROOT"
    log "Dockerfile: $dockerfile_path"
    log "Tag: $image_tag"
    
    cd "$PROJECT_ROOT"
    
    if docker build -f "$dockerfile_path" -t "$image_tag" .; then
        success "$platform container built successfully"
        return 0
    else
        error "Failed to build $platform container"
        return 1
    fi
}

# Build using docker-compose
build_compose() {
    local service=$1
    log "Building $service using docker-compose..."
    
    cd "$DOCKER_DIR"
    
    if docker-compose build "$service"; then
        success "$service built successfully using docker-compose"
        return 0
    else
        error "Failed to build $service using docker-compose"
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
GPU Programming 101 - Docker Build Script

Usage: $0 [OPTIONS] [PLATFORM]

Options:
    -h, --help          Show this help message
    -a, --all           Build all platforms (CUDA + ROCm)
    -c, --compose       Use docker-compose for building
    --clean             Clean existing images before building
    --no-cache          Build without using Docker cache
    --pull              Pull base images before building

Platforms:
    cuda                Build NVIDIA CUDA container
    rocm                Build AMD ROCm container

Examples:
    $0 cuda             Build CUDA container only
    $0 rocm             Build ROCm container only  
    $0 --all            Build both CUDA and ROCm containers
    $0 --compose cuda   Build using docker-compose
    $0 --clean --all    Clean and rebuild all containers

EOF
}

# Clean existing images
clean_images() {
    log "Cleaning existing GPU Programming 101 images..."
    
    # Remove containers first
    if docker ps -a --filter "ancestor=gpu-programming-101:cuda" -q | xargs -r docker rm -f; then
        log "Removed CUDA containers"
    fi
    
    if docker ps -a --filter "ancestor=gpu-programming-101:rocm" -q | xargs -r docker rm -f; then
        log "Removed ROCm containers"
    fi
    
    # Remove images
    if docker images "gpu-programming-101:*" -q | xargs -r docker rmi -f; then
        success "Cleaned existing images"
    else
        warning "No images to clean or cleanup failed"
    fi
}

# Main function
main() {
    local build_all=false
    local use_compose=false
    local clean=false
    local no_cache=""
    local pull=false
    local platforms=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -a|--all)
                build_all=true
                shift
                ;;
            -c|--compose)
                use_compose=true
                shift
                ;;
            --clean)
                clean=true
                shift
                ;;
            --no-cache)
                no_cache="--no-cache"
                shift
                ;;
            --pull)
                pull=true
                shift
                ;;
            cuda|rocm)
                platforms+=("$1")
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set default platforms if none specified
    if [ "$build_all" = true ]; then
        platforms=("cuda" "rocm")
    elif [ ${#platforms[@]} -eq 0 ]; then
        platforms=("cuda")  # Default to CUDA
    fi
    
    log "Starting GPU Programming 101 Docker build process"
    log "Platforms to build: ${platforms[*]}"
    
    check_requirements
    
    if [ "$clean" = true ]; then
        clean_images
    fi
    
    if [ "$pull" = true ]; then
        log "Pulling base images..."
        docker pull nvidia/cuda:12.4-devel-ubuntu22.04 || warning "Failed to pull CUDA base image"
        docker pull rocm/dev-ubuntu-22.04:6.0 || warning "Failed to pull ROCm base image"
    fi
    
    local success_count=0
    local total_count=${#platforms[@]}
    
    for platform in "${platforms[@]}"; do
        log "Building platform: $platform"
        
        if [ "$use_compose" = true ]; then
            if build_compose "$platform-dev"; then
                ((success_count++))
            fi
        else
            # Add no-cache flag if specified
            if [ -n "$no_cache" ]; then
                cd "$PROJECT_ROOT"
                if docker build $no_cache -f "$DOCKER_DIR/$platform/Dockerfile" -t "gpu-programming-101:$platform" .; then
                    ((success_count++))
                fi
            else
                if build_platform "$platform"; then
                    ((success_count++))
                fi
            fi
        fi
    done
    
    log "Build process completed"
    success "Successfully built $success_count out of $total_count platforms"
    
    if [ $success_count -eq $total_count ]; then
        success "All containers built successfully!"
        log "Next steps:"
        log "  - Run: ./docker/scripts/run.sh cuda    (for NVIDIA GPUs)"
        log "  - Run: ./docker/scripts/run.sh rocm    (for AMD GPUs)"
        log "  - Or use: docker-compose -f docker/docker-compose.yml up cuda-dev"
    else
        warning "Some builds failed. Check the logs above for details."
        exit 1
    fi
}

# Run main function
main "$@"