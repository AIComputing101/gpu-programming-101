#!/bin/bash

# GPU Programming 101 - Docker Test Script
# Tests the Docker development environment setup

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
    echo -e "${BLUE}[TEST]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test prerequisites
test_prerequisites() {
    log "Testing prerequisites..."
    local passed=0
    local total=3
    
    # Test Docker
    if command -v docker &> /dev/null; then
        success "Docker is installed"
        ((passed++))
    else
        error "Docker is not installed"
    fi
    
    # Test Docker Compose
    if command -v docker-compose &> /dev/null; then
        success "Docker Compose is installed"
        ((passed++))
    else
        error "Docker Compose is not installed"
    fi
    
    # Test Docker daemon
    if docker info &> /dev/null; then
        success "Docker daemon is running"
        ((passed++))
    else
        error "Docker daemon is not running or accessible"
    fi
    
    log "Prerequisites: $passed/$total passed"
    return $((total - passed))
}

# Test GPU support
test_gpu_support() {
    log "Testing GPU support..."
    local gpu_nvidia=false
    local gpu_amd=false
    
    # Test NVIDIA
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        success "NVIDIA GPU detected"
        gpu_nvidia=true
    else
        warning "NVIDIA GPU not detected or nvidia-smi not available"
    fi
    
    # Test AMD
    if command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null; then
        success "AMD GPU detected"
        gpu_amd=true
    elif [ -c /dev/kfd ] && [ -c /dev/dri/renderD128 ]; then
        success "AMD GPU devices detected"
        gpu_amd=true
    else
        warning "AMD GPU not detected"
    fi
    
    if [ "$gpu_nvidia" = false ] && [ "$gpu_amd" = false ]; then
        warning "No GPU detected - will test CPU-only mode"
        return 1
    fi
    
    return 0
}

# Test Docker build
test_build() {
    local platform=$1
    log "Testing build for $platform..."
    
    if [ ! -f "$DOCKER_DIR/$platform/Dockerfile" ]; then
        error "Dockerfile not found: $DOCKER_DIR/$platform/Dockerfile"
        return 1
    fi
    
    # Try building with a simple test
    cd "$PROJECT_ROOT"
    if docker build -f "$DOCKER_DIR/$platform/Dockerfile" -t "gpu-programming-101:$platform-test" --target test 2>/dev/null || \
       docker build -f "$DOCKER_DIR/$platform/Dockerfile" -t "gpu-programming-101:$platform-test" . >/dev/null 2>&1; then
        success "$platform build test passed"
        # Clean up test image
        docker rmi "gpu-programming-101:$platform-test" >/dev/null 2>&1 || true
        return 0
    else
        error "$platform build test failed"
        return 1
    fi
}

# Test container run
test_container_run() {
    local platform=$1
    local image_name="gpu-programming-101:$platform"
    
    log "Testing container run for $platform..."
    
    # Check if image exists
    if ! docker images "$image_name" | grep -q "$image_name"; then
        warning "Image $image_name not found - skipping run test"
        return 1
    fi
    
    # Test basic container run
    local container_name="gpu101-$platform-test"
    local test_cmd="echo 'Container test passed' && ls /workspace"
    
    if docker run --rm --name "$container_name" "$image_name" /bin/bash -c "$test_cmd" >/dev/null 2>&1; then
        success "$platform container run test passed"
        return 0
    else
        error "$platform container run test failed"
        return 1
    fi
}

# Test GPU access in container
test_gpu_access() {
    local platform=$1
    local image_name="gpu-programming-101:$platform"
    
    log "Testing GPU access in $platform container..."
    
    # Check if image exists
    if ! docker images "$image_name" | grep -q "$image_name"; then
        warning "Image $image_name not found - skipping GPU access test"
        return 1
    fi
    
    local container_name="gpu101-$platform-gpu-test"
    local gpu_args=""
    local test_cmd=""
    
    case $platform in
        cuda)
            gpu_args="--gpus all"
            test_cmd="nvidia-smi && echo 'CUDA GPU access OK'"
            ;;
        rocm)
            if [ -c /dev/kfd ] && [ -c /dev/dri/renderD128 ]; then
                gpu_args="--device=/dev/kfd --device=/dev/dri"
                test_cmd="rocminfo | head -5 && echo 'ROCm GPU access OK'"
            else
                warning "AMD GPU devices not found - skipping GPU access test"
                return 1
            fi
            ;;
        *)
            error "Unknown platform: $platform"
            return 1
            ;;
    esac
    
    if docker run --rm --name "$container_name" $gpu_args "$image_name" /bin/bash -c "$test_cmd" >/dev/null 2>&1; then
        success "$platform GPU access test passed"
        return 0
    else
        warning "$platform GPU access test failed (this may be normal without GPU)"
        return 1
    fi
}

# Test course examples build
test_examples_build() {
    local platform=$1
    local image_name="gpu-programming-101:$platform"
    
    log "Testing examples build in $platform container..."
    
    # Check if image exists
    if ! docker images "$image_name" | grep -q "$image_name"; then
        warning "Image $image_name not found - skipping examples test"
        return 1
    fi
    
    local container_name="gpu101-$platform-examples-test"
    local test_cmd="cd /workspace/gpu-programming-101/modules/module1/examples && make --dry-run"
    
    if docker run --rm --name "$container_name" -v "$PROJECT_ROOT:/workspace/gpu-programming-101:ro" "$image_name" /bin/bash -c "$test_cmd" >/dev/null 2>&1; then
        success "$platform examples build test passed"
        return 0
    else
        error "$platform examples build test failed"
        return 1
    fi
}

# Test docker-compose
test_compose() {
    log "Testing docker-compose configuration..."
    
    cd "$DOCKER_DIR"
    
    if docker-compose config >/dev/null 2>&1; then
        success "docker-compose configuration is valid"
        return 0
    else
        error "docker-compose configuration is invalid"
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
GPU Programming 101 - Docker Test Script

Usage: $0 [OPTIONS] [TESTS]

Options:
    -h, --help          Show this help message
    --verbose           Show detailed output
    --quick             Run only quick tests (no builds)
    --gpu-only          Test only GPU-related functionality

Tests:
    all                 Run all tests (default)
    prereq              Test prerequisites only
    gpu                 Test GPU support only
    build               Test Docker builds only  
    run                 Test container runs only
    examples            Test examples build only
    compose             Test docker-compose only

Examples:
    $0                  Run all tests
    $0 prereq           Test prerequisites only
    $0 --quick          Run quick tests only
    $0 gpu build        Test GPU support and builds

EOF
}

# Main test function
main() {
    local run_all=true
    local verbose=false
    local quick=false
    local gpu_only=false
    local tests=()
    local total_passed=0
    local total_tests=0
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --quick)
                quick=true
                shift
                ;;
            --gpu-only)
                gpu_only=true
                shift
                ;;
            all|prereq|gpu|build|run|examples|compose)
                tests+=("$1")
                run_all=false
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set default tests
    if [ "$run_all" = true ]; then
        if [ "$quick" = true ]; then
            tests=("prereq" "gpu" "compose")
        elif [ "$gpu_only" = true ]; then
            tests=("gpu")
        else
            tests=("prereq" "gpu" "build" "run" "examples" "compose")
        fi
    fi
    
    log "Starting GPU Programming 101 Docker tests"
    log "Tests to run: ${tests[*]}"
    
    # Run tests
    for test in "${tests[@]}"; do
        case $test in
            prereq)
                log "=== Testing Prerequisites ==="
                if test_prerequisites; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
                
            gpu)
                log "=== Testing GPU Support ==="
                if test_gpu_support; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
                
            build)
                log "=== Testing Docker Builds ==="
                local build_passed=0
                local build_total=0
                
                for platform in cuda rocm; do
                    if test_build "$platform"; then
                        ((build_passed++))
                    fi
                    ((build_total++))
                done
                
                if [ $build_passed -eq $build_total ]; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
                
            run)
                log "=== Testing Container Runs ==="
                local run_passed=0
                local run_total=0
                
                for platform in cuda rocm; do
                    if test_container_run "$platform"; then
                        ((run_passed++))
                    fi
                    ((run_total++))
                done
                
                if [ $run_passed -gt 0 ]; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
                
            examples)
                log "=== Testing Examples Build ==="
                local examples_passed=0
                local examples_total=0
                
                for platform in cuda rocm; do
                    if test_examples_build "$platform"; then
                        ((examples_passed++))
                    fi
                    ((examples_total++))
                done
                
                if [ $examples_passed -gt 0 ]; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
                
            compose)
                log "=== Testing Docker Compose ==="
                if test_compose; then
                    ((total_passed++))
                fi
                ((total_tests++))
                ;;
        esac
        
        log "---"
    done
    
    # Summary
    log "=== Test Summary ==="
    log "Passed: $total_passed/$total_tests tests"
    
    if [ $total_passed -eq $total_tests ]; then
        success "All tests passed! 🎉"
        log "Your Docker environment is ready for GPU programming."
        log ""
        log "Next steps:"
        log "  1. Build containers: ./docker/scripts/build.sh --all"
        log "  2. Start development: ./docker/scripts/run.sh --auto"
        log "  3. Test examples: cd modules/module1/examples && make"
        return 0
    else
        warning "Some tests failed. Check the output above for details."
        log ""
        log "Common solutions:"
        log "  - Install missing prerequisites (Docker, Docker Compose)"
        log "  - Install GPU drivers and container runtime"
        log "  - Check Docker daemon permissions"
        return 1
    fi
}

# Run main function
main "$@"