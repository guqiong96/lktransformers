#!/bin/bash

# KTransformers 环境变量设置模块
# 包含完整的环境变量配置逻辑，根据系统硬件特性自动设置优化参数

# 检测CPU指令集支持并设置AVX相关参数
detect_and_set_cpu_instructions() {
    log_info "检测CPU指令集支持并设置AVX相关参数..."
    
    local cpu_flags=""
    # 尝试多种方法获取CPU标志
    if command -v lscpu &> /dev/null; then
        cpu_flags=$(lscpu | grep "Flags:" | cut -d: -f2 2>/dev/null || echo "")
    fi
    
    # 如果lscpu失败，尝试从/proc/cpuinfo获取
    if [[ -z "$cpu_flags" && -f /proc/cpuinfo ]]; then
        cpu_flags=$(grep "flags" /proc/cpuinfo | head -1 | cut -d: -f2 2>/dev/null || echo "")
    fi
    
    # 如果仍然为空，尝试备用方法
    if [[ -z "$cpu_flags" && -f /proc/cpuinfo ]]; then
        cpu_flags=$(awk '/^flags/ {print $0; exit}' /proc/cpuinfo | cut -d: -f2 2>/dev/null || echo "")
    fi
    
    if [[ -z "$cpu_flags" ]]; then
        log_warn "无法获取CPU指令集信息，将使用默认设置"
        cpu_flags=""
    else
        log_info "成功获取CPU指令集信息"
    fi
    
    # 检测支持的指令集
    local supports_amx=false
    local supports_avx512=false
    local supports_avx2=false
    local supports_avx=false
    local supports_fma=false
    local supports_f16c=false
    local supports_sse4=false
    
    # 检测AMX指令集（Intel Advanced Matrix Extensions）
    if echo "$cpu_flags" | grep -qE "\bamx_tile\b|\bamx_int8\b|\bamx_bf16\b|\bamx\b"; then
        supports_amx=true
        log_info "检测到AMX指令集支持"
    fi
    
    # 检测AVX512指令集族
    if echo "$cpu_flags" | grep -qE "\bavx512[a-z_]*\b"; then
        supports_avx512=true
        log_info "检测到AVX512指令集支持"
    fi
    
    # 检测AVX2指令集
    if echo "$cpu_flags" | grep -q "\bavx2\b"; then
        supports_avx2=true
        log_info "检测到AVX2指令集支持"
    fi
    
    # 检测AVX指令集
    if echo "$cpu_flags" | grep -q "\bavx\b"; then
        supports_avx=true
        log_info "检测到AVX指令集支持"
    fi
    
    # 检测FMA指令集
    if echo "$cpu_flags" | grep -q "\bfma\b"; then
        supports_fma=true
        log_info "检测到FMA指令集支持"
    fi
    
    # 检测F16C指令集
    if echo "$cpu_flags" | grep -q "\bf16c\b"; then
        supports_f16c=true
        log_info "检测到F16C指令集支持"
    fi
    
    # 检测SSE4指令集
    if echo "$cpu_flags" | grep -qE "\bsse4_[12]\b"; then
        supports_sse4=true
        log_info "检测到SSE4指令集支持"
    fi
    
    # 构建CMAKE_ARGS中的LLAMA相关参数
    local llama_args=""
    
    # 设置LLAMA_NATIVE（通常设置为OFF以确保兼容性）
    llama_args="-DLLAMA_NATIVE=OFF"
    
    # 根据检测结果设置AVX相关参数
    if [[ "$supports_avx2" == "true" ]]; then
        llama_args="$llama_args -DLLAMA_AVX2=ON"
        log_info "启用LLAMA_AVX2优化"
    else
        llama_args="$llama_args -DLLAMA_AVX2=OFF"
        log_info "禁用LLAMA_AVX2优化（CPU不支持）"
    fi
    
    if [[ "$supports_avx512" == "true" ]]; then
        llama_args="$llama_args -DLLAMA_AVX512=ON"
        log_info "启用LLAMA_AVX512优化"
    else
        llama_args="$llama_args -DLLAMA_AVX512=OFF"
        log_info "禁用LLAMA_AVX512优化（CPU不支持）"
    fi
    
    if [[ "$supports_fma" == "true" ]]; then
        llama_args="$llama_args -DLLAMA_FMA=ON"
        log_info "启用LLAMA_FMA优化"
    else
        llama_args="$llama_args -DLLAMA_FMA=OFF"
        log_info "禁用LLAMA_FMA优化（CPU不支持）"
    fi
    
    if [[ "$supports_f16c" == "true" ]]; then
        llama_args="$llama_args -DLLAMA_F16C=ON"
        log_info "启用LLAMA_F16C优化"
    else
        llama_args="$llama_args -DLLAMA_F16C=OFF"
        log_info "禁用LLAMA_F16C优化（CPU不支持）"
    fi
    
    # 导出检测结果供其他脚本使用
    export CPU_SUPPORTS_AMX="$supports_amx"
    export CPU_SUPPORTS_AVX512="$supports_avx512"
    export CPU_SUPPORTS_AVX2="$supports_avx2"
    export CPU_SUPPORTS_AVX="$supports_avx"
    export CPU_SUPPORTS_FMA="$supports_fma"
    export CPU_SUPPORTS_F16C="$supports_f16c"
    export CPU_SUPPORTS_SSE4="$supports_sse4"
    
    # 设置CMAKE_ARGS中的LLAMA参数
    export LLAMA_CMAKE_ARGS="$llama_args"
    
    # 输出检测摘要
    log_info "CPU指令集检测摘要: AMX=$supports_amx, AVX512=$supports_avx512, AVX2=$supports_avx2, AVX=$supports_avx, FMA=$supports_fma, F16C=$supports_f16c, SSE4=$supports_sse4"
    
    return 0
}

# 设置CUDA相关环境变量
setup_cuda_env_vars() {
    log_info "设置CUDA相关环境变量..."
    
    # 检查CUDA是否可用
    if command -v nvcc &> /dev/null; then
        # 获取CUDA安装路径
        local cuda_home=$(dirname $(dirname $(which nvcc)))
        export CUDA_HOME="$cuda_home"
        export CUDA_ROOT="$cuda_home"
        
        # 设置CUDA编译器
        export CUDACXX="$cuda_home/bin/nvcc"
        
        # 获取CUDA版本
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        if [[ -n "$cuda_version" ]]; then
            export CUDA_VERSION="$cuda_version"
            log_info "检测到CUDA版本: $cuda_version"
        fi
        
        # 设置CUDA架构（如果已检测到GPU）
        if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
            export CMAKE_CUDA_ARCHITECTURES="$TORCH_CUDA_ARCH_LIST"
            log_info "设置CUDA架构: $TORCH_CUDA_ARCH_LIST"
        else
            # 默认设置常用的CUDA架构
            export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
            export CMAKE_CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0"
            log_info "使用默认CUDA架构: 8.0;8.6;8.9;9.0"
        fi
        
        # 启用CUDA支持
        export KTRANSFORMERS_USE_CUDA="ON"
        
        log_info "CUDA环境变量设置完成"
        log_info "  CUDA_HOME: $CUDA_HOME"
        log_info "  CUDACXX: $CUDACXX"
        log_info "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
        log_info "  CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
    else
        log_warn "未检测到CUDA，禁用CUDA支持"
        export KTRANSFORMERS_USE_CUDA="OFF"
    fi
}

# 设置编译器环境变量
setup_compiler_env_vars() {
    log_info "设置编译器环境变量..."
    
    # 设置C/C++编译器（根据系统当前版本决定，不优先推荐特定版本）
    if [[ -n "$CC" && -n "$CXX" ]]; then
        # 如果已经检测到系统编译器，直接使用
        log_info "使用系统检测的编译器: CC=$CC, CXX=$CXX"
    elif command -v gcc-13 &> /dev/null; then
        export CC="gcc-13"
        export CXX="g++-13"
        log_info "设置C编译器: gcc-13"
        log_info "设置C++编译器: g++-13"
    elif command -v gcc-12 &> /dev/null; then
        export CC="gcc-12"
        export CXX="g++-12"
        log_info "设置C编译器: gcc-12"
        log_info "设置C++编译器: g++-12"
    elif command -v gcc-11 &> /dev/null; then
        export CC="gcc-11"
        export CXX="g++-11"
        log_info "设置C编译器: gcc-11"
        log_info "设置C++编译器: g++-11"
    elif command -v gcc &> /dev/null; then
        export CC="gcc"
        export CXX="g++"
        log_info "设置C编译器: gcc"
        log_info "设置C++编译器: g++"
    fi
    
    # 设置编译标志
    export CFLAGS="-O3 -march=native"
    export CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17"
    
    # 设置CMake构建类型
    export CMAKE_BUILD_TYPE="Release"
    
    log_info "编译器环境变量设置完成"
    log_info "  CC: ${CC:-未设置}"
    log_info "  CXX: ${CXX:-未设置}"
    log_info "  CFLAGS: $CFLAGS"
    log_info "  CXXFLAGS: $CXXFLAGS"
    log_info "  CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE"
}

# 设置性能优化环境变量
setup_performance_env_vars() {
    log_info "设置性能优化环境变量..."
    
    # 设置NUMA支持
    if [[ "${USE_NUMA:-0}" == "1" ]]; then
        export USE_NUMA="1"
        export KTRANSFORMERS_USE_NUMA="1"
        log_info "启用NUMA优化"
    else
        export USE_NUMA="0"
        export KTRANSFORMERS_USE_NUMA="0"
        log_info "禁用NUMA优化"
    fi
    
    # 设置负载均衡服务
    export USE_BALANCE_SERVE="1"
    export KTRANSFORMERS_USE_BALANCE_SERVE="1"
    
    # 设置线程数（基于CPU核心数）
    local cpu_cores=$(nproc)
    export OMP_NUM_THREADS="$cpu_cores"
    export MKL_NUM_THREADS="$cpu_cores"
    export NUMEXPR_NUM_THREADS="$cpu_cores"
    
    # 设置CUDA调试模式（生产环境建议设置为0）
    export CUDA_LAUNCH_BLOCKING="0"
    
    log_info "性能优化环境变量设置完成"
    log_info "  USE_NUMA: $USE_NUMA"
    log_info "  USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    log_info "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
    log_info "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
    log_info "  NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
}

# 设置Intel CPU特定优化（仅适用于Intel CPU）
setup_intel_cpu_optimizations() {
    # 检查是否为Intel CPU
    if ! grep -q "Intel" /proc/cpuinfo; then
        log_info "非Intel CPU，跳过Intel特定优化"
        return 0
    fi
    
    log_info "设置Intel CPU特定优化..."
    
    # 设置Intel DNNL最大指令集
    if [[ "$CPU_SUPPORTS_AVX512" == "true" ]]; then
        export DNNL_MAX_CPU_ISA="AVX512"
        export MKL_ENABLE_INSTRUCTIONS="AVX512"
    elif [[ "$CPU_SUPPORTS_AVX2" == "true" ]]; then
        export DNNL_MAX_CPU_ISA="AVX2"
        export MKL_ENABLE_INSTRUCTIONS="AVX2"
    elif [[ "$CPU_SUPPORTS_AVX" == "true" ]]; then
        export DNNL_MAX_CPU_ISA="AVX"
        export MKL_ENABLE_INSTRUCTIONS="AVX"
    fi
    
    # 设置OneDNN浮点数学模式
    export ONEDNN_DEFAULT_FPMATH_MODE="BF16"
    
    log_info "Intel CPU优化设置完成"
    log_info "  DNNL_MAX_CPU_ISA: ${DNNL_MAX_CPU_ISA:-未设置}"
    log_info "  MKL_ENABLE_INSTRUCTIONS: ${MKL_ENABLE_INSTRUCTIONS:-未设置}"
    log_info "  ONEDNN_DEFAULT_FPMATH_MODE: $ONEDNN_DEFAULT_FPMATH_MODE"
}

# 构建完整的CMAKE_ARGS
build_cmake_args() {
    log_info "构建CMAKE_ARGS..."
    
    local cmake_args=""
    
    # 添加基础CMake参数
    cmake_args="-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    
    # 添加LLAMA相关参数
    if [[ -n "$LLAMA_CMAKE_ARGS" ]]; then
        cmake_args="$cmake_args $LLAMA_CMAKE_ARGS"
    fi
    
    # 添加CUDA相关参数
    if [[ "$KTRANSFORMERS_USE_CUDA" == "ON" ]]; then
        cmake_args="$cmake_args -DUSE_CUDA=ON"
        if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
            cmake_args="$cmake_args -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
        fi
    fi
    
    # 添加NUMA相关参数
    if [[ "$USE_NUMA" == "1" ]]; then
        cmake_args="$cmake_args -DUSE_NUMA=ON"
    fi
    
    # 导出CMAKE_ARGS
    export CMAKE_ARGS="$cmake_args"
    
    log_info "CMAKE_ARGS设置完成: $CMAKE_ARGS"
}

# 显示所有环境变量设置摘要
show_env_vars_summary() {
    log_info "=== 环境变量设置摘要 ==="
    
    # CUDA相关
    log_info "CUDA配置:"
    log_info "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-未设置}"
    log_info "  CUDA_HOME: ${CUDA_HOME:-未设置}"
    log_info "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES:-未设置}"
    log_info "  KTRANSFORMERS_USE_CUDA: ${KTRANSFORMERS_USE_CUDA:-未设置}"
    
    # 编译器配置
    log_info "编译器配置:"
    log_info "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE:-未设置}"
    log_info "  CC: ${CC:-未设置}"
    log_info "  CXX: ${CXX:-未设置}"
    log_info "  CUDACXX: ${CUDACXX:-未设置}"
    log_info "  CFLAGS: ${CFLAGS:-未设置}"
    log_info "  CXXFLAGS: ${CXXFLAGS:-未设置}"
    
    # 性能优化配置
    log_info "性能优化配置:"
    log_info "  CMAKE_ARGS: ${CMAKE_ARGS:-未设置}"
    log_info "  USE_NUMA: ${USE_NUMA:-未设置}"
    log_info "  USE_BALANCE_SERVE: ${USE_BALANCE_SERVE:-未设置}"
    
    # CPU指令集支持
    log_info "CPU指令集支持:"
    log_info "  AVX512: ${CPU_SUPPORTS_AVX512:-未检测}"
    log_info "  AVX2: ${CPU_SUPPORTS_AVX2:-未检测}"
    log_info "  AVX: ${CPU_SUPPORTS_AVX:-未检测}"
    log_info "  FMA: ${CPU_SUPPORTS_FMA:-未检测}"
    log_info "  F16C: ${CPU_SUPPORTS_F16C:-未检测}"
    
    log_info "=== 环境变量设置摘要结束 ==="
    
    # 输出确认信息
    echo "Environment variables set successfully"
}

# 统一清理环境变量
cleanup_environment_vars() {
    log_info "清理环境变量..."
    
    # CUDA相关
    unset TORCH_CUDA_ARCH_LIST
    unset CUDA_HOME
    unset CMAKE_CUDA_ARCHITECTURES
    unset KTRANSFORMERS_USE_CUDA
    unset CUDACXX
    
    # 编译器相关
    unset CC
    unset CXX
    unset CFLAGS
    unset CXXFLAGS
    unset CMAKE_BUILD_TYPE
    
    # 性能优化相关
    unset USE_NUMA
    unset KTRANSFORMERS_USE_NUMA
    unset USE_BALANCE_SERVE
    unset KTRANSFORMERS_USE_BALANCE_SERVE
    unset OMP_NUM_THREADS
    unset MKL_NUM_THREADS
    unset NUMEXPR_NUM_THREADS
    unset CUDA_LAUNCH_BLOCKING
    
    # 构建相关
    unset CMAKE_ARGS
    unset LLAMA_CMAKE_ARGS
    
    # Intel CPU优化相关
    unset DNNL_MAX_CPU_ISA
    unset MKL_ENABLE_INSTRUCTIONS
    unset ONEDNN_DEFAULT_FPMATH_MODE
    
    # 其他
    unset KTRANSFORMERS_FORCE_BUILD
    
    log_info "环境变量清理完成"
}

# 主函数 - 设置所有环境变量
setup_env_vars() {
    show_progress "设置环境变量"
    
    log_info "开始设置KTransformers环境变量..."
    
    # 0. 首先清理环境变量
    cleanup_environment_vars
    
    # 1. 检测CPU指令集并设置AVX相关参数
    detect_and_set_cpu_instructions
    
    # 2. 设置CUDA相关环境变量
    setup_cuda_env_vars
    
    # 3. 设置编译器环境变量
    setup_compiler_env_vars
    
    # 4. 设置性能优化环境变量
    setup_performance_env_vars
    
    # 5. 设置Intel CPU特定优化（如果适用）
    setup_intel_cpu_optimizations
    
    # 6. 构建完整的CMAKE_ARGS
    build_cmake_args
    
    # 7. 显示环境变量设置摘要
    show_env_vars_summary
    
    log_info "KTransformers环境变量设置完成"
    return 0
}

# 全局编译环境设置函数（用于解决兼容性问题）
setup_global_compile_environment() {
    log_info "设置全局编译环境以解决兼容性问题..."
    
    # 确保环境变量已设置
    if [[ -z "$CMAKE_ARGS" ]]; then
        log_warn "CMAKE_ARGS未设置，先调用setup_env_vars"
        setup_env_vars
    fi
    
    # 设置全局环境变量，确保所有编译过程都能使用
    if [[ -n "$CC" ]]; then
        export CC
        log_info "全局设置CC: $CC"
    fi
    
    if [[ -n "$CXX" ]]; then
        export CXX
        log_info "全局设置CXX: $CXX"
    fi
    
    if [[ -n "$CUDACXX" ]]; then
        export CUDACXX
        log_info "全局设置CUDACXX: $CUDACXX"
    fi
    
    # 设置PATH以确保编译器可以找到
    if [[ -n "$CUDA_HOME" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
    
    log_info "全局编译环境设置完成"
    return 0
}