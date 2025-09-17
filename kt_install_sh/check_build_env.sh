#!/bin/bash

# KTransformers 编译环境检查模块
# 包含编译环境检查功能

# 检查编译环境 - 基础检查
check_build_env() {
    show_progress "检查编译环境"
    
    # 检查是否在conda环境中
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_error "未检测到激活的conda环境，请先激活环境"
        return 1
    fi
    
    # NUMA配置已在check_system阶段完成，这里只显示当前配置
    if [[ "$USE_NUMA" == "1" ]]; then
        log_info "NUMA优化已启用 (USE_NUMA=1)"
    else
        log_info "NUMA优化已禁用 (USE_NUMA=${USE_NUMA:-0})"
    fi
    
    # 检查关键工具
    check_critical_build_tools
    
    # 检查GPU和CUDA环境
    check_gpu_cuda
    
    log_info "编译环境检查完成"
    return 0
}

# 检查关键编译工具
check_critical_build_tools() {
    log_info "检查关键编译工具..."
    local critical_tools=("gcc" "g++" "cmake" "ninja" "make" "git")
    local missing_tools=()
    
    for tool in "${critical_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warn "以下关键编译工具未安装: ${missing_tools[*]}"
        log_warn "可能影响后续编译步骤"
    else
        log_info "所有关键编译工具已安装"
        
        # 输出版本信息
        log_info "GCC版本: $(gcc --version | head -n 1)"
        log_info "CMake版本: $(cmake --version | head -n 1)"
    fi
}

# 检查GPU和CUDA环境
check_gpu_cuda() {
    log_info "检查GPU和CUDA环境..."
    local gpu_detected=false
    local arch_list=""
    
    # 检查NVIDIA驱动
    if command -v nvidia-smi &> /dev/null; then
        gpu_detected=true
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        log_info "NVIDIA驱动版本: $driver_version"
        
        # 显示GPU信息
        log_info "GPU信息:"
        # 使用临时文件存储nvidia-smi输出，避免在while循环中丢失变量修改
        local temp_file=$(mktemp)
        
        # 添加超时和错误检查
        if ! timeout 30s nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader > "$temp_file" 2>/dev/null; then
            log_warn "nvidia-smi查询GPU信息失败或超时，跳过GPU信息收集"
            rm -f "$temp_file"
            return 0
        fi
        
        local gpu_count=0
        declare -a gpu_names=()
        declare -a gpu_arches=()
        local max_lines=16  # 限制最大处理行数，防止无限循环
        
        while read -r line && [ $gpu_count -lt $max_lines ]; do
            log_info "  $line"
            # 提取GPU名称和计算能力
            local gpu_name=$(echo "$line" | awk -F', ' '{print $1}')
            local compute_cap=$(echo "$line" | awk -F', ' '{print $3}')
            
            gpu_names+=("$gpu_name")
            gpu_count=$((gpu_count + 1))
            
            if [[ -n "$compute_cap" ]]; then
                # 确保计算能力格式正确
                if [[ "$compute_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
                    gpu_arches+=("$compute_cap")
                    
                    # 添加到架构列表
                    if [[ -z "$arch_list" ]]; then
                        arch_list="${compute_cap}"
                    else
                        # 检查是否已经包含此计算能力
                        if [[ ! "$arch_list" =~ $compute_cap ]]; then
                            arch_list="${arch_list};${compute_cap}"
                        fi
                    fi
                else
                    gpu_arches+=("auto")
                    log_info "  警告: GPU计算能力格式无效: '$compute_cap'"
                fi
            else
                gpu_arches+=("auto")
                log_info "  警告: 无法获取GPU计算能力"
            fi
        done < "$temp_file"
        rm -f "$temp_file"
        
        # 如果检测到多个GPU，不自动设置TORCH_CUDA_ARCH_LIST，留给用户选择
        if [[ $gpu_count -gt 1 ]]; then
            log_info "检测到 $gpu_count 个GPU，将在用户配置阶段提供选择"
            # 导出GPU信息供用户交互使用
            export DETECTED_GPU_COUNT="$gpu_count"
            export DETECTED_GPU_NAMES="${gpu_names[*]}"
            export DETECTED_GPU_ARCHES="${gpu_arches[*]}"
        elif [[ $gpu_count -eq 1 ]]; then
            # 单个GPU，自动设置
            if [[ -n "$arch_list" ]]; then
                log_info "检测到单个GPU: ${gpu_names[0]}"
                log_info "自动设置TORCH_CUDA_ARCH_LIST=${arch_list}"
                export TORCH_CUDA_ARCH_LIST="${arch_list}"
            fi
        fi
    else
        log_info "未检测到NVIDIA GPU"
    fi
    
    # 检查AMD GPU (ROCm)
    if command -v rocm-smi &> /dev/null; then
        gpu_detected=true
        log_info "检测到AMD GPU (ROCm)"
        
        # 显示AMD GPU信息
        log_info "AMD GPU信息:"
        rocm-smi --showproductname --showmeminfo vram | grep -v "===" | grep -v "^$" | while read -r line; do
            log_info "  $line"
        done
        
        # 设置ROCm相关环境变量
        log_info "设置ROCm环境变量"
        export HIP_VISIBLE_DEVICES=0,1,2,3
        export PYTORCH_ROCM_ARCH="gfx90a;gfx908;gfx906;gfx1030;gfx1100"
    fi
    
    if ! $gpu_detected; then
        log_warn "未检测到支持的GPU (NVIDIA或AMD)"
    fi
    
    # 检查CUDA运行时
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        log_info "CUDA Toolkit版本: $cuda_version"
    else
        if $gpu_detected; then
            log_warn "未检测到nvcc命令，可能未安装CUDA Toolkit"
        fi
    fi
    
    # 检测CPU架构并设置优化
    check_cpu_arch
    
    # 验证KTransformers环境变量设置
    validate_ktransformers_env_vars
}

# 验证KTransformers环境变量设置
validate_ktransformers_env_vars() {
    log_info "验证KTransformers环境变量设置..."
    
    # 验证KTRANSFORMERS_USE_CUDA设置
    if [[ -n "$KTRANSFORMERS_USE_CUDA" ]]; then
        if [[ "$KTRANSFORMERS_USE_CUDA" == "ON" ]]; then
            log_info "✓ KTRANSFORMERS_USE_CUDA: $KTRANSFORMERS_USE_CUDA (CUDA支持已启用)"
            
            # 检查CUDA相关环境变量
            if command -v nvcc &> /dev/null; then
                log_info "✓ CUDA Toolkit已安装"
            else
                log_warn "⚠ KTRANSFORMERS_USE_CUDA=ON 但未检测到nvcc命令"
            fi
        else
            log_info "✓ KTRANSFORMERS_USE_CUDA: $KTRANSFORMERS_USE_CUDA (CPU模式)"
        fi
    fi
    
    # 验证CMAKE_BUILD_TYPE
    if [[ -n "$CMAKE_BUILD_TYPE" ]]; then
        if [[ "$CMAKE_BUILD_TYPE" == "Release" || "$CMAKE_BUILD_TYPE" == "Debug" || "$CMAKE_BUILD_TYPE" == "RelWithDebInfo" ]]; then
            log_info "✓ CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE"
        else
            log_warn "⚠ CMAKE_BUILD_TYPE值可能无效: $CMAKE_BUILD_TYPE"
        fi
    fi
    
    # 验证TORCH_CUDA_ARCH_LIST
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        log_info "✓ TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
        
        # 验证CMAKE_CUDA_ARCHITECTURES是否与TORCH_CUDA_ARCH_LIST一致
        if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
            log_info "✓ CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
        fi
    fi
    
    # 验证NUMA设置
    if [[ -n "$USE_NUMA" ]]; then
        if [[ "$USE_NUMA" == "1" ]]; then
            if command -v numactl &> /dev/null; then
                log_info "✓ USE_NUMA: $USE_NUMA (NUMA支持已启用)"
            else
                log_warn "⚠ USE_NUMA=1 但numactl未安装"
            fi
        else
            log_info "✓ USE_NUMA: $USE_NUMA (NUMA支持已禁用)"
        fi
    fi
    
    # 验证负载均衡设置
    if [[ -n "$USE_BALANCE_SERVE" ]]; then
        log_info "✓ USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    fi
    
    # 验证编译器设置
    if [[ -n "$CC" ]]; then
        if command -v "$CC" &> /dev/null; then
            log_info "✓ C编译器: $CC ($(${CC} --version | head -n 1))"
        else
            log_warn "⚠ C编译器不可用: $CC"
        fi
    fi
    
    if [[ -n "$CXX" ]]; then
        if command -v "$CXX" &> /dev/null; then
            log_info "✓ C++编译器: $CXX ($(${CXX} --version | head -n 1))"
        else
            log_warn "⚠ C++编译器不可用: $CXX"
        fi
    fi
    
    if [[ -n "$CUDACXX" ]]; then
        if command -v "$CUDACXX" &> /dev/null; then
            log_info "✓ CUDA编译器: $CUDACXX ($(${CUDACXX} --version | grep "release" | head -n 1))"
        else
            log_warn "⚠ CUDA编译器不可用: $CUDACXX"
        fi
    fi
    
    # 验证高级设置（如果启用）
    if [[ -n "$ENABLE_THREAD_OPT" && "$ENABLE_THREAD_OPT" == "y" ]]; then
        if [[ -n "$OMP_NUM_THREADS" ]]; then
            log_info "✓ OpenMP线程数: $OMP_NUM_THREADS"
        fi
        
        if [[ -n "$MKL_NUM_THREADS" ]]; then
            log_info "✓ MKL线程数: $MKL_NUM_THREADS"
        fi
        
        if [[ -n "$NUMEXPR_NUM_THREADS" ]]; then
            log_info "✓ NumExpr线程数: $NUMEXPR_NUM_THREADS"
        fi
    fi
    
    # 验证Intel CPU优化设置（如果启用）
    if [[ -n "$ENABLE_CPU_ARCH_OPT" && "$ENABLE_CPU_ARCH_OPT" == "y" ]] && grep -q "Intel" /proc/cpuinfo; then
        if [[ -n "$DNNL_MAX_CPU_ISA" ]]; then
            log_info "✓ Intel CPU ISA优化: $DNNL_MAX_CPU_ISA"
        fi
        
        if [[ -n "$MKL_ENABLE_INSTRUCTIONS" ]]; then
            log_info "✓ MKL指令集优化: $MKL_ENABLE_INSTRUCTIONS"
        fi
        
        if [[ -n "$ONEDNN_DEFAULT_FPMATH_MODE" ]]; then
            log_info "✓ OneDNN浮点模式: $ONEDNN_DEFAULT_FPMATH_MODE"
        fi
    fi
    
    # 验证CUDA头文件路径（如果设置）
    if [[ -n "$NVTE_CUDA_INCLUDE_PATH" ]]; then
        if [[ -d "$NVTE_CUDA_INCLUDE_PATH" ]]; then
            log_info "✓ CUDA头文件路径: $NVTE_CUDA_INCLUDE_PATH"
        else
            log_warn "⚠ CUDA头文件路径不存在: $NVTE_CUDA_INCLUDE_PATH"
        fi
    fi
    
    # 验证CMAKE_ARGS
    if [[ -n "$CMAKE_ARGS" ]]; then
        log_info "✓ CMAKE_ARGS已设置"
        log_debug "CMAKE_ARGS内容: $CMAKE_ARGS"
    fi
    
    # 检查是否有不推荐的环境变量
    check_deprecated_env_vars
    
    log_info "KTransformers环境变量验证完成"
}

# 检查不推荐或过时的环境变量
check_deprecated_env_vars() {
    log_info "检查不推荐的环境变量..."
    
    # 检查可能冲突的环境变量
    local deprecated_vars=()
    
    # 检查是否设置了Intel GPU相关变量（脚本不支持Intel GPU）
    if [[ -n "$SYCL_CACHE_PERSISTENT" || -n "$ONEAPI_DEVICE_SELECTOR" || -n "$SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS" ]]; then
        log_warn "⚠ 检测到Intel GPU相关环境变量，但此脚本不支持Intel GPU"
        deprecated_vars+=("Intel GPU variables")
    fi
    
    # 检查是否设置了不必要的CUDA调试变量（除非用户明确启用）
    if [[ "$CUDA_LAUNCH_BLOCKING" == "1" && "$ENABLE_CUDA_DEBUG" != "y" ]]; then
        log_warn "⚠ CUDA_LAUNCH_BLOCKING=1 会降低性能，建议在生产环境中设置为0"
    fi
    
    if [[ ${#deprecated_vars[@]} -eq 0 ]]; then
        log_info "✓ 未发现不推荐的环境变量"
    fi
}

# CPU架构检测已在check_system阶段完成，这里只使用已设置的值
check_cpu_arch() {
    log_info "使用已检测的CPU架构配置..."
    
    # 使用check_system阶段已设置的CPU_ARCH_OPT值
    if [[ -n "$CPU_ARCH_OPT" ]]; then
        export KT_CPU_ARCH="$CPU_ARCH_OPT"
        log_info "CPU架构优化: $KT_CPU_ARCH"
    else
        export KT_CPU_ARCH="default"
        log_info "使用默认CPU优化"
    fi
}