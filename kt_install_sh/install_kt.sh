#!/bin/bash

# KTransformers 核心安装模块
# 包含KTransformers核心组件的克隆和安装

###############################
# 用户交互函数（调用user_interaction.sh）
###############################

# 询问KT安装目录
ask_kt_install_directory() {
    # 如果KT_ROOT已经设置（来自user_interaction.sh），直接使用
    if [[ -n "${KT_ROOT:-}" ]]; then
        log_info "使用已配置的KT_ROOT: $KT_ROOT"
        return 0
    fi
    
    # 默认使用项目根目录（没有kt_install_sh/kt子目录）
    local project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    KT_ROOT="$project_root"
    
    log_info "设置KT_ROOT: $KT_ROOT"
}

# 简单的询问是否卸载重新安装
ask_user_reinstall() {
    # 在自动化环境中，直接返回继续安装
    if [[ "${AUTO_INSTALL:-0}" == "1" ]]; then
        log_info "自动化模式：将自动卸载并重新安装"
        return 0
    fi
    
    # 交互模式询问用户
    echo
    echo -e "${YELLOW}检测到KTransformers已安装。${NC}"
    echo -e "${GREEN}[y]${NC} 是，卸载并重新安装 ${BLUE}(推荐)${NC}"
    echo -e "${RED}[n]${NC} 否，保持当前安装"
    echo
    read -p "请选择 [y/n]: " choice
    
    case "$choice" in
        [Yy]|[Yy][Ee][Ss])
            log_info "用户选择卸载并重新安装"
            return 0
            ;;
        [Nn]|[Nn][Oo]|*)
            log_info "用户选择保持当前安装"
            return 1
            ;;
    esac
}

###############################
# 变量设置函数部分
###############################
    log_info "检查和配置编译环境变量..."
    
    # 检查和配置CUDA环境
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        export CUDA_VERSION="$cuda_version"
        log_info "检测到CUDA版本: $CUDA_VERSION"
        
        # 设置CUDA_HOME
        if [[ -z "$CUDA_HOME" ]]; then
            export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
            log_info "设置CUDA_HOME: $CUDA_HOME"
        fi
    else
        log_warn "未检测到nvcc命令，CUDA可能未正确安装"
    fi
    
    # 检查和配置TORCH_CUDA_ARCH_LIST
    if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
        # 如果未设置，尝试从GPU检测结果获取
        if [[ -n "$DETECTED_GPU_ARCHES" ]]; then
            # 处理检测到的GPU架构，去除不必要的后缀
            local arch_list=""
            for arch in $DETECTED_GPU_ARCHES; do
                # 去除+RTX等后缀，只保留数字版本
                local clean_arch=$(echo "$arch" | sed 's/+.*$//' | grep -E '^[0-9]+\.[0-9]+$')
                if [[ -n "$clean_arch" ]]; then
                    if [[ -z "$arch_list" ]]; then
                        arch_list="$clean_arch"
                    else
                        # 检查是否已经包含此架构
                        if [[ ! "$arch_list" =~ $clean_arch ]]; then
                            arch_list="${arch_list};${clean_arch}"
                        fi
                    fi
                fi
            done
            
            if [[ -n "$arch_list" ]]; then
                export TORCH_CUDA_ARCH_LIST="${arch_list}"
                log_info "自动设置TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
            fi
        fi
        
        # 如果仍然未设置，使用默认值
        if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
            export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
            log_info "使用默认TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
        fi
    else
        # 清理现有的TORCH_CUDA_ARCH_LIST，去除不必要的后缀
        local cleaned_list=$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/+RTX//g')
        if [[ "$cleaned_list" != "$TORCH_CUDA_ARCH_LIST" ]]; then
            export TORCH_CUDA_ARCH_LIST="$cleaned_list"
            log_info "清理TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
        fi
    fi
    
    # 设置CMAKE_CUDA_ARCHITECTURES与TORCH_CUDA_ARCH_LIST保持一致
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        local cmake_arches=$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/+PTX//g' | sed 's/;/ /g')
        export CMAKE_CUDA_ARCHITECTURES="$cmake_arches"
        log_info "设置CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
    fi
    
    # 验证架构版本格式
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        local invalid_format=false
        IFS=';' read -ra ARCH_ARRAY <<< "$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/+PTX//')"
        for arch in "${ARCH_ARRAY[@]}"; do
            if [[ ! "$arch" =~ ^[0-9]+\.[0-9]+$ ]]; then
                log_warn "检测到无效的架构版本格式: $arch"
                invalid_format=true
            fi
        done
        
        if $invalid_format; then
            log_warn "TORCH_CUDA_ARCH_LIST包含无效格式，建议检查设置"
        fi
    fi

# 输出所有环境变量配置（仅显示，不可执行）
output_env_vars() {
    local phase="$1"  # "pre" 或 "post"
    
    echo ""
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${GREEN}[INFO]    环境变量配置 ($phase)${NC}"
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo ""
    
    echo -e "${BLUE}CUDA相关环境变量:${NC}"
    echo -e "  CUDA_VERSION: ${YELLOW}${CUDA_VERSION:-未设置}${NC}"
    echo -e "  CUDA_HOME: ${YELLOW}${CUDA_HOME:-未设置}${NC}"
    echo -e "  TORCH_CUDA_ARCH_LIST: ${YELLOW}${TORCH_CUDA_ARCH_LIST:-未设置}${NC}"
    echo -e "  CMAKE_CUDA_ARCHITECTURES: ${YELLOW}${CMAKE_CUDA_ARCHITECTURES:-未设置}${NC}"
    echo -e "  KTRANSFORMERS_USE_CUDA: ${YELLOW}${KTRANSFORMERS_USE_CUDA:-未设置}${NC}"
    echo ""
    
    echo -e "${BLUE}编译相关环境变量:${NC}"
    echo -e "  CMAKE_BUILD_TYPE: ${YELLOW}${CMAKE_BUILD_TYPE:-未设置}${NC}"
    echo -e "  CC: ${YELLOW}${CC:-未设置}${NC}"
    echo -e "  CXX: ${YELLOW}${CXX:-未设置}${NC}"
    echo -e "  CUDACXX: ${YELLOW}${CUDACXX:-未设置}${NC}"
    echo -e "  CFLAGS: ${YELLOW}${CFLAGS:-未设置}${NC}"
    echo -e "  CXXFLAGS: ${YELLOW}${CXXFLAGS:-未设置}${NC}"
    echo ""
    
    echo -e "${BLUE}CPU优化相关环境变量:${NC}"
    echo -e "  CPU_ARCH_OPT: ${YELLOW}${CPU_ARCH_OPT:-未设置}${NC}"
    echo -e "  DNNL_MAX_CPU_ISA: ${YELLOW}${DNNL_MAX_CPU_ISA:-未设置}${NC}"
    echo -e "  MKL_ENABLE_INSTRUCTIONS: ${YELLOW}${MKL_ENABLE_INSTRUCTIONS:-未设置}${NC}"
    echo ""
    
    echo -e "${BLUE}线程相关环境变量:${NC}"
    echo -e "  OMP_NUM_THREADS: ${YELLOW}${OMP_NUM_THREADS:-未设置}${NC}"
    echo -e "  MKL_NUM_THREADS: ${YELLOW}${MKL_NUM_THREADS:-未设置}${NC}"
    echo -e "  NUMEXPR_NUM_THREADS: ${YELLOW}${NUMEXPR_NUM_THREADS:-未设置}${NC}"
    echo ""
    
    echo -e "${BLUE}其他环境变量:${NC}"
    echo -e "  USE_NUMA: ${YELLOW}${USE_NUMA:-未设置}${NC}"
    echo -e "  USE_BALANCE_SERVE: ${YELLOW}${USE_BALANCE_SERVE:-未设置}${NC}"
    echo -e "  CUDA_LAUNCH_BLOCKING: ${YELLOW}${CUDA_LAUNCH_BLOCKING:-未设置}${NC}"
    echo ""
}

# 生成可直接复制执行的环境变量配置命令（基于系统实际检测）
generate_env_commands() {
    echo ""
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${GREEN}[INFO]    可直接复制执行的环境变量配置命令${NC}"
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo ""
    
    # 卸载旧版本命令
    echo -e "${BLUE}# 卸载旧版本${NC}"
    echo "pip uninstall ktransformers -y"
    echo ""
    
    # 清理构建目录命令
    echo -e "${BLUE}# 清理构建目录${NC}"
    echo "rm -rf build"
    echo "rm -rf *.egg-info"
    echo "rm -rf csrc/build"
    echo "rm -rf csrc/ktransformers_ext/build"
    echo "rm -rf csrc/ktransformers_ext/cuda/build"
    echo "rm -rf csrc/ktransformers_ext/cuda/dist"
    echo "rm -rf csrc/ktransformers_ext/cuda/*.egg-info"
    echo "rm -rf ~/.ktransformers"
    echo ""
    
    # 清理环境变量命令
    echo -e "${BLUE}# 清理旧环境变量${NC}"
    echo "unset USE_NUMA"
    echo "unset TORCH_CUDA_ARCH_LIST"
    echo "unset CUDA_HOME"
    echo "unset CMAKE_CUDA_ARCHITECTURES"
    echo "unset KTRANSFORMERS_USE_CUDA"
    echo "unset CMAKE_BUILD_TYPE"
    echo "unset CC"
    echo "unset CXX"
    echo "unset CUDACXX"
    echo "unset CFLAGS"
    echo "unset CXXFLAGS"
    echo "unset CMAKE_ARGS"
    echo "unset USE_BALANCE_SERVE"
    echo ""
    
    # 设置并行编译参数（基于实际CPU核心数）
    local cpu_cores=$(nproc)
    echo -e "${BLUE}# 设置并行编译参数（系统检测：${cpu_cores}核心）${NC}"
    echo "export MAKEFLAGS=\"-j${cpu_cores}\"                    # 使用所有CPU核心"
    echo "export CMAKE_BUILD_PARALLEL_LEVEL=${cpu_cores}      # CMake并行编译"
    echo "export MAX_JOBS=${cpu_cores}                        # PyTorch并行编译"
    echo ""
    
    # 环境信息（不输出验证信息）
    echo "# CPU核心数: $(nproc)"
    echo "# 编译线程数: \$MAKEFLAGS"
    echo ""
    
    # NUMA设置（基于实际检测）
    echo -e "${BLUE}# NUMA设置（系统检测）${NC}"
    echo "export USE_NUMA=0  # 实际系统有${DETECTED_NUMA_NODES:-未知}个NUMA节点"
    echo ""
    
    # GPU计算能力设置（基于实际检测）
    if [[ -n "$DETECTED_GPU_ARCHES" ]]; then
        echo -e "${BLUE}# GPU计算能力设置（系统检测）${NC}"
        local gpu_info=""
        if [[ -n "$DETECTED_GPU_NAME" ]]; then
            gpu_info="实际检测：$DETECTED_GPU_NAME"
        fi
        
        # 清理架构列表格式
        local clean_arch_list=$(echo "$DETECTED_GPU_ARCHES" | sed 's/+PTX//g' | sed 's/+RTX//g')
        
        echo "# ${gpu_info}"
        echo "export CUDA_ARCH_LIST=\"${clean_arch_list}\""
        echo "export TORCH_CUDA_ARCH_LIST=\"${clean_arch_list}\""
        
        # 生成CMAKE格式（去除小数点）
        local cmake_arches=""
        for arch in $(echo "$clean_arch_list" | tr ';' ' '); do
            local cmake_arch=$(echo "$arch" | sed 's/\.//')
            if [[ -n "$cmake_arches" ]]; then
                cmake_arches="${cmake_arches};${cmake_arch}"
            else
                cmake_arches="${cmake_arch}"
            fi
        done
        echo "export CMAKE_CUDA_ARCHITECTURES=\"${cmake_arches}\""
    else
        echo -e "${BLUE}# GPU计算能力设置（默认）${NC}"
        echo "export CUDA_ARCH_LIST=\"8.6\""
        echo "export TORCH_CUDA_ARCH_LIST=\"8.6\""
        echo "export CMAKE_CUDA_ARCHITECTURES=\"86\""
    fi
    echo ""
    
    # CUDA路径设置（基于实际检测）
    if [[ -n "$CUDA_HOME" ]]; then
        echo -e "${BLUE}# CUDA路径设置（系统检测）${NC}"
        echo "export CUDA_HOME=${CUDA_HOME}  # 实际路径确认正确"
        echo "export CUDACXX=${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
    elif [[ -d "/usr/local/cuda" ]]; then
        echo -e "${BLUE}# CUDA路径设置（默认）${NC}"
        echo "export CUDA_HOME=/usr/local/cuda"
        echo "export CUDACXX=/usr/local/cuda/bin/nvcc"
    fi
    echo ""
    
    # 编译器设置（基于实际检测，不推荐使用特定版本）
    if [[ -n "$DETECTED_OS" && -n "$CC" ]]; then
        echo -e "${BLUE}# 编译器设置（系统检测）${NC}"
        echo "# 实际检测：${DETECTED_OS}, ${CC} $(gcc --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')"
        echo "export CC=${CC}"
        echo "export CXX=${CXX}"
    elif command -v gcc-13 &> /dev/null; then
        echo -e "${BLUE}# 编译器设置（系统当前版本）${NC}"
        echo "export CC=gcc-13"
        echo "export CXX=g++-13"
    elif command -v gcc-12 &> /dev/null; then
        echo -e "${BLUE}# 编译器设置（系统当前版本）${NC}"
        echo "export CC=gcc-12"
        echo "export CXX=g++-12"
    else
        echo -e "${BLUE}# 编译器设置（系统默认）${NC}"
        echo "export CC=gcc"
        echo "export CXX=g++"
    fi
    echo ""
    
    # 构建类型设置
    echo -e "${BLUE}# 构建类型设置${NC}"
    echo "export KTRANSFORMERS_USE_CUDA=ON"
    echo "export CMAKE_BUILD_TYPE=Release"
    echo ""
    
    # Ubuntu 24.04 glibc兼容性修复
    if grep -q "Ubuntu 24" /etc/os-release 2>/dev/null; then
        echo -e "${BLUE}# Ubuntu 24.04 glibc兼容性修复${NC}"
        echo "export CFLAGS=\"-O3 -march=native -D_Float128=__float128\""
        echo "export CXXFLAGS=\"-O3 -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17 -D_Float128=__float128\""
        echo "export KTRANSFORMERS_FORCE_BUILD=TRUE"
    else
        echo -e "${BLUE}# 编译优化标志${NC}"
        echo "export CFLAGS=\"-O3 -march=native\""
        echo "export CXXFLAGS=\"-O3 -march=native -std=c++17\""
    fi
    echo ""
    
    # 完整的CMAKE_ARGS（基于实际检测生成）
    echo -e "${BLUE}# 完整的CMAKE_ARGS（基于系统检测生成）${NC}"
    local cmake_args="-DKTRANSFORMERS_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
    
    if [[ -n "$DETECTED_GPU_ARCHES" ]]; then
        local clean_arch_list=$(echo "$DETECTED_GPU_ARCHES" | sed 's/+PTX//g' | sed 's/+RTX//g')
        local cmake_arches=""
        for arch in $(echo "$clean_arch_list" | tr ';' ' '); do
            local cmake_arch=$(echo "$arch" | sed 's/\.//')
            if [[ -n "$cmake_arches" ]]; then
                cmake_arches="${cmake_arches};${cmake_arch}"
            else
                cmake_arches="${cmake_arch}"
            fi
        done
        cmake_args="${cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${cmake_arches}"
    fi
    
    cmake_args="${cmake_args} -DUSE_BALANCE_SERVE=1 -DUSE_NUMA=0"
    
    if [[ -n "$CC" ]]; then
        cmake_args="${cmake_args} -DCMAKE_C_COMPILER=${CC}"
    fi
    if [[ -n "$CXX" ]]; then
        cmake_args="${cmake_args} -DCMAKE_CXX_COMPILER=${CXX}"
    fi
    if [[ -n "$CUDACXX" ]]; then
        cmake_args="${cmake_args} -DCMAKE_CUDA_COMPILER=${CUDACXX}"
    fi
    
    cmake_args="${cmake_args} -DLLAMA_NATIVE=ON"
    
    if grep -q "Ubuntu 24" /etc/os-release 2>/dev/null; then
        cmake_args="${cmake_args} -D_GLIBCXX_USE_CXX11_ABI=1"
    fi
    
    echo "export CMAKE_ARGS=\"${cmake_args}\""
    echo ""
    
    # 平衡服务设置
    echo -e "${BLUE}# 平衡服务设置${NC}"
    echo "export USE_BALANCE_SERVE=1"
    echo ""
    
    # 设置信息（不输出验证信息）
    echo "# CUDA_HOME: \$CUDA_HOME"
    echo "# TORCH_CUDA_ARCH_LIST: \$TORCH_CUDA_ARCH_LIST"
    echo "# CMAKE_CUDA_ARCHITECTURES: \$CMAKE_CUDA_ARCHITECTURES"
    echo "# CC: \$CC"
    echo "# CXX: \$CXX"
    echo "# CUDACXX: \$CUDACXX"
    echo ""
    
    # 最终安装命令
    echo -e "${BLUE}# 执行安装${NC}"
    echo "USE_BALANCE_SERVE=1 USE_NUMA=0 bash install.sh"
    echo ""
}

# 编译时变量设置函数 - 设置编译优化相关环境变量
set_common_variables() {
    log_info "清理旧环境并设置编译环境变量..."
    
    # 使用统一的环境变量清理函数（如果存在）
    if declare -f cleanup_environment_vars > /dev/null; then
        cleanup_environment_vars
    else
        # 清理旧环境变量（备用清理逻辑）
        unset USE_NUMA
        unset TORCH_CUDA_ARCH_LIST
        unset CUDA_HOME
        unset CMAKE_CUDA_ARCHITECTURES
        unset KTRANSFORMERS_USE_CUDA
        unset CMAKE_BUILD_TYPE
        unset CC
        unset CXX
        unset CUDACXX
        unset CFLAGS
        unset CXXFLAGS
        unset CMAKE_ARGS
        unset USE_BALANCE_SERVE
        unset MAKEFLAGS
        unset CMAKE_BUILD_PARALLEL_LEVEL
        unset KTRANSFORMERS_FORCE_BUILD
        unset OMP_NUM_THREADS
        unset MKL_NUM_THREADS
        unset NUMEXPR_NUM_THREADS
        unset CUDA_LAUNCH_BLOCKING
    fi
    
    # 卸载旧版本（避免版本冲突）
    pip uninstall ktransformers -y 2>/dev/null || true
    
    # 清理构建目录
    rm -rf build 2>/dev/null || true
    rm -rf *.egg-info 2>/dev/null || true
    rm -rf csrc/build 2>/dev/null || true
    rm -rf csrc/ktransformers_ext/build 2>/dev/null || true
    rm -rf csrc/ktransformers_ext/cuda/build 2>/dev/null || true
    rm -rf csrc/ktransformers_ext/cuda/dist 2>/dev/null || true
    rm -rf csrc/ktransformers_ext/cuda/*.egg-info 2>/dev/null || true
    rm -rf ~/.ktransformers 2>/dev/null || true
    
    # 设置并行编译参数
    local cpu_cores=$(nproc)
    export MAKEFLAGS="-j${cpu_cores}"
    export CMAKE_BUILD_PARALLEL_LEVEL=${cpu_cores}
    
    # 首先检测系统环境
    detect_system_environment
    
    # 设置KTransformers官方推荐的CMake参数
    export KTRANSFORMERS_USE_CUDA="ON"
    log_info "启用CUDA支持: KTRANSFORMERS_USE_CUDA=ON"
    
    # 设置编译器优化标志
    if [[ -n "${COMPILER_OPTIMIZATION_LEVEL}" ]]; then
        log_info "使用用户指定的编译优化级别: ${COMPILER_OPTIMIZATION_LEVEL}"
        export CFLAGS="${COMPILER_OPTIMIZATION_LEVEL}"
        export CXXFLAGS="${COMPILER_OPTIMIZATION_LEVEL}"
    else
        # 默认使用-O3优化
        export CFLAGS="-O3 -march=native"
        export CXXFLAGS="-O3 -march=native -std=c++17"
        log_info "使用默认编译优化级别: -O3 -march=native"
    fi
    
    # Ubuntu 24.04 glibc兼容性修复
    if grep -q "Ubuntu 24" /etc/os-release 2>/dev/null; then
        export CFLAGS="${CFLAGS} -D_Float128=__float128"
        export CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=1 -D_Float128=__float128"
        log_info "应用Ubuntu 24.04兼容性修复"
    fi
    
    # 设置编译器路径（如果系统有多个版本）
    if command -v gcc-13 &> /dev/null; then
        export CC="gcc-13"
        export CXX="g++-13"
        log_info "使用GCC 13编译器"
    elif command -v gcc-12 &> /dev/null; then
        export CC="gcc-12"
        export CXX="g++-12"
        log_info "使用GCC 12编译器"
    elif command -v gcc-11 &> /dev/null; then
        export CC="gcc-11"
        export CXX="g++-11"
        log_info "使用GCC 11编译器"
    else
        log_info "使用系统默认GCC编译器"
    fi
    
    # 设置CUDA编译器路径
    if command -v nvcc &> /dev/null; then
        export CUDACXX="$(which nvcc)"
        log_info "设置CUDA编译器: $CUDACXX"
    fi
    
    # 设置CMake构建类型
    export CMAKE_BUILD_TYPE="Release"
    log_info "设置CMake构建类型: Release"
    
    # 设置编译并行度
    if [[ -n "${BUILD_PARALLEL_JOBS}" ]]; then
        export MAX_JOBS="${BUILD_PARALLEL_JOBS}"
        log_info "设置编译并行度: ${MAX_JOBS}"
    else
        # 默认使用CPU核心数的一半
        local cpu_cores=$(nproc)
        export MAX_JOBS=$((cpu_cores / 2))
        log_info "设置默认编译并行度: ${MAX_JOBS} (CPU核心数的一半)"
    fi
    
    # 获取GPU计算能力用于编译 - 优先使用用户选择的配置
    if [[ -n "$(command -v nvidia-smi)" ]]; then
        if [[ -n "$TORCH_CUDA_ARCH_LIST" && "$TORCH_CUDA_ARCH_LIST" != "auto" ]]; then
            log_info "使用用户选择的GPU架构配置: $TORCH_CUDA_ARCH_LIST"
            # 将TORCH_CUDA_ARCH_LIST转换为CMAKE_CUDA_ARCHITECTURES格式
            export CMAKE_CUDA_ARCHITECTURES="${TORCH_CUDA_ARCH_LIST//;/ }"
            export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES//+PTX/}"
            log_info "设置CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
        elif nvidia-smi --query-gpu=compute_cap --format=csv,noheader &>/dev/null; then
            # 如果没有用户配置，则自动检测
            local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
            log_info "检测到GPU计算能力: $compute_cap"
            export TORCH_CUDA_ARCH_LIST="$compute_cap"
            export CMAKE_CUDA_ARCHITECTURES="$compute_cap"
        else
            log_warn "无法获取GPU计算能力，使用默认CUDA架构"
            export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0"
            export CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
        fi
    else
        log_info "未检测到NVIDIA GPU，使用CPU模式"
        export KTRANSFORMERS_USE_CUDA="OFF"
    fi
    
    # NUMA支持已在check_system阶段检测并设置，这里使用已设置的值
    if [[ -z "$USE_NUMA" ]]; then
        export USE_NUMA=0
        log_info "USE_NUMA未设置，默认禁用NUMA支持"
    else
        log_info "使用已检测的NUMA配置: USE_NUMA=$USE_NUMA"
    fi
    
    # 构建CMAKE_ARGS
    local cmake_args_list=()
    cmake_args_list+=("-DKTRANSFORMERS_USE_CUDA=$KTRANSFORMERS_USE_CUDA")
    cmake_args_list+=("-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE")
    
    if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
        cmake_args_list+=("-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES")
    fi
    
    if [[ "$USE_NUMA" == "1" ]]; then
        cmake_args_list+=("-DUSE_NUMA=1")
    else
        cmake_args_list+=("-DUSE_NUMA=0")
    fi
    
    if [[ "$USE_BALANCE_SERVE" == "1" ]]; then
        cmake_args_list+=("-DUSE_BALANCE_SERVE=1")
    fi
    
    # 设置编译器路径到CMAKE_ARGS
    if [[ -n "$CC" ]]; then
        cmake_args_list+=("-DCMAKE_C_COMPILER=$CC")
    fi
    
    if [[ -n "$CXX" ]]; then
        cmake_args_list+=("-DCMAKE_CXX_COMPILER=$CXX")
    fi
    
    if [[ -n "$CUDACXX" ]]; then
        cmake_args_list+=("-DCMAKE_CUDA_COMPILER=$CUDACXX")
    fi
    
    # 启用LLAMA_NATIVE优化
    cmake_args_list+=("-DLLAMA_NATIVE=ON")
    
    # Ubuntu 24.04兼容性修复
    if [[ "$CXXFLAGS" == *"_GLIBCXX_USE_CXX11_ABI=1"* ]]; then
        cmake_args_list+=("-D_GLIBCXX_USE_CXX11_ABI=1")
    fi
    
    export CMAKE_ARGS="${cmake_args_list[*]}"
    log_info "设置CMAKE_ARGS: $CMAKE_ARGS"
    
    log_info "编译环境变量设置完成"
    
    # 输出编译配置摘要
    log_info "编译配置摘要:"
    log_info "  KTRANSFORMERS_USE_CUDA: ${KTRANSFORMERS_USE_CUDA}"
    log_info "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}"
    log_info "  编译优化级别: ${CFLAGS}"
    log_info "  编译并行度: ${MAX_JOBS}"
    log_info "  CPU架构优化: ${CPU_ARCH_OPT}"
    log_info "  NUMA支持: ${USE_NUMA}"
    log_info "  多并发支持: ${USE_BALANCE_SERVE}"
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        log_info "  GPU架构: ${TORCH_CUDA_ARCH_LIST}"
    fi
    if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
        log_info "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}"
    fi
    if [[ -n "$CMAKE_ARGS" ]]; then
        log_info "  CMAKE_ARGS: ${CMAKE_ARGS}"
    fi
}

# 官方最小配置函数 - 仅包含官方要求的基本设置




###############################
# 检查和验证函数部分
###############################

# 检查KTransformers安装状态并管理安装
check_and_manage_kt() {
    local action="$1"  # 可能的值: check, verify, uninstall
    
    case "$action" in
        "check")
            log_info "检查KTransformers是否已安装..."
            
            if pip list | grep -q "ktransformers"; then
                local kt_version=$(pip show ktransformers | grep Version | awk '{print $2}')
                log_info "检测到KTransformers已安装，版本: $kt_version"
                return 0
            else
                log_info "未检测到KTransformers"
                return 1
            fi
            ;;
            
        "verify")
            log_info "验证KTransformers功能..."
            
            # 1. 查看ktransformers安装版本
            log_info "查看ktransformers安装版本..."
            if ! pip show ktransformers; then
                log_error "无法显示ktransformers版本信息"
                return 1
            fi
            
            # 2. 验证ktransformers基本导入
            log_info "验证ktransformers基本导入..."
            if ! python -c "import ktransformers" 2>/dev/null; then
                log_error "ktransformers基本导入失败"
                return 1
            fi
            log_info "ktransformers基本导入验证通过"
            
            # 3. 验证ktransformers版本查询
            log_info "验证ktransformers版本查询..."
            if ! python -c "import ktransformers; print('KTransformers版本:', ktransformers.__version__)" 2>/dev/null; then
                log_error "ktransformers版本查询失败"
                return 1
            fi
            log_info "ktransformers版本查询验证通过"
            
            # 4. 验证local_chat模块导入
            log_info "验证local_chat模块导入..."
            if ! python -c "import ktransformers.local_chat" 2>/dev/null; then
                log_error "local_chat模块导入失败"
                return 1
            fi
            log_info "local_chat模块导入验证通过"
            
            # 5. 验证local_chat命令行工具可用性
            log_info "验证local_chat命令行工具可用性..."
            if ! python -m ktransformers.local_chat --help >/dev/null 2>&1; then
                log_error "local_chat命令行工具验证失败"
                return 1
            fi
            log_info "local_chat命令行工具验证通过"
            
            # 6. 验证NUMA支持
            log_info "验证NUMA支持..."
            if test -f /proc/self/numa_maps; then
                log_info "NUMA支持已启用"
            else
                log_info "NUMA未启用"
                # 如果用户配置了USE_NUMA=1但系统不支持NUMA，发出警告
                if [[ "$USE_NUMA" == "1" ]]; then
                    log_warn "用户配置启用了NUMA，但系统不支持NUMA"
                fi
            fi
            
            log_info "KTransformers功能验证完成，所有检查均通过"
            return 0
            ;;
            
        "uninstall")
            log_info "卸载KTransformers..."
            
            # 尝试卸载包
            pip uninstall -y ktransformers || log_warn "卸载KTransformers失败，继续安装"
            
            # 清理可能的临时文件和构建缓存
            if [ -d "$KT_ROOT" ]; then
                log_info "清理KTransformers构建文件..."
                find "$KT_ROOT" -name "*.so" -delete 2>/dev/null || true
                find "$KT_ROOT" -name "*.o" -delete 2>/dev/null || true
                find "$KT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
                rm -rf "$KT_ROOT/build/" 2>/dev/null || true
                rm -rf "$KT_ROOT/csrc/build/" 2>/dev/null || true
                rm -rf "$KT_ROOT/csrc/ktransformers_ext/build/" 2>/dev/null || true
                rm -rf "$KT_ROOT/csrc/ktransformers_ext/cuda/build/" 2>/dev/null || true
                rm -rf "$KT_ROOT/csrc/ktransformers_ext/cuda/dist/" 2>/dev/null || true
                rm -rf "$KT_ROOT/csrc/ktransformers_ext/cuda/"*.egg-info 2>/dev/null || true
                rm -rf "$KT_ROOT/"*.egg-info 2>/dev/null || true
            fi
            
            # 清理用户目录下的缓存
            rm -rf ~/.ktransformers 2>/dev/null || true
            
            # 清理pip缓存中的相关文件
            pip cache purge 2>/dev/null || true
            
            return 0
            ;;
            
        *)
            log_error "未知的操作: $action"
            return 1
            ;;
    esac
}

###############################
# 安装函数部分
###############################



# 检测系统环境并生成动态配置
detect_system_environment() {
    log_info "开始检测系统环境..."
    
    # 初始化检测结果变量
    local system_info=""
    
    # CPU信息检测
    local cpu_cores=$(nproc)
    local cpu_model=$(grep "model name" /proc/cpuinfo | head -n1 | cut -d':' -f2 | xargs)
    system_info="CPU核心数: $cpu_cores, 型号: $cpu_model"
    
    # NUMA节点检测
    local numa_nodes=$(lscpu | grep "NUMA node(s)" | awk '{print $3}' 2>/dev/null || echo "1")
    system_info="$system_info, NUMA节点: $numa_nodes"
    
    # GPU信息检测
    local gpu_name="未检测到"
    local compute_cap=""
    local cuda_version="未检测到"
    
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        if [[ -n "$gpu_info" ]]; then
            gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
            compute_cap=$(echo "$gpu_info" | cut -d',' -f2 | xargs)
            
            # CUDA版本检测
            if [[ -f "/usr/local/cuda/version.txt" ]]; then
                cuda_version=$(cat /usr/local/cuda/version.txt | cut -d' ' -f3)
            elif command -v nvcc &>/dev/null; then
                cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
            fi
        fi
    fi
    
    system_info="$system_info, GPU: $gpu_name"
    if [[ -n "$compute_cap" ]]; then
        system_info="$system_info, 计算能力: $compute_cap"
    fi
    system_info="$system_info, CUDA: $cuda_version"
    
    # 编译器检测
    local gcc_version="未检测到"
    if command -v gcc &>/dev/null; then
        gcc_version=$(gcc --version | head -n1 | awk '{print $3}')
    fi
    system_info="$system_info, GCC: $gcc_version"
    
    log_info "系统环境检测完成: $system_info"
    
    # 返回检测结果
    echo "$cpu_cores|$numa_nodes|$gpu_name|$compute_cap|$cuda_version|$gcc_version"
}

# 使用用户环境设置方案安装
install_with_user_env_settings() {
    log_info "使用用户环境设置方案安装KTransformers..."
    
    # 检查kt目录是否存在
    if [[ ! -d "kt" ]]; then
        log_error "kt目录不存在，无法进入kt目录"
        log_info "生成基于系统检测的可直接复制执行的环境变量配置命令:"
        generate_env_commands
        return 1
    fi
    
    # 进入kt目录
    cd kt || {
        log_error "无法进入kt目录"
        log_info "生成基于系统检测的可直接复制执行的环境变量配置命令:"
        generate_env_commands
        return 1
    }
    
    # 检测系统环境
    local env_info=$(detect_system_environment)
    IFS='|' read -r cpu_cores numa_nodes gpu_name compute_cap cuda_version gcc_version <<< "$env_info"
    
    # 清理旧环境
    log_info "清理旧环境..."
    pip uninstall ktransformers -y 2>/dev/null || log_warn "卸载旧版本失败或无旧版本"
    
    # 清理构建目录
    log_info "清理构建目录..."
    rm -rf build *.egg-info csrc/build csrc/ktransformers_ext/build
    rm -rf csrc/ktransformers_ext/cuda/build csrc/ktransformers_ext/cuda/dist
    rm -rf csrc/ktransformers_ext/cuda/*.egg-info ~/.ktransformers 2>/dev/null || true
    
    # 清理环境变量（用户方案）
    log_info "清理环境变量..."
    unset USE_NUMA TORCH_CUDA_ARCH_LIST CUDA_HOME CMAKE_CUDA_ARCHITECTURES
    unset KTRANSFORMERS_USE_CUDA CMAKE_BUILD_TYPE CC CXX CUDACXX
    unset CFLAGS CXXFLAGS CMAKE_ARGS USE_BALANCE_SERVE
    
    # 设置编译并行度
    log_info "设置编译并行度..."
    export MAKEFLAGS="-j$cpu_cores"
    export CMAKE_BUILD_PARALLEL_LEVEL=$cpu_cores
    export MAX_JOBS=$cpu_cores
    
    log_info "使用检测到的CPU核心数: $cpu_cores"
    
    # 根据检测结果设置环境变量
    log_info "根据系统检测结果设置环境变量..."
    
    # NUMA设置（根据实际检测的NUMA节点数）
    if [[ $numa_nodes -gt 1 ]]; then
        log_info "检测到 $numa_nodes 个NUMA节点，设置USE_NUMA=0"
        export USE_NUMA=0
    else
        export USE_NUMA=0
    fi
    
    # GPU和CUDA设置（根据实际检测）
    if [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        # 转换计算能力格式
        local arch_num=$(echo "$compute_cap" | sed 's/\.//')
        export CUDA_ARCH_LIST="$compute_cap"
        export TORCH_CUDA_ARCH_LIST="$compute_cap"
        export CMAKE_CUDA_ARCHITECTURES="$arch_num"
        
        log_info "检测到GPU: $gpu_name, 计算能力: $compute_cap"
        log_info "设置CUDA架构: $compute_cap -> $arch_num"
    else
        log_warn "未检测到GPU计算能力，使用CPU模式"
        export CUDA_ARCH_LIST=""
        export TORCH_CUDA_ARCH_LIST=""
        export CMAKE_CUDA_ARCHITECTURES=""
    fi
    
    # CUDA路径检测（根据实际CUDA版本）
    if [[ "$cuda_version" != "未检测到" && -n "$cuda_version" ]]; then
        # 尝试找到对应版本的CUDA路径
        local cuda_major_minor=$(echo "$cuda_version" | cut -d'.' -f1,2)
        local cuda_paths=(
            "/usr/local/cuda-$cuda_major_minor"
            "/usr/local/cuda"
            "/opt/cuda-$cuda_major_minor"
            "/opt/cuda"
        )
        
        for cuda_path in "${cuda_paths[@]}"; do
            if [[ -d "$cuda_path" && -f "$cuda_path/bin/nvcc" ]]; then
                export CUDA_HOME="$cuda_path"
                export CUDACXX="$cuda_path/bin/nvcc"
                log_info "检测到CUDA路径: $cuda_path"
                break
            fi
        done
        
        if [[ -z "$CUDA_HOME" ]]; then
            log_warn "未找到CUDA $cuda_version 的安装路径"
        fi
    else
        log_warn "未检测到CUDA版本"
    fi
    
    # 编译器检测（根据实际GCC版本）
    if [[ "$gcc_version" != "未检测到" && -n "$gcc_version" ]]; then
        local gcc_major=$(echo "$gcc_version" | cut -d'.' -f1)
        
        # 尝试使用检测到的GCC版本
        if command -v "gcc-$gcc_major" &>/dev/null; then
            export CC="gcc-$gcc_major"
            export CXX="g++-$gcc_major"
            log_info "使用检测到的GCC版本: gcc-$gcc_major"
        else
            # 回退到系统默认
            export CC=gcc
            export CXX=g++
            log_info "使用系统默认GCC: $gcc_version"
        fi
    else
        export CC=gcc
        export CXX=g++
        log_info "使用系统默认编译器"
    fi
    
    # 设置构建类型和优化标志
    if [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        export KTRANSFORMERS_USE_CUDA=ON
        log_info "启用CUDA支持"
    else
        export KTRANSFORMERS_USE_CUDA=OFF
        log_info "使用CPU模式"
    fi
    
    export CMAKE_BUILD_TYPE=Release
    export CFLAGS="-O3 -march=native -D_Float128=__float128"
    export CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17 -D_Float128=__float128"
    export KTRANSFORMERS_FORCE_BUILD=TRUE
    export USE_BALANCE_SERVE=1
    
    # 设置CMAKE_ARGS（根据检测结果动态生成）
    export CMAKE_ARGS="-DKTRANSFORMERS_USE_CUDA=$KTRANSFORMERS_USE_CUDA -DCMAKE_BUILD_TYPE=Release"
    if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
    fi
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_BALANCE_SERVE=1 -DUSE_NUMA=$USE_NUMA"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"
    if [[ -n "$CUDACXX" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=$CUDACXX"
    fi
    CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_NATIVE=ON -D_GLIBCXX_USE_CXX11_ABI=1"
    
    # 显示最终配置
    log_info "=== 最终环境配置 ==="
    log_info "CPU核心数: $cpu_cores"
    log_info "NUMA设置: USE_NUMA=$USE_NUMA"
    log_info "GPU: $gpu_name"
    if [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        log_info "CUDA架构: $compute_cap -> $CMAKE_CUDA_ARCHITECTURES"
        log_info "CUDA路径: ${CUDA_HOME:-未设置}"
        log_info "CUDA编译器: ${CUDACXX:-未设置}"
    fi
    log_info "C编译器: $CC"
    log_info "C++编译器: $CXX"
    log_info "构建类型: $CMAKE_BUILD_TYPE"
    log_info "CUDA支持: $KTRANSFORMERS_USE_CUDA"
    
    # 安装依赖
    log_info "安装Python依赖..."
    if [[ -f "requirements-local_chat.txt" ]]; then
        pip install -r requirements-local_chat.txt || log_warn "安装主要依赖失败"
    fi
    
    if [[ -f "ktransformers/server/requirements.txt" ]]; then
        log_info "安装服务器依赖..."
        pip install -r ktransformers/server/requirements.txt || log_warn "安装服务器依赖失败"
    fi
    
    # 执行安装
    log_info "执行KTransformers安装..."
    if USE_BALANCE_SERVE=1 USE_NUMA=$USE_NUMA pip install -v . --no-build-isolation; then
        log_info "KTransformers安装成功（用户环境设置方案）"
        
        # 验证安装
        if python -c "import ktransformers" 2>/dev/null; then
            log_info "KTransformers基本导入验证通过"
            cd ..
            return 0
        else
            log_warn "KTransformers导入验证失败，将尝试官方方案"
            cd ..
            return 1
        fi
    else
        log_error "KTransformers安装失败（用户环境设置方案）"
        cd ..
        return 1
    fi
}





# 安装KTransformers核心函数


# 方法1：使用用户提供的环境变量进行安装
install_kt_method1() {
    log_info "方法1：使用用户提供的环境变量进行安装..."
    
    # 输出环境变量（方法1也需要输出）
    output_env_vars "pre"
    
    # 检查kt目录是否存在
    if [[ ! -d "kt" ]]; then
        log_error "kt目录不存在，无法进入kt目录"
        log_info "生成基于系统检测的可直接复制执行的环境变量配置命令:"
        generate_env_commands
        return 1
    fi
    
    # 进入kt目录
    cd kt || {
        log_error "无法进入kt目录"
        log_info "生成基于系统检测的可直接复制执行的环境变量配置命令:"
        generate_env_commands
        return 1
    }
    
    # 清理旧环境
    log_info "清理旧环境..."
    pip uninstall ktransformers -y 2>/dev/null || log_warn "卸载旧版本失败或无旧版本"
    
    # 清理构建目录
    log_info "清理构建目录..."
    rm -rf build *.egg-info csrc/build csrc/ktransformers_ext/build
    rm -rf csrc/ktransformers_ext/cuda/build csrc/ktransformers_ext/cuda/dist
    rm -rf csrc/ktransformers_ext/cuda/*.egg-info ~/.ktransformers 2>/dev/null || true
    
    # 清理环境变量
    log_info "清理环境变量..."
    unset USE_NUMA TORCH_CUDA_ARCH_LIST CUDA_HOME CMAKE_CUDA_ARCHITECTURES
    unset KTRANSFORMERS_USE_CUDA CMAKE_BUILD_TYPE CC CXX CUDACXX
    unset CFLAGS CXXFLAGS CMAKE_ARGS USE_BALANCE_SERVE
    
    # 检测系统环境并设置编译参数
    log_info "检测系统环境并设置编译参数..."
    local env_info=$(detect_system_environment)
    IFS='|' read -r cpu_cores numa_nodes gpu_name compute_cap cuda_version gcc_version <<< "$env_info"
    
    # 设置编译并行度
    export MAKEFLAGS="-j$cpu_cores"
    export CMAKE_BUILD_PARALLEL_LEVEL=$cpu_cores
    export MAX_JOBS=$cpu_cores
    log_info "使用检测到的CPU核心数: $cpu_cores"
    
    # NUMA设置 - 优先使用用户配置
    if [[ -n "$USE_NUMA" ]]; then
        log_info "使用用户配置的NUMA设置: USE_NUMA=$USE_NUMA"
    elif [[ $numa_nodes -gt 1 ]]; then
        log_info "检测到 $numa_nodes 个NUMA节点，设置USE_NUMA=0"
        export USE_NUMA=0
    else
        export USE_NUMA=0
    fi
    
    # GPU和CUDA设置 - 优先使用用户配置
    if [[ -n "$TORCH_CUDA_ARCH_LIST" && "$TORCH_CUDA_ARCH_LIST" != "auto" ]]; then
        log_info "使用用户配置的GPU架构: $TORCH_CUDA_ARCH_LIST"
        # 清理架构列表格式
        local clean_arch_list=$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/+PTX//g' | sed 's/+RTX//g')
        export TORCH_CUDA_ARCH_LIST="$clean_arch_list"
        
        # 生成CMAKE格式（去除小数点）
        local cmake_arches=""
        for arch in $(echo "$clean_arch_list" | tr ';' ' '); do
            local cmake_arch=$(echo "$arch" | sed 's/\.//')
            if [[ -n "$cmake_arches" ]]; then
                cmake_arches="${cmake_arches};${cmake_arch}"
            else
                cmake_arches="${cmake_arch}"
            fi
        done
        export CMAKE_CUDA_ARCHITECTURES="$cmake_arches"
        log_info "设置CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
    elif [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        local arch_num=$(echo "$compute_cap" | sed 's/\.//')
        export CUDA_ARCH_LIST="$compute_cap"
        export TORCH_CUDA_ARCH_LIST="$compute_cap"
        export CMAKE_CUDA_ARCHITECTURES="$arch_num"
        log_info "检测到GPU: $gpu_name, 计算能力: $compute_cap"
        log_info "设置CUDA架构: $compute_cap -> $arch_num"
    else
        log_warn "未检测到GPU计算能力，使用CPU模式"
        export CUDA_ARCH_LIST=""
        export TORCH_CUDA_ARCH_LIST=""
        export CMAKE_CUDA_ARCHITECTURES=""
    fi
    
    # CUDA路径检测 - 优先使用用户配置
    if [[ -n "$CUDA_HOME" ]]; then
        log_info "使用用户配置的CUDA_HOME: $CUDA_HOME"
        export CUDACXX="${CUDA_HOME}/bin/nvcc"
    elif [[ "$cuda_version" != "未检测到" && -n "$cuda_version" ]]; then
        local cuda_major_minor=$(echo "$cuda_version" | cut -d'.' -f1,2)
        local cuda_paths=(
            "/usr/local/cuda-$cuda_major_minor"
            "/usr/local/cuda"
            "/opt/cuda-$cuda_major_minor"
            "/opt/cuda"
        )
        
        for cuda_path in "${cuda_paths[@]}"; do
            if [[ -d "$cuda_path" && -f "$cuda_path/bin/nvcc" ]]; then
                export CUDA_HOME="$cuda_path"
                export CUDACXX="$cuda_path/bin/nvcc"
                log_info "检测到CUDA路径: $cuda_path"
                break
            fi
        done
        
        if [[ -z "$CUDA_HOME" ]]; then
            log_warn "未找到CUDA $cuda_version 的安装路径"
        fi
    else
        log_warn "未检测到CUDA版本"
    fi
    
    # 编译器检测 - 优先使用用户配置
    if [[ -n "$CC" ]]; then
        log_info "使用用户配置的编译器: CC=$CC, CXX=$CXX"
    elif [[ "$gcc_version" != "未检测到" && -n "$gcc_version" ]]; then
        local gcc_major=$(echo "$gcc_version" | cut -d'.' -f1)
        if command -v "gcc-$gcc_major" &>/dev/null; then
            export CC="gcc-$gcc_major"
            export CXX="g++-$gcc_major"
            log_info "使用检测到的GCC版本: gcc-$gcc_major"
        else
            export CC=gcc
            export CXX=g++
            log_info "使用系统默认GCC: $gcc_version"
        fi
    else
        export CC=gcc
        export CXX=g++
        log_info "使用系统默认编译器"
    fi
    
    # 设置构建类型和优化标志 - 优先使用用户配置
    if [[ -n "$CMAKE_BUILD_TYPE" ]]; then
        log_info "使用用户配置的构建类型: CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    else
        export CMAKE_BUILD_TYPE=Release
    fi
    
    if [[ -n "$CFLAGS" ]]; then
        log_info "使用用户配置的编译标志: CFLAGS=$CFLAGS"
    else
        export CFLAGS="-O3 -march=native -D_Float128=__float128"
    fi
    
    if [[ -n "$CXXFLAGS" ]]; then
        log_info "使用用户配置的C++编译标志: CXXFLAGS=$CXXFLAGS"
    else
        export CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17 -D_Float128=__float128"
    fi
    
    # CUDA支持设置
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        export KTRANSFORMERS_USE_CUDA=ON
        log_info "启用CUDA支持"
    elif [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        export KTRANSFORMERS_USE_CUDA=ON
        log_info "启用CUDA支持"
    else
        export KTRANSFORMERS_USE_CUDA=OFF
        log_info "使用CPU模式"
    fi
    
    export KTRANSFORMERS_FORCE_BUILD=TRUE
    export USE_BALANCE_SERVE=1
    
    # 设置CMAKE_ARGS - 优先使用用户配置
    if [[ -n "$CMAKE_ARGS" ]]; then
        log_info "使用用户配置的CMAKE_ARGS: $CMAKE_ARGS"
    else
        export CMAKE_ARGS="-DKTRANSFORMERS_USE_CUDA=$KTRANSFORMERS_USE_CUDA -DCMAKE_BUILD_TYPE=Release"
        if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
        fi
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_BALANCE_SERVE=1 -DUSE_NUMA=$USE_NUMA"
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"
        if [[ -n "$CUDACXX" ]]; then
            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=$CUDACXX"
        fi
        CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_NATIVE=ON -D_GLIBCXX_USE_CXX11_ABI=1"
    fi
    
    # 显示最终配置
    log_info "=== 最终环境配置 ==="
    log_info "CPU核心数: $cpu_cores"
    log_info "NUMA设置: USE_NUMA=$USE_NUMA"
    log_info "GPU: $gpu_name"
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        log_info "CUDA架构: $TORCH_CUDA_ARCH_LIST -> $CMAKE_CUDA_ARCHITECTURES"
        log_info "CUDA路径: ${CUDA_HOME:-未设置}"
        log_info "CUDA编译器: ${CUDACXX:-未设置}"
    elif [[ -n "$compute_cap" && "$compute_cap" != "未检测到" ]]; then
        log_info "CUDA架构: $compute_cap -> $CMAKE_CUDA_ARCHITECTURES"
        log_info "CUDA路径: ${CUDA_HOME:-未设置}"
        log_info "CUDA编译器: ${CUDACXX:-未设置}"
    fi
    log_info "C编译器: $CC"
    log_info "C++编译器: $CXX"
    log_info "构建类型: $CMAKE_BUILD_TYPE"
    log_info "CUDA支持: $KTRANSFORMERS_USE_CUDA"
    
    # 安装依赖
    log_info "安装Python依赖..."
    if [[ -f "requirements-local_chat.txt" ]]; then
        pip install -r requirements-local_chat.txt || log_warn "安装主要依赖失败"
    fi
    
    if [[ -f "ktransformers/server/requirements.txt" ]]; then
        log_info "安装服务器依赖..."
        pip install -r ktransformers/server/requirements.txt || log_warn "安装服务器依赖失败"
    fi
    
    # 执行安装
    log_info "执行KTransformers安装..."
    if USE_BALANCE_SERVE=1 USE_NUMA=$USE_NUMA pip install -v . --no-build-isolation; then
        log_info "KTransformers安装成功（用户环境设置方案）"
        
        # 验证安装
        if python -c "import ktransformers" 2>/dev/null; then
            log_info "KTransformers基本导入验证通过"
            cd ..
            return 0
        else
            log_warn "KTransformers导入验证失败"
            cd ..
            return 1
        fi
    else
        log_error "KTransformers安装失败（用户环境设置方案）"
        cd ..
        return 1
    fi
}

# 方法2：使用官方默认环境变量进行安装
install_kt_method2() {
    log_info "方法2：使用官方默认环境变量进行安装..."
    
    # 输出环境变量（方法2也需要输出）
    output_env_vars "pre"
    
    # 进入kt目录
    cd kt || {
        log_error "无法进入kt目录"
        return 1
    }
    
    # 清理旧环境
    log_info "清理旧环境..."
    pip uninstall ktransformers -y 2>/dev/null || log_warn "卸载旧版本失败或无旧版本"
    
    # 清理构建目录
    log_info "清理构建目录..."
    rm -rf build *.egg-info csrc/build csrc/ktransformers_ext/build
    rm -rf csrc/ktransformers_ext/cuda/build csrc/ktransformers_ext/cuda/dist
    rm -rf csrc/ktransformers_ext/cuda/*.egg-info ~/.ktransformers 2>/dev/null || true
    
    # 清理环境变量
    log_info "清理环境变量..."
    unset USE_NUMA TORCH_CUDA_ARCH_LIST CUDA_HOME CMAKE_CUDA_ARCHITECTURES
    unset KTRANSFORMERS_USE_CUDA CMAKE_BUILD_TYPE CC CXX CUDACXX
    unset CFLAGS CXXFLAGS CMAKE_ARGS USE_BALANCE_SERVE
    
    # 设置官方推荐的环境变量 - 优先使用用户配置
    log_info "设置官方推荐的环境变量..."
    
    # CUDA路径设置 - 优先使用用户配置
    if [[ -n "$CUDA_HOME" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        export CUDA_PATH="$CUDA_HOME"
        log_info "使用用户配置的CUDA路径: $CUDA_HOME"
    else
        # 必需的CUDA路径设置
        if [ -d "/usr/local/cuda/bin" ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            log_info "添加CUDA到PATH: /usr/local/cuda/bin"
        fi
        
        if [ -d "/usr/local/cuda/lib64" ]; then
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
            log_info "添加CUDA到LD_LIBRARY_PATH: /usr/local/cuda/lib64"
        fi
        
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_PATH="/usr/local/cuda"
            log_info "设置CUDA_PATH: /usr/local/cuda"
        fi
    fi
    
    # 设置编译选项 - 优先使用用户配置
    if [[ -n "$KTRANSFORMERS_FORCE_BUILD" ]]; then
        log_info "使用用户配置的KTRANSFORMERS_FORCE_BUILD: $KTRANSFORMERS_FORCE_BUILD"
    else
        export KTRANSFORMERS_FORCE_BUILD=1
    fi
    
    if [[ -n "$USE_NUMA" ]]; then
        log_info "使用用户配置的NUMA设置: USE_NUMA=$USE_NUMA"
    else
        export USE_NUMA=0
    fi
    
    if [[ -n "$USE_BALANCE_SERVE" ]]; then
        log_info "使用用户配置的USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    else
        export USE_BALANCE_SERVE=1
    fi
    
    if [[ -n "$CMAKE_BUILD_TYPE" ]]; then
        log_info "使用用户配置的构建类型: CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    else
        export CMAKE_BUILD_TYPE=Release
    fi
    
    # 设置CMAKE_ARGS - 优先使用用户配置
    if [[ -n "$CMAKE_ARGS" ]]; then
        log_info "使用用户配置的CMAKE_ARGS: $CMAKE_ARGS"
    else
        export CMAKE_ARGS="-DKTRANSFORMERS_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DUSE_BALANCE_SERVE=1 -DUSE_NUMA=0 -DLLAMA_NATIVE=ON"
    fi
    
    # 安装依赖
    log_info "安装Python依赖..."
    if [[ -f "requirements-local_chat.txt" ]]; then
        pip install -r requirements-local_chat.txt || log_warn "安装主要依赖失败"
    fi
    
    if [[ -f "ktransformers/server/requirements.txt" ]]; then
        log_info "安装服务器依赖..."
        pip install -r ktransformers/server/requirements.txt || log_warn "安装服务器依赖失败"
    fi
    
    # 执行安装
    log_info "执行KTransformers安装..."
    if USE_BALANCE_SERVE=$USE_BALANCE_SERVE USE_NUMA=$USE_NUMA pip install -v . --no-build-isolation; then
        log_info "KTransformers安装成功（官方环境设置方案）"
        
        # 验证安装
        if python -c "import ktransformers" 2>/dev/null; then
            log_info "KTransformers基本导入验证通过"
            cd ..
            return 0
        else
            log_warn "KTransformers导入验证失败"
            cd ..
            return 1
        fi
    else
        log_error "KTransformers安装失败（官方环境设置方案）"
        cd ..
        return 1
    fi
}

###############################
# 主函数部分
###############################

# 最终检查函数 - 只在成功安装后执行
check_final_verification() {
    log_info "开始最终安装验证..."
    
    # 检查是否成功安装了KTransformers
    if ! check_and_manage_kt "check"; then
        log_warn "KTransformers未安装，跳过最终检查"
        return 0
    fi
    
    # 验证功能是否正常
    if check_and_manage_kt "verify"; then
        log_info "最终验证通过：KTransformers安装正常且功能可用"
        
        # 显示安装摘要
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}KTransformers 安装成功!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "安装目录: ${YELLOW}$KT_ROOT${NC}"
        echo -e "Conda环境: ${YELLOW}$CONDA_ENV${NC}"
        echo -e "Python版本: ${YELLOW}$PYTHON_VERSION${NC}"
        echo -e "PyTorch版本: ${YELLOW}$PYTORCH_VERSION${NC}"
        echo -e "CUDA版本: ${YELLOW}$CUDA_VERSION${NC}"
        echo -e "${GREEN}========================================${NC}"
        
        return 0
    else
        log_error "最终验证失败：KTransformers功能异常"
        return 1
    fi
}

# 安装KTransformers主函数（简化版本）
install_kt() {
    show_progress "安装KTransformers"
    
    log_info "开始安装KTransformers..."
    
    # 确保在conda环境中
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_error "未检测到激活的conda环境，无法安装KTransformers"
        return 1
    fi
    
    # 询问KT目录，默认使用脚本同级目录
    ask_kt_install_directory
    export KT_ROOT
    log_info "KTransformers将安装到: $KT_ROOT"
    
    # 1. 检查是否已安装kt
    if check_and_manage_kt "check"; then
        log_info "检测到KTransformers已安装，验证功能..."
        
        # 验证功能是否正常
        local kt_functional=false
        if check_and_manage_kt "verify"; then
            log_info "KTransformers安装正常且可用"
            kt_functional=true
        else
            log_warn "KTransformers安装存在问题"
        fi
        
        # 无论功能是否正常，都询问用户是否卸载重新安装
        if ask_user_reinstall; then
            log_info "用户选择卸载并重新安装"
            check_and_manage_kt "uninstall"
        else
            log_info "用户选择保持当前安装"
            return 0
        fi
    else
        log_info "未检测到KTransformers，将进行安装"
    fi
    
    # 2. 若未安装，优先采用方案1进行安装
    if [[ ! -d "$KT_ROOT" ]]; then
        log_error "未检测到KTransformers项目目录: $KT_ROOT"
        
        # 安装文件缺失，直接输出可直接复制执行的环境变量配置命令
        log_info "生成基于系统检测的可直接复制执行的环境变量配置命令:"
        generate_env_commands
        return 1
    fi
    
    # 优先尝试方法1，在函数内部完成环境设置和安装流程
    log_info "优先采用方案1进行安装..."
    if install_kt_method1; then
        log_info "KTransformers安装成功(使用用户环境变量)"
        # 3. 最终检查阶段（仅在安装成功时执行）
        check_final_verification
        return 0
    fi
    
    log_warn "方案1安装失败，执行回退操作..."
    
    # 安装失败则执行回退操作，调用install_kt_method2函数改用官方环境变量设置进行安装
    log_info "回退到方案2: 使用官方默认环境变量进行安装..."
    if install_kt_method2; then
        log_info "KTransformers安装成功(使用官方环境变量)"
        # 最终检查阶段（仅在安装成功时执行）
        check_final_verification
        return 0
    fi
    
    # 若再次失败，输出可直接复制执行的环境变量配置命令后终止安装流程
    log_error "所有安装方案均失败，请手动设置以下环境变量后重试:"
    generate_env_commands
    return 1
}







# 脚本结束