#!/bin/bash

# KTransformers 系统检查模块
# 包含环境检查和网络检查功能

# 检查必要依赖的函数
check_dependencies() {
    local missing_deps=()
    local required_commands=("curl" "git")
    
    log_info "检查必要依赖..."
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        else
            log_debug "找到命令: $cmd"
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "缺少必要依赖: ${missing_deps[*]}"
        log_info "请先安装缺少的依赖，然后重新运行脚本"
        return 1
    fi
    
    log_info "所有必要依赖检查通过"
    return 0
}

# 验证网络连接的函数
check_network_connectivity() {
    # 脚本中使用的主要域名
    local main_domains=(
        "https://github.com"           # GitHub主域名
        "https://pypi.org"             # PyPI官方源
        "https://mirrors.tuna.tsinghua.edu.cn"  # 清华镜像
        "https://repo.anaconda.com"    # Conda官方源
        "https://mirrors.bfsu.edu.cn"  # 北外镜像
    )
    
    # 如果启用了Git代理，添加代理地址到检测列表
    if [[ "${USE_GIT_HOSTS}" == "y" ]]; then
        # 使用hosts配置进行连接测试
        main_domains+=("https://github.com")  # 测试GitHub主域名
        log_message "INFO" "已启用GitHub hosts配置，将测试GitHub连接"
    fi
    
    local fallback_urls=("https://www.baidu.com" "https://www.google.com")
    local connected=false
    local accessible_domains=()
    local failed_domains=()
    
    log_info "检查网络连接状态..."
    
    # 检查主要域名连接性
    log_info "检查脚本依赖的主要域名:"
    for url in "${main_domains[@]}"; do
        local domain=$(echo "$url" | sed 's|https\?://||' | cut -d'/' -f1)
        if curl -s --connect-timeout 5 --max-time 15 "$url" > /dev/null 2>&1; then
            log_info "  ✓ $domain - 连接正常"
            accessible_domains+=("$domain")
            connected=true
        else
            log_warn "  ✗ $domain - 连接失败"
            failed_domains+=("$domain")
        fi
    done
    
    # 如果主要域名都无法连接，检查基础网络连接
    if [[ "$connected" == "false" ]]; then
        log_warn "主要域名均无法访问，检查基础网络连接..."
        for url in "${fallback_urls[@]}"; do
            local domain=$(echo "$url" | sed 's|https\?://||' | cut -d'/' -f1)
            if curl -s --connect-timeout 5 --max-time 15 "$url" > /dev/null 2>&1; then
                log_info "  ✓ $domain - 连接正常"
                connected=true
                break
            else
                log_warn "  ✗ $domain - 连接失败"
            fi
        done
    fi
    
    # 输出检查结果摘要
    if [[ "$connected" == "true" ]]; then
        log_info "网络连接检查完成 - 基础网络正常"
        if [[ ${#accessible_domains[@]} -gt 0 ]]; then
            log_info "可访问的域名: ${accessible_domains[*]}"
        fi
        if [[ ${#failed_domains[@]} -gt 0 ]]; then
            log_warn "无法访问的域名: ${failed_domains[*]}"
            log_warn "这可能影响某些功能，建议检查网络配置或使用镜像源"
        fi
        return 0
    else
        log_error "网络连接检查失败 - 无法访问任何测试域名"
        log_error "请检查以下项目:"
        log_error "  1. 网络连接是否正常"
        log_error "  2. 防火墙或代理设置"
        log_error "  3. DNS解析是否正常"
        log_error "  4. 是否需要配置网络代理"
        return 1
    fi
}

# 检测CPU架构优化支持
detect_cpu_architecture() {
    log_info "检测CPU架构优化支持..."
    
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
    local supports_sse4=false
    
    # 检测AMX指令集（Intel Advanced Matrix Extensions）
    # AMX包含多个子指令集：amx_tile, amx_int8, amx_bf16
    if echo "$cpu_flags" | grep -qE "\bamx_tile\b|\bamx_int8\b|\bamx_bf16\b|\bamx\b"; then
        supports_amx=true
        log_info "检测到AMX指令集支持"
        # 详细检测AMX子集
        local amx_features=""
        if echo "$cpu_flags" | grep -q "amx_tile"; then
            amx_features="${amx_features}TILE "
        fi
        if echo "$cpu_flags" | grep -q "amx_int8"; then
            amx_features="${amx_features}INT8 "
        fi
        if echo "$cpu_flags" | grep -q "amx_bf16"; then
            amx_features="${amx_features}BF16 "
        fi
        if [[ -n "$amx_features" ]]; then
            log_info "  AMX子集: ${amx_features}"
        fi
    fi
    
    # 检测AVX512指令集族（更精确的匹配）
    if echo "$cpu_flags" | grep -qE "\bavx512[a-z_]*\b"; then
        supports_avx512=true
        log_info "检测到AVX512指令集支持"
        # 详细检测AVX512子集
        local avx512_features=""
        if echo "$cpu_flags" | grep -q "avx512f"; then
            avx512_features="${avx512_features}F "
        fi
        if echo "$cpu_flags" | grep -q "avx512dq"; then
            avx512_features="${avx512_features}DQ "
        fi
        if echo "$cpu_flags" | grep -q "avx512cd"; then
            avx512_features="${avx512_features}CD "
        fi
        if echo "$cpu_flags" | grep -q "avx512bw"; then
            avx512_features="${avx512_features}BW "
        fi
        if echo "$cpu_flags" | grep -q "avx512vl"; then
            avx512_features="${avx512_features}VL "
        fi
        if echo "$cpu_flags" | grep -q "avx512_vnni"; then
            avx512_features="${avx512_features}VNNI "
        fi
        if [[ -n "$avx512_features" ]]; then
            log_info "  AVX512子集: ${avx512_features}"
        fi
    fi
    
    # 检测AVX2指令集（精确匹配）
    if echo "$cpu_flags" | grep -q "\bavx2\b"; then
        supports_avx2=true
        log_info "检测到AVX2指令集支持"
    fi
    
    # 检测AVX指令集
    if echo "$cpu_flags" | grep -q "\bavx\b"; then
        supports_avx=true
        log_info "检测到AVX指令集支持"
    fi
    
    # 检测SSE4指令集
    if echo "$cpu_flags" | grep -qE "\bsse4_[12]\b"; then
        supports_sse4=true
        log_info "检测到SSE4指令集支持"
    fi
    
    # 自动设置CPU架构优化（按性能从高到低排序）
    if [[ "$supports_amx" == "true" ]]; then
        export CPU_ARCH_OPT="amx"
        export ENABLE_AMX="1"
        export CPU_INSTRUCTION_SET="AMX"
        log_info "自动启用AMX优化 (CPU_ARCH_OPT=amx, ENABLE_AMX=1)"
    elif [[ "$supports_avx512" == "true" ]]; then
        export CPU_ARCH_OPT="avx512"
        export ENABLE_AMX="0"
        export CPU_INSTRUCTION_SET="AVX512"
        log_info "自动启用AVX512优化 (CPU_ARCH_OPT=avx512)"
    elif [[ "$supports_avx2" == "true" ]]; then
        export CPU_ARCH_OPT="avx2"
        export ENABLE_AMX="0"
        export CPU_INSTRUCTION_SET="AVX2"
        log_info "自动启用AVX2优化 (CPU_ARCH_OPT=avx2)"
    elif [[ "$supports_avx" == "true" ]]; then
        export CPU_ARCH_OPT="avx"
        export ENABLE_AMX="0"
        export CPU_INSTRUCTION_SET="AVX"
        log_info "自动启用AVX优化 (CPU_ARCH_OPT=avx)"
    elif [[ "$supports_sse4" == "true" ]]; then
        export CPU_ARCH_OPT="sse4"
        export ENABLE_AMX="0"
        export CPU_INSTRUCTION_SET="SSE4"
        log_info "自动启用SSE4优化 (CPU_ARCH_OPT=sse4)"
    else
        export CPU_ARCH_OPT="auto"
        export ENABLE_AMX="0"
        export CPU_INSTRUCTION_SET="AUTO"
        log_info "使用自动检测优化 (CPU_ARCH_OPT=auto)"
    fi
    
    # 导出检测结果供其他脚本使用
    export CPU_SUPPORTS_AMX="$supports_amx"
    export CPU_SUPPORTS_AVX512="$supports_avx512"
    export CPU_SUPPORTS_AVX2="$supports_avx2"
    export CPU_SUPPORTS_AVX="$supports_avx"
    export CPU_SUPPORTS_SSE4="$supports_sse4"
    
    # 输出检测摘要
    log_info "CPU指令集检测摘要: AMX=$supports_amx, AVX512=$supports_avx512, AVX2=$supports_avx2, AVX=$supports_avx, SSE4=$supports_sse4"
    
    # 设置编译优化级别
    export COMPILER_OPTIMIZATION_LEVEL="-O3 -march=native"
    log_info "设置编译优化级别: $COMPILER_OPTIMIZATION_LEVEL"
    
    # 设置构建并行度
    local cpu_cores=$(nproc)
    export BUILD_PARALLEL_JOBS="$cpu_cores"
    log_info "设置构建并行度: $BUILD_PARALLEL_JOBS"
    
    return 0
}

# 检查NUMA拓扑结构
check_numa_topology() {
    log_info "检查NUMA拓扑结构..."
    
    # 检查是否安装了numactl
    if ! command -v numactl &> /dev/null; then
        log_warn "未安装numactl，无法获取详细NUMA信息"
        log_info "建议安装numactl: apt install -y numactl"
        
        # 尝试通过lscpu获取基本NUMA信息
        if command -v lscpu &> /dev/null; then
            local numa_nodes=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
            if [[ -n "$numa_nodes" && "$numa_nodes" -gt 1 ]]; then
                log_info "检测到 $numa_nodes 个NUMA节点"
                return $numa_nodes
            else
                log_info "未检测到多NUMA节点系统或无法确定NUMA节点数量"
                return 1
            fi
        else
            log_warn "无法检测NUMA拓扑，假设为单NUMA节点"
            return 1
        fi
    fi
    
    # 使用numactl获取详细NUMA信息
    log_info "NUMA硬件信息:"
    local numa_info=$(numactl --hardware)
    echo "$numa_info" | while IFS= read -r line; do
        log_info "  $line"
    done
    
    # 获取NUMA节点数量
    local numa_nodes=$(echo "$numa_info" | grep "available:" | awk '{print $2}')
    if [[ -n "$numa_nodes" && "$numa_nodes" -gt 1 ]]; then
        log_info "检测到 $numa_nodes 个NUMA节点"
        
        # 自动设置USE_NUMA环境变量
        if [[ "$USE_NUMA" != "1" ]]; then
            log_info "自动启用NUMA优化 (USE_NUMA=1)"
            export USE_NUMA=1
        fi
        
        # 提供NUMA绑定建议
        log_info "NUMA绑定建议:"
        log_info "  1. 在运行模型时，使用numactl绑定到特定NUMA节点可提升性能"
        log_info "  2. 示例命令: numactl --cpunodebind=0 --membind=0 python -m ktransformers.local_chat ..."
        log_info "  3. 对于多NUMA节点系统，确保USE_NUMA=1已设置"
        log_info "  4. 在多NUMA系统上，建议将模型的不同部分分配到不同NUMA节点"
        
        return $numa_nodes
    else
        log_info "系统为单NUMA节点或无法确定NUMA节点数量"
        return 1
    fi
}

# 主函数 - 系统环境检查
check_system() {
    show_progress "系统环境检查"
    
    # 检查依赖
    check_dependencies || {
        log_error "基础依赖检查失败"
        return 1
    }
    
    # 检查网络连接
    check_network_connectivity || {
        log_warn "网络连接检查失败，但将继续安装"
    }
    
    # 检查是否为Ubuntu系统
    local is_ubuntu=false
    local ubuntu_version=""
    
    # 首先尝试通过/etc/os-release检测
    if [[ -f /etc/os-release ]]; then
        if grep -qi "ubuntu" /etc/os-release; then
            is_ubuntu=true
            ubuntu_version=$(grep VERSION_ID /etc/os-release 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "unknown")
            log_info "通过/etc/os-release检测到Ubuntu系统版本: $ubuntu_version"
        fi
    fi
    
    # 如果/etc/os-release检测失败，尝试使用lsb_release
    if [[ "$is_ubuntu" == "false" ]] && command -v lsb_release &> /dev/null; then
        if lsb_release -d | grep -qi "ubuntu"; then
            is_ubuntu=true
            ubuntu_version=$(lsb_release -rs)
            log_info "通过lsb_release检测到Ubuntu系统版本: $ubuntu_version"
        fi
    fi
    
    # 根据检测结果输出信息
    if [[ "$is_ubuntu" == "false" ]]; then
        log_warn "未检测到Ubuntu系统，某些功能可能不可用"
        log_info "当前系统信息:"
        if [[ -f /etc/os-release ]]; then
            local os_name=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "Unknown")
            log_info "  操作系统: $os_name"
        fi
    else
        # 检查Ubuntu版本是否符合要求
        if command -v bc &> /dev/null && [[ "$ubuntu_version" != "unknown" ]]; then
            if (( $(echo "$ubuntu_version < 20.04" | bc -l) )); then
                log_warn "推荐使用Ubuntu 20.04或更高版本，当前版本: $ubuntu_version"
            else
                log_info "Ubuntu版本检查通过: $ubuntu_version"
            fi
        else
            log_info "无法进行版本比较检查（缺少bc命令或版本信息不完整）"
        fi
    fi
    
    # 检查CPU
    log_info "检查CPU配置..."
    local cpu_info=$(lscpu)
    local cpu_cores=$(echo "$cpu_info" | grep "^CPU(s):" | awk '{print $2}')
    local cpu_model=$(echo "$cpu_info" | grep "Model name:" | sed 's/Model name: *//')
    
    log_info "CPU: $cpu_model (${cpu_cores}核)"
    
    # 检测CPU架构优化支持
    detect_cpu_architecture
    
    # 检查NUMA拓扑
    local numa_nodes=1
    check_numa_topology
    numa_nodes=$?
    
    # 检查内存
    log_info "检查内存..."
    local mem_info=$(free -h)
    local total_mem=$(echo "$mem_info" | grep "^Mem:" | awk '{print $2}')
    local avail_mem=$(echo "$mem_info" | grep "^Mem:" | awk '{print $7}')
    local total_mem_kb=$(free | grep "^Mem:" | awk '{print $2}')
    
    log_info "总内存: $total_mem, 可用内存: $avail_mem"
    
    # 检查内存是否足够
    if (( total_mem_kb < 8000000 )); then  # 大约8GB
        log_warn "系统内存不足，推荐至少8GB内存"
    fi
    
    # 检查磁盘空间
    log_info "检查磁盘空间..."
    local root_free=$(df -h / | tail -n 1 | awk '{print $4}')
    local home_free=$(df -h $HOME | tail -n 1 | awk '{print $4}')
    local root_free_kb=$(df / | tail -n 1 | awk '{print $4}')
    local home_free_kb=$(df $HOME | tail -n 1 | awk '{print $4}')
    
    log_info "/ 分区可用空间: $root_free"
    log_info "$HOME 目录可用空间: $home_free"
    
    # 检查磁盘空间是否足够
    if (( root_free_kb < 10000000 )); then  # 大约10GB
        log_warn "根分区空间不足，推荐至少10GB"
    fi
    
    if (( home_free_kb < 100000000 )); then  # 大约100GB
        log_warn "$HOME 目录空间不足，推荐至少100GB用于模型和数据"
    fi
    
    # 检查防火墙
    log_info "检查防火墙状态..."
    if command -v ufw &> /dev/null; then
        local ufw_status=""
        # 以root权限直接执行
        ufw_status=$(ufw status 2>/dev/null | grep "Status:" | awk '{print $2}')
        
        if [[ "$ufw_status" == "active" ]]; then
            log_warn "UFW防火墙处于启用状态，可能需要开放端口"
        elif [[ "$ufw_status" == "inactive" ]]; then
            log_info "UFW防火墙未启用"
        elif [[ -n "$ufw_status" ]]; then
            log_info "UFW防火墙状态: $ufw_status"
        fi
    else
        log_info "系统未安装UFW防火墙"
    fi
    
    log_info "系统环境检查完成"
    return 0
}