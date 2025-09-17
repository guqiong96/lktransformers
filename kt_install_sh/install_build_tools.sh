#!/bin/bash

# KTransformers 编译工具安装模块
# 包含编译工具和优化库安装

# 引入conda环境管理模块以使用configure_pip_mirror函数
source "${MODULES_DIR:-$(dirname "$0")}/conda_env_manager.sh"

# 测试PPA源连通性
test_ppa_connectivity() {
    local ppa_url="$1"
    local timeout=10
    
    log_message "INFO" "测试PPA源连通性: $ppa_url"
    
    # 使用curl测试连通性，设置超时时间
    if curl -s --connect-timeout $timeout --max-time $timeout "$ppa_url" >/dev/null 2>&1; then
        log_message "INFO" "PPA源连通性测试成功: $ppa_url"
        return 0
    else
        log_message "WARN" "PPA源连通性测试失败: $ppa_url"
        return 1
    fi
}

# 设置软件仓库（支持国内镜像源）
setup_repositories() {
    log_message "INFO" "设置软件仓库..."
    
    # 检查是否使用国内镜像源
    local use_china_mirror=false
    if [[ "${USE_APT_MIRROR:-n}" == "y" ]]; then
        use_china_mirror=true
        log_message "INFO" "检测到用户选择使用国内镜像源，将优先使用国内PPA镜像"
    fi
    
    # 尝试添加GCC PPA源
    log_message "INFO" "尝试添加GCC PPA源..."
    local ppa_added=false
    
    if [[ "$use_china_mirror" == true ]]; then
        # 国内用户优先使用中科大PPA反向代理
        local china_ppa_url="https://launchpad.proxy.ustclug.org/ubuntu-toolchain-r/test/ubuntu"
        local china_ppa_url_backup="http://launchpad.proxy.ustclug.org/ubuntu-toolchain-r/test/ubuntu"
        
        # 清理所有可能存在的ubuntu-toolchain-r相关源配置，避免冲突
        log_message "INFO" "清理可能存在的ubuntu-toolchain-r相关源配置"
        sudo rm -f /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list
        sudo rm -f /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-*.list
        # 清理可能通过add-apt-repository添加的源（不同Ubuntu版本可能有不同的文件名）
        sudo rm -f /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-jammy.list
        sudo rm -f /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-focal.list
        sudo rm -f /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-bionic.list
        # 清理可能的其他格式
        sudo find /etc/apt/sources.list.d/ -name "*ubuntu-toolchain-r*" -type f -delete 2>/dev/null || true
        
        # 测试中科大HTTPS镜像连通性
        if test_ppa_connectivity "$china_ppa_url"; then
            log_message "INFO" "使用中科大PPA镜像源(HTTPS): $china_ppa_url"
            # 手动添加PPA源到sources.list.d
            local ubuntu_version=$(lsb_release -cs 2>/dev/null || echo "jammy")
            echo "deb $china_ppa_url $ubuntu_version main" | sudo tee /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list > /dev/null
            
            # 添加GPG密钥
            if sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F 2>/dev/null; then
                log_message "INFO" "成功添加GCC PPA GPG密钥"
                ppa_added=true
                log_message "INFO" "已成功配置中科大PPA镜像源(HTTPS)，跳过官方源"
                # 显示实际配置的源
                log_message "INFO" "当前PPA源配置: $(cat /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list 2>/dev/null || echo '配置文件不存在')"
            else
                log_message "WARN" "添加GCC PPA GPG密钥失败，尝试备用方法"
                # 尝试备用密钥服务器
                if sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F 2>/dev/null; then
                    log_message "INFO" "成功添加GCC PPA GPG密钥(备用服务器)"
                    ppa_added=true
                    log_message "INFO" "已成功配置中科大PPA镜像源(HTTPS)，跳过官方源"
                    # 显示实际配置的源
                    log_message "INFO" "当前PPA源配置: $(cat /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list 2>/dev/null || echo '配置文件不存在')"
                fi
            fi
        elif test_ppa_connectivity "$china_ppa_url_backup"; then
            log_message "INFO" "使用中科大PPA镜像源(HTTP): $china_ppa_url_backup"
            # 手动添加PPA源到sources.list.d
            local ubuntu_version=$(lsb_release -cs 2>/dev/null || echo "jammy")
            echo "deb $china_ppa_url_backup $ubuntu_version main" | sudo tee /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list > /dev/null
            
            # 添加GPG密钥
            if sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F 2>/dev/null; then
                log_message "INFO" "成功添加GCC PPA GPG密钥"
                ppa_added=true
                log_message "INFO" "已成功配置中科大PPA镜像源(HTTP)，跳过官方源"
                # 显示实际配置的源
                log_message "INFO" "当前PPA源配置: $(cat /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list 2>/dev/null || echo '配置文件不存在')"
            else
                log_message "WARN" "添加GCC PPA GPG密钥失败，尝试备用方法"
                # 尝试备用密钥服务器
                if sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F 2>/dev/null; then
                    log_message "INFO" "成功添加GCC PPA GPG密钥(备用服务器)"
                    ppa_added=true
                    log_message "INFO" "已成功配置中科大PPA镜像源(HTTP)，跳过官方源"
                    # 显示实际配置的源
                    log_message "INFO" "当前PPA源配置: $(cat /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list 2>/dev/null || echo '配置文件不存在')"
                fi
            fi
        else
            log_message "WARN" "中科大PPA镜像源连通性测试失败，将尝试官方源"
        fi
    fi
    
    # 如果国内镜像失败或未启用，尝试官方PPA源
    if [[ "$ppa_added" == false ]]; then
        log_message "INFO" "尝试使用官方GCC PPA源..."
        
        # 清理可能存在的中科大源配置文件，避免冲突
        log_message "INFO" "清理可能存在的PPA源配置文件"
        sudo find /etc/apt/sources.list.d/ -name "*ubuntu-toolchain-r*" -type f -delete 2>/dev/null || true
        
        # 使用add-apt-repository添加官方PPA源
        log_message "INFO" "添加官方GCC PPA源..."
        if sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test 2>/dev/null; then
            log_message "INFO" "成功添加官方GCC PPA源"
            ppa_added=true
            # 显示实际配置的源
            log_message "INFO" "当前使用官方PPA源"
        else
            log_message "WARN" "使用add-apt-repository添加官方PPA源失败"
            
            # 尝试手动添加官方源
            local official_ppa_url="https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu"
            if test_ppa_connectivity "$official_ppa_url"; then
                log_message "INFO" "尝试手动添加官方PPA源"
                local ubuntu_version=$(lsb_release -cs 2>/dev/null || echo "jammy")
                echo "deb $official_ppa_url $ubuntu_version main" | sudo tee /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list > /dev/null
                
                # 添加GPG密钥
                if sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F 2>/dev/null; then
                    log_message "INFO" "成功添加官方GCC PPA GPG密钥"
                    ppa_added=true
                    # 显示实际配置的源
                    log_message "INFO" "当前PPA源配置: $(cat /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list 2>/dev/null || echo '配置文件不存在')"
                else
                    log_message "WARN" "添加官方GCC PPA GPG密钥失败"
                fi
            else
                log_message "WARN" "官方PPA源连通性测试失败"
            fi
        fi
    fi
    
    # 如果所有PPA源都失败，记录警告但继续安装
    if [[ "$ppa_added" == false ]]; then
        log_message "WARN" "所有GCC PPA源都无法访问，将使用系统默认GCC版本"
        log_message "WARN" "这可能会影响某些需要较新GCC版本的编译任务"
    fi
    
    # 更新软件包列表
    log_message "INFO" "更新软件包列表..."
    if ! sudo apt-get update; then
        log_message "WARN" "更新软件包列表失败，但将继续安装"
    fi
    
    log_message "INFO" "软件仓库设置完成"
    return 0
}

# 安装系统依赖
install_system_dependencies() {
    log_message "INFO" "安装系统级依赖..."
    
    # 分批安装系统依赖
    
    # 1. 基础编译工具（不包含cmake，单独处理）
    log_message "INFO" "安装基础编译工具..."
    local base_tools=("build-essential" "git" "pkg-config" "patchelf" "ninja-build" "gcc-13" "g++-13")
    if ! sudo apt-get install -y ${base_tools[*]}; then
        log_message "WARN" "部分基础编译工具安装失败，但将继续安装"
    fi
    
    # 单独安装CMake，使用多种方法
    install_cmake
    
    # 2. 开发库 - 压缩相关
    log_message "INFO" "安装压缩相关开发库..."
    local compression_libs=("zlib1g-dev" "libbz2-dev" "liblzma-dev" "xz-utils")
    if ! sudo apt-get install -y ${compression_libs[*]}; then
        log_message "WARN" "部分压缩相关开发库安装失败，但将继续安装"
    fi
    
    # 3. 开发库 - 系统相关
    log_message "INFO" "安装系统相关开发库..."
    local system_libs=("libreadline-dev" "libsqlite3-dev" "libncurses5-dev" "libncursesw5-dev" "tk-dev" "libffi-dev")
    if ! sudo apt-get install -y ${system_libs[*]}; then
        log_message "WARN" "部分系统相关开发库安装失败，但将继续安装"
    fi
    
    # 4. 开发库 - 网络和性能相关
    log_message "INFO" "安装网络和性能相关开发库..."
    local network_perf_libs=("libssl-dev" "libcurl4-openssl-dev" "libtbb-dev" "libfmt-dev" "libgflags-dev")
    if ! sudo apt-get install -y ${network_perf_libs[*]}; then
        log_message "WARN" "部分网络和性能相关开发库安装失败，但将继续安装"
    fi
    
    # 5. NUMA相关库 - 无论是否启用NUMA都安装
    log_message "INFO" "安装NUMA相关库..."
    local numa_libs=("libnuma-dev" "numactl")
    if ! sudo apt-get install -y ${numa_libs[*]}; then
        log_message "WARN" "NUMA相关库安装失败，但将继续安装"
    fi
    
    # 6. 安装TBB库 - 解决CMake找不到TBB的问题
    log_message "INFO" "安装TBB库..."
    local tbb_libs=("libtbb-dev" "libtbbmalloc2")
    if ! sudo apt-get install -y ${tbb_libs[*]}; then
        log_message "WARN" "TBB库安装失败，尝试安装替代版本"
        
        # 尝试安装其他可能的TBB包名
        if ! sudo apt-get install -y libtbb-dev; then
            log_message "WARN" "无法安装libtbb-dev"
        fi
    fi
    
    log_message "INFO" "系统依赖安装完成"
}

# 安装CMake的函数
install_cmake() {
    log_message "INFO" "安装CMake..."
    
    # 方法1: 尝试使用snap安装最新版本的CMake
    if command -v snap >/dev/null 2>&1; then
        log_message "INFO" "尝试使用snap安装CMake..."
        if sudo snap install cmake --classic; then
            log_message "INFO" "CMake通过snap安装成功"
            return 0
        else
            log_message "WARN" "snap安装CMake失败，尝试其他方法"
        fi
    else
        log_message "INFO" "snap不可用，跳过snap安装方法"
    fi
    
    # 方法2: 使用apt安装系统默认版本
    log_message "INFO" "尝试使用apt安装CMake..."
    if sudo apt-get install -y cmake; then
        log_message "INFO" "CMake通过apt安装成功"
        return 0
    else
        log_message "WARN" "apt安装CMake失败"
    fi
    
    # 方法3: 如果都失败了，记录警告但不阻止安装
    log_message "WARN" "所有CMake安装方法都失败了，但将继续安装过程"
    log_message "WARN" "您可能需要手动安装CMake"
    return 0
}

# 安装Conda环境包
install_conda_packages() {
    log_message "INFO" "安装Conda环境包..."
    
    # C++库相关Conda包
    log_message "INFO" "安装C++库相关Conda包..."
    local cpp_conda_packages=("libstdcxx-ng")
    echo -e "${GREEN}安装C++库相关Conda包: ${cpp_conda_packages[*]}${NC}"
    if ! conda install -y -c conda-forge "${cpp_conda_packages[@]}"; then
        log_message "ERROR" "安装C++库相关Conda包失败，可能影响编译"
        return 1
    fi
    
    log_message "INFO" "Conda环境包安装完成"
    return 0
}

# 安装Pip包
install_pip_packages() {
    log_message "INFO" "安装Pip包..."
    
    # 按照KTransformers官方文档顺序安装核心依赖
    log_message "INFO" "安装KTransformers核心依赖包..."
    local core_pip_packages=("packaging" "ninja" "cpufeature" "numpy")
    echo -e "${GREEN}安装KTransformers核心依赖包: ${core_pip_packages[*]}${NC}"
    if ! pip install -i https://mirrors.aliyun.com/pypi/simple/ "${core_pip_packages[@]}"; then
        log_message "WARN" "部分核心依赖包安装失败，尝试逐个安装"
        for pkg in "${core_pip_packages[@]}"; do
            log_message "INFO" "使用pip安装: $pkg"
            pip install -i https://mirrors.aliyun.com/pypi/simple/ "$pkg" || log_message "WARN" "无法安装: $pkg"
        done
    fi
    
    # 编译相关Pip包（去除与核心依赖重复的包）
    log_message "INFO" "安装编译相关Pip包..."
    local build_pip_packages=("cython" "pybind11" "cmake" "setuptools" "wheel" "build")
    echo -e "${GREEN}安装编译相关Pip包: ${build_pip_packages[*]}${NC}"
    if ! pip install -i https://mirrors.aliyun.com/pypi/simple/ "${build_pip_packages[@]}"; then
        log_message "WARN" "部分编译相关Pip包安装失败，尝试逐个安装"
        for pkg in "${build_pip_packages[@]}"; do
            log_message "INFO" "使用pip安装: $pkg"
            pip install -i https://mirrors.aliyun.com/pypi/simple/ "$pkg" || log_message "WARN" "无法安装: $pkg"
        done
    fi
    
    # 测试相关Pip包
    log_message "INFO" "安装测试相关Pip包..."
    local test_pip_packages=("pytest" "pytest-benchmark")
    echo -e "${GREEN}安装测试相关Pip包: ${test_pip_packages[*]}${NC}"
    if ! pip install -i https://mirrors.aliyun.com/pypi/simple/ "${test_pip_packages[@]}"; then
        log_message "WARN" "部分测试相关Pip包安装失败，尝试逐个安装"
        for pkg in "${test_pip_packages[@]}"; do
            log_message "INFO" "使用pip安装: $pkg"
            pip install -i https://mirrors.aliyun.com/pypi/simple/ "$pkg" || log_message "WARN" "无法安装: $pkg"
        done
    fi
    
    # 代码质量相关Pip包
    log_message "INFO" "安装代码质量相关Pip包..."
    local quality_pip_packages=("black" "isort" "mypy" "ruff" "cmake_format")
    echo -e "${GREEN}安装代码质量相关Pip包: ${quality_pip_packages[*]}${NC}"
    if ! pip install -i https://mirrors.aliyun.com/pypi/simple/ "${quality_pip_packages[@]}"; then
        log_message "WARN" "部分代码质量相关Pip包安装失败，尝试逐个安装"
        for pkg in "${quality_pip_packages[@]}"; do
            log_message "INFO" "使用pip安装: $pkg"
            pip install -i https://mirrors.aliyun.com/pypi/simple/ "$pkg" || log_message "WARN" "无法安装: $pkg"
        done
    fi
    
    # 其他科学计算相关Pip包（去除numpy重复）
    log_message "INFO" "安装其他科学计算相关Pip包..."
    local scientific_pip_packages=("openai")
    echo -e "${GREEN}安装其他科学计算相关Pip包: ${scientific_pip_packages[*]}${NC}"
    if ! pip install -i https://mirrors.aliyun.com/pypi/simple/ "${scientific_pip_packages[@]}"; then
        log_message "WARN" "部分科学计算相关Pip包安装失败，尝试逐个安装"
        for pkg in "${scientific_pip_packages[@]}"; do
            log_message "INFO" "使用pip安装: $pkg"
            pip install -i https://mirrors.aliyun.com/pypi/simple/ "$pkg" || log_message "WARN" "无法安装: $pkg"
        done
    fi
    
    log_message "INFO" "Pip包安装完成"
    return 0
}

# 安装libaio1依赖（单独处理，因为在不同的Ubuntu版本中可能会有不同的包名）
install_libaio() {
    log_message "INFO" "开始安装libaio依赖..."
    
    # 检查是否已经安装
    if dpkg -l | grep -q "libaio1\|libaio1t64"; then
        log_message "INFO" "libaio已安装，跳过安装"
        return 0
    fi
    
    # 安装libaio-dev函数
    install_libaio_dev() {
        if sudo apt-get install -y libaio-dev; then
            log_message "INFO" "成功安装libaio-dev"
            return 0
        else
            log_message "WARN" "安装libaio-dev失败，但将继续安装"
            return 1
        fi
    }
    
    # 先尝试使用apt直接安装
    log_message "INFO" "尝试通过apt安装libaio..."
    
    # 尝试安装libaio1
    if sudo apt-get install -y libaio1; then
        log_message "INFO" "成功安装libaio1"
        install_libaio_dev
        return 0
    # 在Ubuntu 24.04上可能需要安装libaio1t64
    elif sudo apt-get install -y libaio1t64; then
        log_message "INFO" "成功安装libaio1t64"
        install_libaio_dev
        return 0
    fi
    
    # 如果apt安装失败，尝试通过deb包安装
    log_message "INFO" "通过apt安装失败，尝试通过deb包安装libaio1..."
    
    # 创建临时目录
    local temp_dir=$(mktemp -d)
    cd "$temp_dir" || {
        log_error "无法创建临时目录"
        return 1
    }
    
    # 获取Ubuntu版本
    local ubuntu_version=""
    if command -v lsb_release &> /dev/null; then
        ubuntu_version=$(lsb_release -rs)
    fi
    log_message "INFO" "检测到Ubuntu版本: $ubuntu_version"
    
    # 为不同的Ubuntu版本选择不同的deb包
    local deb_urls=()
    
    if [[ "$ubuntu_version" == "24.04" ]]; then
        # Ubuntu 24.04 Noble
        deb_urls+=("http://archive.ubuntu.com/ubuntu/pool/main/liba/libaio/libaio1t64_0.3.113-6build1_amd64.deb")
    elif [[ "$ubuntu_version" == "22.04" ]]; then
        # Ubuntu 22.04 Jammy
        deb_urls+=("http://archive.ubuntu.com/ubuntu/pool/main/liba/libaio/libaio1_0.3.112-13build1_amd64.deb")
    fi
    
    # 添加通用版本作为备选
    deb_urls+=("http://archive.ubuntu.com/ubuntu/pool/main/liba/libaio/libaio1_0.3.112-5_amd64.deb")
    
    # 尝试下载并安装deb包
    for deb_url in "${deb_urls[@]}"; do
        log_message "INFO" "尝试下载并安装: $deb_url"
        if wget "$deb_url" -O libaio.deb && sudo dpkg -i libaio.deb; then
            log_message "INFO" "成功通过deb包安装libaio1"
            install_libaio_dev
            cd - > /dev/null || true
            rm -rf "$temp_dir"
            return 0
        fi
    done
    
    # 清理并返回错误
    cd - > /dev/null || true
    rm -rf "$temp_dir"
    log_message "ERROR" "无法安装libaio1依赖，请手动安装"
    return 1
}

# 设置编译器版本（仅设置编译器版本，环境变量由setup_env_vars.sh处理）
setup_compiler_versions() {
    log_message "INFO" "设置编译器版本..."
    
    # 设置GCC-13为默认版本
    log_message "INFO" "设置GCC-13为默认版本..."
    if ! sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100; then
        log_message "WARN" "设置gcc-13优先级失败"
    fi
    
    if ! sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100; then
        log_message "WARN" "设置g++-13优先级失败"
    fi
    
    # 自动选择gcc-13和g++-13作为默认版本
    if ! sudo update-alternatives --set gcc /usr/bin/gcc-13; then
        log_message "WARN" "设置gcc-13为默认版本失败"
    fi
    
    if ! sudo update-alternatives --set g++ /usr/bin/g++-13; then
        log_message "WARN" "设置g++-13为默认版本失败"
    fi
    
    # 验证GCC版本
    log_message "INFO" "验证GCC版本..."
    local gcc_version=$(gcc --version | head -n 1)
    local gpp_version=$(g++ --version | head -n 1)
    log_message "INFO" "当前GCC版本: $gcc_version"
    log_message "INFO" "当前G++版本: $gpp_version"
    
    log_message "INFO" "编译器版本设置完成"
    return 0
}

# 安装编译工具和conda环境包
install_build_tools() {
    show_progress "安装编译工具和开发环境"
    
    log_message "INFO" "开始安装编译工具和开发环境..."
    
    # 确保在conda环境中
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_message "ERROR" "未检测到激活的conda环境，请先激活环境"
        return 1
    fi
    
    # 设置软件仓库和密钥
    setup_repositories
    
    # 安装系统级编译工具
    install_system_dependencies
    
    # 安装libaio依赖
    install_libaio
    
    # 安装Conda环境包
    install_conda_packages
    
    # 配置pip镜像源
configure_pip_mirror || {
    log_message "WARN" "配置pip镜像源失败，但将继续安装"
}

# 安装pip包
install_pip_packages

    # 设置编译器版本（环境变量由setup_env_vars.sh处理）
    setup_compiler_versions
    
    log_message "SUCCESS" "编译工具和开发环境安装完成"
    return 0
}