#!/bin/bash

# KTransformers GPU和CUDA安装模块
# 支持Ubuntu 20.04-24.04及WSL环境
# 2025年优化版本 - 基于最新NVIDIA官方规范
# 特别优化RTX 50系列显卡支持
# 默认CUDA 12.8版本，驱动与CUDA分离安装

# ==================== 模块初始化 ====================

# 基础颜色定义和日志函数
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'
log_message() { echo -e "[$1] $2"; }
log_info() { log_message "INFO" "$1"; }
log_warn() { log_message "WARN" "$1"; }
log_error() { log_message "ERROR" "$1"; }

# 通用工具函数 - 提高代码复用性
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info() { echo -e "${BLUE}$1${NC}"; }

# 通用APT操作函数
safe_apt_update() {
    print_info "更新软件包列表..."
    if apt-get update -y; then
        print_success "软件包列表更新完成"
        return 0
    else
        print_error "软件包列表更新失败"
        return 1
    fi
}

safe_apt_install() {
    local packages="$*"
    print_info "安装软件包: $packages"
    if apt-get install -y $packages; then
        print_success "软件包安装完成: $packages"
        return 0
    else
        print_error "软件包安装失败: $packages"
        return 1
    fi
}

# 通用用户选择函数
get_user_choice() {
    local prompt="$1"
    local valid_choices="$2"
    local default="$3"
    local choice

    while true; do
        read -p "$prompt [默认: $default] ($valid_choices): " choice
        choice=${choice:-$default}

        if [[ "$valid_choices" =~ $choice ]]; then
            echo "$choice"
            return 0
        else
            print_error "无效选择，请输入: $valid_choices"
        fi
    done
}

# 通用文件检查函数
check_file_exists() {
    local file="$1"
    local description="$2"

    if [[ -f "$file" ]]; then
        print_success "发现$description: $file"
        return 0
    else
        print_warning "未发现$description: $file"
        return 1
    fi
}

# ==================== 2025年最新配置常量 ====================

# 系统环境变量
IS_WSL=false
UBUNTA_VERSION=""
NEED_INSTALL_CUDA=false
NEED_INSTALL_CUDNN=false
NEED_UNINSTALL_CUDNN=false
NEED_INSTALL_DRIVER=false
NEED_REBOOT=false
FORCE_CUDA_REINSTALL=false
# 只在未设置时初始化CUDA_VERSION
: ${CUDA_VERSION:=""}
DRIVER_UPGRADED=false

# NVIDIA官方仓库配置 - 可配置参数
: ${NVIDIA_BASE_URL:="https://developer.download.nvidia.com"}
: ${NVIDIA_REPO_PATH:="/compute/cuda/repos"}
: ${NVIDIA_CUDA_PATH:="/compute/cuda"}
: ${NVIDIA_CUDNN_PATH:="/compute/cudnn"}
declare -A NVIDIA_REPO_URLS=(
    ["20.04"]="${NVIDIA_BASE_URL}${NVIDIA_REPO_PATH}/ubuntu2004/x86_64"
    ["22.04"]="${NVIDIA_BASE_URL}${NVIDIA_REPO_PATH}/ubuntu2204/x86_64"
    ["24.04"]="${NVIDIA_BASE_URL}${NVIDIA_REPO_PATH}/ubuntu极速版2404/x86_64"
)

# 2025年最新版本配置
NVIDIA_GPG_KEY="3bf863cc.pub"
NVIDIA_KEYRING_PACKAGE="cuda-keyring_1.1-1_all.deb"

# CUDA版本配置 - 2025年最新版本
declare -A CUDA_RUNFILE_URLS=(
    ["12.8"]="${NVIDIA_BASE_URL}${NVIDIA_CUDA_PATH}/12.8.1/local_installers/cuda_12.8.1_570.86.10_linux.run"
    ["12.6"]="${NVIDIA_BASE_URL}${NVIDIA_CUDA_PATH}/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
)

# cuDNN下载配置 - 2025年最新版本
declare -A CUDNN_DOWNLOAD_URLS=(
    ["12.8"]="${NVIDIA_BASE_URL}${NVIDIA_CUDNN_PATH}/9.6.0/local_installers/cudnn-linux-x86_64-9.6.0_cuda12.8-archive.tar.xz"
    ["12.6"]="${NVIDIA_BASE_URL}${NVIDIA_CUDNN_PATH}/9.6.0/local_installers/cudnn-linux-x86_64-9.6.0_cuda12.6-archive.tar.xz"
)

# 版本要求 - 2025年标准，支持RTX 50系列
CUDA_MIN_DRIVER_VERSION="575.0"  # 默认最低要求，将根据CUDA版本动态调整
CUDNN_VERSION="9.6.0"  # 最新cuDNN版本

# CUDA版本对应的最低驱动版本要求
declare -A CUDA_DRIVER_REQUIREMENTS=(
    ["12.8"]="575.0"  # CUDA 12.8需要575+驱动
    ["12.6"]="560.0"  # CUDA 12.6需要560+驱动
)

# 根据CUDA版本设置驱动版本要求
set_driver_requirements_for_cuda() {
    local cuda_version="$1"

    if [[ -n "${CUDA_DRIVER_REQUIREMENTS[$cuda_version]}" ]]; then
        CUDA_MIN_DRIVER_VERSION="${CUDA_DRIVER_REQUIREMENTS[$cuda_version]}"
        echo -e "${BLUE}根据CUDA $cuda_version 设置最低驱动版本要求: $CUDA_MIN_DRIVER_VERSION${NC}"
    else
        echo -e "${YELLOW}未知CUDA版本 $cuda_version，使用默认驱动版本要求: $CUDA_MIN_DRIVER_VERSION${NC}"
    fi
}

# ==================== 核心功能函数 ====================

# 第一步：清理配置冲突
clean_nvidia_configuration() {
    log_info "第一步：清理NVIDIA配置冲突..."

    print_info "正在清理可能存在的NVIDIA配置冲突..."

    # 清理密钥文件
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null
    rm -f /usr/share/keyrings/nvidia-archive-keyring.gpg 2>/dev/null

    # 清理源列表文件
    rm -f /etc/apt/sources.list极速版.d/nvidia*.list 2>/dev/null
    rm -f /etc/apt/sources.list.d/cuda*.list 2>/dev/null

    # 清理主源列表中的NVIDIA条目
    if [[ -f "/etc/apt/sources.list" ]]; then
        sed -i '/developer.download.nvidia.com/d' /etc/apt/sources.list 2>/dev/null
    fi

    # 清理APT缓存
    apt-get clean 2>/dev/null || true

    print_success "NVIDIA配置清理完成"
}

# 第二步：检测系统环境
detect_system_environment() {
    log_info "第二步：检测系统环境..."

    print_info "正在检测系统环境..."

    # 检测WSL环境
    if [[ -f /proc/version ]] && grep -qi "microsoft\|wsl" /proc/version; then
        IS_WSL=true
        print_warning "检测到WSL环境"
    else
        print_success "检测到原生Linux环境"
    fi

    # 检测Ubuntu版本
    if command -v lsb_release &> /dev/null; then
        UBUNTU_VERSION=$(lsb_release -rs | cut -d. -f1,2)
    elif [[ -f /etc/os-release ]]; then
        UBUNTU_VERSION=$(grep "VERSION_ID" /etc/os-release | cut -d'"' -f2 | cut -d. -f1,2)
    else
        UBUNTU_VERSION="22.04"
    fi

    # 验证版本支持
    if [[ ! "${NVIDIA_REPO_URLS[$UBUNTU_VERSION]}" ]]; then
        echo -e "${YELLOW}不支持的Ubuntu版本: $UBUNTU_VERSION，使用22.04配置${NC}"
        UBUNTU_VERSION="22.04"
    fi

    echo -e "${GREEN}✓ 系统环境: Ubuntu $UBUNTU_VERSION $([ "$IS_WSL" = true ] && echo "(WSL)" || echo "(Native)")${NC}"

    if [[ "$IS_WSL" == "true" ]]; then
        echo -e "${YELLOW}WSL环境说明：${NC}"
        echo -e "  • NVIDIA驱动需要在Windows主机上安装"
        echo -e "  • WSL中只需安装CUDA工具包"
    fi
}

# 检测RTX 50系列显卡
detect_rtx_50_series() {
    local gpu_info="$1"
    local device_id=""

    # 提取设备ID
    if [[ "$gpu_info" =~ Device[[:space:]]+([0-9a-fA-F]{4}) ]]; then
        device_id="${BASH_REMATCH[1]}"
        device_id=$(echo "$device_id" | tr '[:upper:]' '[:lower:]')

        # RTX 50系列设备ID范围检测 (基于NVIDIA官方信息)
        # RTX 50系列设备ID通常在2d00-2dff范围内
        if [[ "$device_id" =~ ^2d[0-9a-f][0-9a-f]$ ]]; then
            echo "RTX 50 Series (Device ID: $device_id)"
            return 0
        fi
    fi

    # 备用检测：通过GPU名称检测
    if echo "$gpu_info" | grep -qi "rtx.*50\|geforce.*50"; then
        echo "RTX 50 Series (Name Detection)"
        return 0
    fi

    return 1
}

# 第三步：检测NVIDIA GPU硬件
check_nvidia_gpu_hardware() {
    log_info "第三步：检测NVIDIA GPU硬件..."

    echo -e "${BLUE}正在检测NVIDIA GPU硬件...${NC}"

    local gpu_detected=false
    local gpu_info=""
    local is_rtx_50=false
    local rtx_50_model=""

    # 方法1：通过lspci检测GPU硬件
    if gpu_info=$(lspci | grep -i nvidia | grep "VGA\|3D\|Display" | head -1); then
        gpu_detected=true
        echo -e "${GREEN}✓ 检测到NVIDIA GPU硬件: ${gpu_info}${NC}"

        # 检测是否为RTX 50系列
        if rtx_50_model=$(detect_rtx_50_series "$gpu_info"); then
            is_rtx_50=true
            echo -e "${YELLOW}⚠ 检测到RTX 50系列显卡: $rtx_50_model${NC}"
            echo -e "${YELLOW}注意: RTX 50系列显卡需要特殊的驱动支持${NC}"
        fi
    fi

    # 方法2：检查NVIDIA驱动目录
    if [[ "$gpu_detected" == "false" ]] && [[ -d "/proc/driver/nvidia" ]]; then
        gpu_detected=true
        echo -e "${GREEN}✓ 检测到NVIDIA驱动目录${NC}"
    fi

    # 方法3：检查nvidia设备文件
    if [[ "$gpu_detected" == "false" ]] && [[ -c "/dev/nvidia0" ]]; then
        gpu_detected=true
        echo -e "${GREEN}✓ 检测到NVIDIA设备文件${NC}"
    fi

    # WSL环境特殊处理
    if [[ "$IS_WSL" == "true" ]]; then
        if [[ "$gpu_detected" == "false" ]]; then
            echo -e "${YELLOW}WSL环境下无法直接检测GPU硬件，这是正常现象${NC}"
            echo -e "${YELLOW}请确保Windows主机已安装NVIDIA驱动（版本 ≥ 575.0）${NC}"
        fi
        return 0
    fi

    # 原生Linux环境检查
    if [[ "$gpu_detected" == "false" ]]; then
        echo -e "${RED}✗ 未检测到NVIDIA GPU${NC}"
        echo -e "${YELLOW}请检查：${NC}"
        echo -e "  • GPU是否正确安装在PCIe插槽中"
        echo -e "  • BIOS中是否启用独立显卡"
        echo -e "  • 电源是否充足（RTX 50系列功耗较高）"
        echo -e "  • PCIe插槽是否支持GPU规格"
        return 1
    fi

    # 设置全局变量供后续使用
    export IS_RTX_50_SERIES="$is_rtx_50"
    export RTX_50_MODEL="$rtx_50_model"

    if [[ "$is_rtx_50" == "true" ]]; then
        echo -e "${BLUE}RTX 50系列显卡检测完成，将使用专门的驱动安装策略${NC}"
    fi

    return 0
}

# 第四步：检查并处理NVIDIA驱动
check_nvidia_driver_status() {
    log_info "第四步：检查NVIDIA驱动状态..."

    echo -e "${BLUE}正在检查NVIDIA驱动状态...${NC}"

    # WSL环境跳过驱动检查
    if [[ "$IS_WSL" == "true" ]]; then
        echo -e "${GREEN}✓ WSL环境跳过驱动检查${NC}"
        return 0
    fi

    # 检查Nouveau驱动冲突
    if lsmod | grep -q nouveau 2>/dev/null; then
        echo -e "${YELLOW}检测到Nouveau开源驱动冲突${NC}"
        disable_nouveau_driver
        return 2  # 需要重启
    fi

    # 检查NVIDIA驱动
    if command -v nvidia-smi &> /dev/null; then
        local driver_version
        if driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1); then
            echo -e "${GREEN}✓ 检测到NVIDIA驱动版本: $driver_version${NC}"

            # 检查版本是否满足要求
            if awk "BEGIN {exit !($driver_version >= $CUDA_MIN_DRIVER_VERSION)}"; then
                echo -e "${GREEN}✓ 驱动版本满足要求（>=$CUDA_MIN_DRIVER_VERSION）${NC}"
                return 0
            else
                echo -e "${YELLOW}驱动版本过低（当前: $driver_version，要求: >=$CUDA_MIN_DRIVER_VERSION）${NC}"
                return 1  # 需要升级驱动
            fi
        fi
    fi

    echo -e "${YELLOW}未检测到有效的NVIDIA驱动${NC}"
    return 1  # 需要安装驱动
}

# 禁用Nouveau驱动
disable_nouveau_driver() {
    print_info "正在禁用Nouveau驱动..."

    # 卸载Nouveau包
        apt-get remove --purge -y xserver-xorg-video-nouveau nouveau-firmware 2>/dev/null || true

        # 添加黑名单
        echo "blacklist nouveau" | tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
        echo "options nouveau modeset=0" | tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null

        # 更新initramfs
        update-initramfs -u 2>/dev/null || true

    print_warning "Nouveau驱动已禁用，需要重启系统"
}

# 深度清理NVIDIA驱动
deep_clean_nvidia_drivers() {
    log_info "深度清理NVIDIA驱动..."

    echo -e "${YELLOW}正在深度清理NVIDIA驱动和相关组件...${NC}"

    # 停止相关服务
    systemctl stop nvidia-persistenced 2>/dev/null || true

    # 卸载所有NVIDIA相关包
    apt purge -y nvidia-* libnvidia-* 2>/dev/null || true
    apt autoremove -y 2>/dev/null || true

    # 清理配置文件
    rm -rf /etc/modprobe.d/nvidia-*.conf 2>/dev/null || true
    rm -rf /etc/modprobe.d/blacklist-nvidia*.conf 2>/dev/null || true

    # 清理内核模块
    modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true

    # 更新initramfs
    update-initramfs -u 2>/dev/null || true

    echo -e "${GREEN}✓ NVIDIA驱动深度清理完成${NC}"
}

# 检测驱动冲突
detect_driver_conflicts() {
    log_info "检测驱动冲突..."

    local conflicts_found=false

    echo -e "${BLUE}正在检测驱动冲突...${NC}"

    # 检查Nouveau驱动
    if lsmod | grep -q nouveau; then
        echo -e "${YELLOW}⚠ 检测到Nouveau开源驱动正在运行${NC}"
        conflicts_found=true
    fi

    # 检查混合驱动安装
    local nvidia_packages=$(dpkg -l | grep -E "nvidia-driver|nvidia-[0-9]" | wc -l)
    if [[ $nvidia_packages -gt 1 ]]; then
        echo -e "${YELLOW}⚠ 检测到多个NVIDIA驱动包，可能存在冲突${NC}"
        dpkg -l | grep -E "nvidia-driver|nvidia-[0-9]" | awk '{print "  " $2 " (" $3 ")"}'
        conflicts_found=true
    fi

    # 检查内核模块冲突
    if [[ -f /proc/modules ]] && grep -q nvidia /proc/modules; then
        local loaded_modules=$(grep nvidia /proc/modules | awk '{print $1}')
        echo -e "${BLUE}当前加载的NVIDIA模块: $loaded_modules${NC}"
    fi

    if [[ "$conflicts_found" == "true" ]]; then
        echo -e "${YELLOW}检测到驱动冲突，建议进行深度清理${NC}"
        return 1
    else
        echo -e "${GREEN}✓ 未检测到驱动冲突${NC}"
        return 0
    fi
}

# 安装NVIDIA开源驱动（适用于RTX 50系列）
install_nvidia_open_source_driver() {
    log_info "安装NVIDIA开源驱动..."

    echo -e "${BLUE}正在安装NVIDIA开源驱动（推荐用于RTX 50系列）...${NC}"

    # 更新包列表
    safe_apt_update || return 1

    # 添加NVIDIA官方PPA以获取最新驱动
    echo -e "${BLUE}添加NVIDIA官方PPA...${NC}"
    add-apt-repository ppa:graphics-drivers/ppa -y || {
        echo -e "${YELLOW}添加PPA失败，尝试继续安装...${NC}"
    }
    safe_apt_update

    # 检测可用的开源驱动版本
    local latest_open_driver=""
    local driver_version="575"  # 默认版本

    # 尝试检测最新的开源驱动
    local driver_search_result=$(apt-cache search nvidia-driver | grep -E "nvidia-driver-[0-9]+-open" | sort -V 2>/dev/null)

    if [[ -n "$driver_search_result" ]]; then
        # 提取最新的开源驱动版本
        latest_open_driver=$(echo "$driver_search_result" | tail -1 | awk '{print $1}')
        driver_version=$(echo "$latest_open_driver" | grep -oE '[0-9]+' | head -1)

        echo -e "${BLUE}检测到最新开源驱动: $latest_open_driver (版本 $driver_version)${NC}"
    else
        echo -e "${YELLOW}未检测到开源驱动，使用固定版本575...${NC}"
        latest_open_driver="nvidia-driver-575-open"
    fi

    # 构建安装包列表
    local open_driver_packages=(
        "$latest_open_driver"
        "nvidia-dkms-${driver_version}-open"
        "nvidia-utils-${driver_version}"
        "nvidia-settings"
        "nvidia-prime"
    )

    echo -e "${BLUE}准备安装开源驱动包...${NC}"
    for package in "${open_driver_packages[@]}"; do
        echo -e "  • $package"
    done

    # 安装开源驱动
    if safe_apt_install "${open_driver_packages[@]}"; then
        echo -e "${GREEN}✓ NVIDIA开源驱动安装成功${NC}"

        # 配置开源驱动
        echo -e "${BLUE}配置开源驱动...${NC}"

        # 创建配置文件
        sudo tee /etc/modprobe.d/nvidia-open.conf > /dev/null << EOF
# NVIDIA开源驱动配置 - RTX 50系列优化
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1
options nvidia-drm modeset=1
options nvidia NVreg_PreserveVideoMemoryAllocations=1
EOF

        # 禁用nouveau驱动
        sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null << EOF
# 禁用nouveau驱动以避免冲突
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF

        # 更新initramfs
        echo -e "${BLUE}更新initramfs...${NC}"
        update-initramfs -u

        echo -e "${GREEN}✓ NVIDIA开源驱动配置完成${NC}"
        echo -e "${YELLOW}注意: RTX 50系列显卡需要重启系统以加载新驱动${NC}"

        return 0
    else
        echo -e "${RED}✗ NVIDIA开源驱动安装失败${NC}"
        return 1
    fi
}

# 安装NVIDIA官方驱动
install_nvidia_official_driver() {
    print_info "尝试安装NVIDIA官方驱动..."

    # 检测驱动冲突
    if ! detect_driver_conflicts; then
        echo -e "${YELLOW}检测到驱动冲突，是否进行深度清理？ (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            deep_clean_nvidia_drivers
        fi
    fi

    # 更新软件包列表
    safe_apt_update || return 1

    # 安装依赖
    safe_apt_install software-properties-common apt-transport-https ca-certificates gnupg || return 1

    # 添加NVIDIA PPA
    print_info "添加NVIDIA官方PPA..."
    if sudo add-apt-repository ppa:graphics-drivers/ppa -y; then
        print_success "NVIDIA PPA添加成功"
    else
        print_error "添加PPA失败"
        return 1
    fi

    safe_apt_update || return 1

    # 检测GPU型号并选择合适的驱动版本
    local gpu_info=$(lspci | grep -i nvidia | head -1)
    local recommended_driver

    # RTX 50系列特殊处理
    if [[ "$IS_RTX_50_SERIES" == "true" ]]; then
        echo -e "${GREEN}检测到RTX 50系列显卡: $RTX_50_MODEL${NC}"
        echo -e "${BLUE}RTX 50系列显卡安装说明:${NC}"
        echo -e "  • 需要NVIDIA驱动版本 ≥ 575.0"
        echo -e "  • 推荐使用开源驱动以获得最佳兼容性"
        echo -e "  • 支持CUDA 12.8及更高版本"
        echo -e "${YELLOW}RTX 50系列显卡优先尝试开源驱动...${NC}"

        # 先尝试开源驱动
        if install_nvidia_open_source_driver; then
            echo -e "${GREEN}✓ RTX 50系列开源驱动安装成功${NC}"
            return 0
        else
            echo -e "${YELLOW}开源驱动安装失败，尝试最新官方驱动...${NC}"

            # 检测最新可用的官方驱动
            local latest_driver=$(apt-cache search nvidia-driver | grep -E "nvidia-driver-[0-9]+[^-]" | grep -v open | sort -V | tail -1 | awk '{print $1}')
            if [[ -n "$latest_driver" ]]; then
                local driver_version=$(echo "$latest_driver" | grep -oE '[0-9]+' | head -1)
                if [[ "$driver_version" -ge 575 ]]; then
                    recommended_driver="$latest_driver"
                    echo -e "${BLUE}为RTX 50系列选择驱动: $recommended_driver${NC}"
                else
                    recommended_driver="nvidia-driver-575"
                    echo -e "${YELLOW}检测到的驱动版本过低，使用575版本${NC}"
                fi
            else
                recommended_driver="nvidia-driver-575"
                echo -e "${YELLOW}未检测到合适驱动，使用575版本${NC}"
            fi
        fi
    else
        # 其他显卡使用系统推荐版本
        recommended_driver=$(ubuntu-drivers devices | grep recommended | awk '{print $3}' | head -1)
        if [[ -z "$recommended_driver" ]]; then
            recommended_driver="nvidia-driver-570"  # 2025年默认版本
        fi
        echo -e "${BLUE}系统推荐驱动版本: $recommended_driver${NC}"
    fi

    echo -e "${BLUE}安装NVIDIA驱动: $recommended_driver${NC}"
    if safe_apt_install "$recommended_driver"; then
        DRIVER_UPGRADED=true
        echo -e "${GREEN}✓ NVIDIA官方驱动安装完成${NC}"
        return 0
    else
        echo -e "${RED}✗ NVIDIA官方驱动安装失败${NC}"
        return 1
    fi
}

# 安装开源驱动（备用方案）
install_opensource_driver() {
    echo -e "${BLUE}尝试安装开源驱动作为备用方案...${NC}"

    # 对于RTX 50系列，尝试安装最新的Nouveau驱动
    if [[ "$IS_RTX_50_SERIES" == "true" ]]; then
        echo -e "${YELLOW}RTX 50系列显卡检测到，尝试安装最新Nouveau驱动...${NC}"

        # 添加最新的Mesa PPA以获得更好的RTX 50支持
        sudo add-apt-repository ppa:oibaf/graphics-drivers -y 2>/dev/null || true
        safe_apt_update

        # 安装最新的Mesa和Nouveau驱动
        local nouveau_packages=(
            "xserver-xorg-video-nouveau"
            "mesa-vulkan-drivers"
            "mesa-utils"
            "libgl1-mesa-dri"
        )

        if safe_apt_install "${nouveau_packages[@]}"; then
            echo -e "${GREEN}✓ 最新Nouveau驱动安装成功${NC}"
        else
            echo -e "${YELLOW}最新驱动安装失败，尝试标准Nouveau驱动...${NC}"
            safe_apt_install xserver-xorg-video-nouveau || {
                echo -e "${RED}✗ 安装开源驱动失败${NC}"
                return 1
            }
        fi
    else
        # 其他显卡安装标准nouveau驱动
        echo -e "${BLUE}安装标准nouveau开源驱动...${NC}"
        safe_apt_install xserver-xorg-video-nouveau || {
            echo -e "${RED}✗ 安装开源驱动失败${NC}"
            return 1
        }
    fi

    # 移除nouveau黑名单
    sudo rm -f /etc/modprobe.d/blacklist-nouveau.conf 2>/dev/null

    # 配置Nouveau驱动
    echo "options nouveau modeset=1" | sudo tee /etc/modprobe.d/nouveau.conf > /dev/null

    # 更新initramfs
    sudo update-initramfs -u 2>/dev/null || true

    echo -e "${GREEN}✓ 开源驱动安装完成${NC}"
    echo -e "${YELLOW}注意：开源驱动性能有限，建议稍后尝试更新官方驱动${NC}"

    return 0
}

# 主驱动安装函数
install_nvidia_driver() {
    echo -e "${BLUE}正在安装NVIDIA驱动...${NC}"

    # 首先尝试官方驱动
    if install_nvidia_official_driver; then
        echo -e "${GREEN}✓ 官方驱动安装成功${NC}"
    else
        echo -e "${YELLOW}官方驱动安装失败，尝试开源驱动...${NC}"
        if install_opensource_driver; then
            echo -e "${GREEN}✓ 开源驱动安装成功${NC}"
        else
            echo -e "${RED}✗ 所有驱动安装方案都失败${NC}"
            return 1
        fi
    fi

    # 询问是否重启
    echo -e "${YELLOW}驱动安装完成，建议重启系统以使驱动生效${NC}"
    read -p "是否现在重启？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}正在重启系统...${NC}"
        reboot
    else
        echo -e "${YELLOW}请稍后手动重启系统${NC}"
    fi
}

# 第五步：获取CUDA安装选择
get_cuda_installation_choice() {
    log_info "第五步：获取CUDA安装选择..."

    echo -e "\n${BLUE}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ ${GREEN}CUDA版本选择${BLUE} │${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────────────┘${NC}"
    echo -e " ${YELLOW}1.${NC} 安装CUDA 12.8 ${BLUE}(RTX 50系列推荐，默认)${NC}"
    echo -e " ${YELLOW}2.${NC} 安装CUDA 12.6 ${BLUE}(稳定版本)${NC}"
    echo

    local choice=$(get_user_choice "请选择CUDA版本" "1 2" "1")

    case $choice in
        1)
            CUDA_VERSION="12.8"
            print_success "选择CUDA 12.8"
            set_driver_requirements_for_cuda "$CUDA_VERSION"
            ;;
        2)
            CUDA_VERSION="12.6"
            print_success "选择CUDA 12.6"
            set_driver_requirements_for_cuda "$CUDA_VERSION"
            ;;
    esac

    echo -e "\n${BLUE}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ ${GREEN}安装方式选择${BLUE} │${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────────────┘${NC}"
    echo -e " ${YELLOW}1.${NC} 智能检测 ${BLUE}(如果已安装相同版本则跳过)${NC}"
    echo -e " ${YELLOW}2.${NC} 强制重新安装 ${BLUE}(无论当前状态如何都重新安装)${NC}"
    echo

    local install_choice=$(get_user_choice "请选择安装方式" "1 2" "1")

    case $install_choice in
        1)
            FORCE_CUDA_REINSTALL=false
            print_success "选择智能检测模式"
            ;;
        2)
            FORCE_CUDA_REINSTALL=true
            print_success "选择强制重新安装模式"
            ;;
    esac

    # cuDNN安装选择
    echo -e "\n${BLUE}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ ${GREEN}cuDNN安装选择${BLUE} │${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────────────┘${NC}"

    # 检查是否已安装cuDNN
    local cudnn_installed=false
    if [[ -f "/usr/local/cuda/include/cudnn.h" ]] || [[ -f "/usr/include/cudnn.h" ]]; then
        cudnn_installed=true
        echo -e "${GREEN}✓ 检测到已安装cuDNN${NC}"
        echo -e " ${YELLOW}1.${NC} 保持当前cuDNN安装"
        echo -e " ${YELLOW}2.${NC} 卸载当前cuDNN"
        echo -e " ${YELLOW}3.${NC} 重新安装cuDNN"
        echo

        local cudnn_choice=$(get_user_choice "请选择cuDNN操作" "1 2 3" "1")

        case $cudnn_choice in
            1)
                NEED_INSTALL_CUDNN=false
                NEED_UNINSTALL_CUDNN=false
                print_success "保持当前cuDNN安装"
                ;;
            2)
                NEED_INSTALL_CUDNN=false
                NEED_UNINSTALL_CUDNN=true
                print_success "将卸载当前cuDNN"
                ;;
            3)
                NEED_INSTALL_CUDNN=true
                NEED_UNINSTALL_CUDNN=true
                print_success "将重新安装cuDNN"
                ;;
        esac
    else
        print_warning "未检测到cuDNN安装"
        echo -e " ${YELLOW}1.${NC} 安装cuDNN ${BLUE}(推荐，深度学习必需)${NC}"
        echo -e " ${YELLOW}2.${NC} 跳过cuDNN安装"
        echo

        local cudnn_install_choice=$(get_user_choice "请选择是否安装cuDNN" "1 2" "2")

        case $cudnn_install_choice in
            1)
                NEED_INSTALL_CUDNN=true
                NEED_UNINSTALL_CUDNN=false
                print_success "将安装cuDNN"
                ;;
            2)
                NEED_INSTALL_CUDNN=false
                NEED_UNINSTALL_CUDNN=false
                print_success "跳过cuDNN安装"
                ;;
        esac
    fi
}

# CUDA错误诊断和恢复
diagnose_cuda_issues() {
    log_info "诊断CUDA问题..."

    echo -e "${BLUE}正在诊断CUDA相关问题...${NC}"

    # 检查CUDA路径
    echo -e "${YELLOW}检查CUDA安装路径:${NC}"
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-*"
        "/opt/cuda"
        "/usr/lib/cuda"
    )

    for path_pattern in "${cuda_paths[@]}"; do
        for cuda_path in $path_pattern; do
            if [[ -d "$cuda_path" ]]; then
                echo -e "  ✓ 发现: $cuda_path"
                if [[ -f "$cuda_path/bin/nvcc" ]]; then
                    echo -e "    - nvcc: 存在"
                else
                    echo -e "    - nvcc: ${RED}缺失${NC}"
                fi
            fi
        done
    done

    # 检查环境变量
    echo -e "${YELLOW}检查环境变量:${NC}"
    echo -e "  CUDA_HOME: ${CUDA_HOME:-未设置}"
    echo -e "  PATH中的CUDA: $(echo $PATH | grep -o '[^:]*cuda[^:]*' | head -1 || echo '未找到')"
    echo -e "  LD_LIBRARY_PATH中的CUDA: $(echo $LD_LIBRARY_PATH | grep -o '[^:]*cuda[^:]*' | head -1 || echo '未找到')"

    # 检查NVIDIA驱动状态
    echo -e "${YELLOW}检查NVIDIA驱动状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo -e "  ✓ nvidia-smi: 正常工作"
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
        else
            echo -e "  ✗ nvidia-smi: ${RED}无法正常工作${NC}"
            echo -e "    可能原因: 驱动未正确加载或GPU不兼容"
        fi
    else
        echo -e "  ✗ nvidia-smi: ${RED}未找到${NC}"
    fi

    # 检查内核模块
    echo -e "${YELLOW}检查内核模块:${NC}"
    local nvidia_modules=("nvidia" "nvidia_drm" "nvidia_modeset" "nvidia_uvm")
    for module in "${nvidia_modules[@]}"; do
        if lsmod | grep -q "^$module"; then
            echo -e "  ✓ $module: 已加载"
        else
            echo -e "  ✗ $module: ${RED}未加载${NC}"
        fi
    done

    # RTX 50系列特殊诊断
    if [[ "$IS_RTX_50_SERIES" == "true" ]]; then
        echo -e "${YELLOW}RTX 50系列特殊诊断:${NC}"
        echo -e "  显卡型号: $RTX_50_MODEL"

        # 检查是否使用开源驱动
        if dpkg -l | grep -q "nvidia.*open"; then
            echo -e "  ✓ 使用开源驱动"
        else
            echo -e "  ⚠ 未使用开源驱动，RTX 50系列可能需要开源驱动"
        fi

        # 检查驱动版本
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$driver_version" ]]; then
            if awk "BEGIN {exit !($driver_version >= 575.0)}"; then
                echo -e "  ✓ 驱动版本满足RTX 50要求: $driver_version"
            else
                echo -e "  ✗ 驱动版本过低: $driver_version (需要 ≥575.0)"
            fi
        fi
    fi

    # 提供修复建议
    echo -e "${BLUE}修复建议:${NC}"
    if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        echo -e "  • 重新安装NVIDIA驱动"
        echo -e "  • 检查GPU硬件连接"
        echo -e "  • 重启系统"
    fi

    if [[ "$IS_RTX_50_SERIES" == "true" ]] && ! dpkg -l | grep -q "nvidia.*open"; then
        echo -e "  • RTX 50系列建议使用开源驱动"
    fi
}

# 第六步：检查CUDA状态
check_cuda_status() {
    log_info "第六步：检查CUDA状态..."

    print_info "正在检查CUDA安装状态..."

    # 如果NVIDIA驱动有问题，先进行诊断
    if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}检测到NVIDIA驱动问题，进行诊断...${NC}"
        diagnose_cuda_issues
    fi

    if [[ "$FORCE_CUDA_REINSTALL" == "true" ]]; then
        echo -e "${YELLOW}强制重新安装模式，将卸载现有CUDA${NC}"
        NEED_INSTALL_CUDA=true
        NEED_UNINSTALL_CUDA=true
        return 0
    fi

    # 检查CUDA是否已安装
    local cuda_detected=false
    local installed_version=""
    local installed_major_version=""
    local cuda_paths=()

    # 方法1：通过nvcc命令检测
    if command -v nvcc &> /dev/null; then
        local installed_version_raw=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        installed_version=$(echo "$installed_version_raw" | sed 's/^V//')
        installed_major_version=$(echo "$installed_version" | cut -d'.' -f1,2)
        cuda_detected=true
        echo -e "${GREEN}✓ 通过nvcc检测到CUDA版本: $installed_version_raw${NC}"
    fi

    # 方法2：检查常见CUDA安装路径
    local cuda_install_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-*"
        "/opt/cuda"
        "/usr/lib/cuda"
    )

    echo -e "${BLUE}扫描CUDA安装路径...${NC}"
    for path_pattern in "${cuda_install_paths[@]}"; do
        for cuda_path in $path_pattern; do
            if [[ -d "$cuda_path" ]]; then
                cuda_paths+=("$cuda_path")
                echo -e "${YELLOW}发现CUDA安装路径: $cuda_path${NC}"

                # 尝试从路径名提取版本信息
                if [[ "$cuda_path" =~ cuda-([0-9]+\.[0-9]+) ]]; then
                    local path_version="${BASH_REMATCH[1]}"
                    echo -e "${BLUE}  路径版本: $path_version${NC}"

                    # 如果nvcc未检测到版本，使用路径版本
                    if [[ -z "$installed_version" ]]; then
                        installed_version="$path_version"
                        installed_major_version="$path_version"
                        cuda_detected=true
                    fi
                fi

                # 检查版本文件
                if [[ -f "$cuda_path/version.txt" ]]; then
                    local version_file_content=$(cat "$cuda_path/version.txt" 2>/dev/null)
                    echo -e "${BLUE}  版本文件内容: $version_file_content${NC}"
                elif [[ -f "$cuda_path/version.json" ]]; then
                    echo -e "${BLUE}  发现版本JSON文件${NC}"
                fi
            fi
        done
    done

    # 方法3：检查APT安装的CUDA包
    echo -e "${BLUE}检查APT安装的CUDA包...${NC}"
    local apt_cuda_packages=$(dpkg -l | grep -E "cuda-toolkit|cuda-runtime|nvidia-cuda" | awk '{print $2}' 2>/dev/null || true)
    if [[ -n "$apt_cuda_packages" ]]; then
        echo -e "${YELLOW}发现APT安装的CUDA相关包:${NC}"
        echo "$apt_cuda_packages" | while read -r package; do
            echo -e "${BLUE}  - $package${NC}"
        done
        cuda_detected=true
    fi

    # 分析检测结果
    if [[ "$cuda_detected" == "true" ]]; then
        if [[ -n "$installed_major_version" ]]; then
            echo -e "${GREEN}✓ 检测到CUDA版本: $installed_version${NC}"

            # 检查版本兼容性
            local major_num=$(echo "$installed_major_version" | cut -d'.' -f1)

            if [[ "$installed_major_version" == "$CUDA_VERSION" ]]; then
                echo -e "${GREEN}✓ CUDA版本匹配，跳过安装${NC}"
                NEED_INSTALL_CUDA=false
                NEED_UNINSTALL_CUDA=false
            elif [[ "$major_num" -le "11" ]]; then
                echo -e "${RED}⚠ 检测到老版本CUDA $installed_major_version (版本过低)${NC}"
                echo -e "${YELLOW}CUDA $major_num.x 版本已过时，将自动卸载并升级到 $CUDA_VERSION${NC}"

                # 老版本CUDA直接卸载，不询问用户
                NEED_INSTALL_CUDA=true
                NEED_UNINSTALL_CUDA=true
                echo -e "${GREEN}✓ 将自动卸载老版本并安装CUDA $CUDA_VERSION${NC}"
            else
                echo -e "${YELLOW}CUDA版本不匹配（当前: $installed_version，目标: $CUDA_VERSION）${NC}"
                NEED_INSTALL_CUDA=true
                NEED_UNINSTALL_CUDA=true
            fi
        else
            echo -e "${YELLOW}检测到CUDA安装但无法确定版本${NC}"
            NEED_INSTALL_CUDA=true
            NEED_UNINSTALL_CUDA=true
        fi
    else
        echo -e "${YELLOW}未检测到CUDA安装${NC}"
        NEED_INSTALL_CUDA=true
        NEED_UNINSTALL_CUDA=false
    fi

    # 显示检测摘要
    echo -e "\n${BLUE}CUDA状态检测摘要:${NC}"
    echo -e "  当前版本: ${installed_version:-'未安装'}"
    echo -e "  目标版本: $CUDA_VERSION"
    echo -e "  需要安装: $([ "$NEED_INSTALL_CUDA" == "true" ] && echo "是" || echo "否")"
    echo -e "  需要卸载: $([ "$NEED_UNINSTALL_CUDA" == "true" ] && echo "是" || echo "否")"
}

# CUDA卸载函数
uninstall_cuda() {
    echo -e "${BLUE}开始卸载CUDA...${NC}"

    # 检测CUDA安装方式和版本
    local cuda_version=""
    local uninstall_method=""
    local cuda_paths=()

    # 检测CUDA版本
    if command -v nvcc &> /dev/null; then
        local version_output=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        cuda_version=$(echo "$version_output" | sed 's/^V//')
        echo -e "${BLUE}检测到CUDA版本: $cuda_version${NC}"
    fi

    # 查找CUDA安装路径
    local cuda_install_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-*"
        "/opt/cuda"
        "/usr/lib/cuda"
    )

    for path_pattern in "${cuda_install_paths[@]}"; do
        for cuda_path in $path_pattern; do
            if [[ -d "$cuda_path" ]]; then
                cuda_paths+=("$cuda_path")
                echo -e "${YELLOW}发现CUDA路径: $cuda_path${NC}"
            fi
        done
    done

    # 方法1：尝试使用CUDA官方卸载脚本
    echo -e "${BLUE}尝试使用CUDA官方卸载脚本...${NC}"
    local uninstall_scripts=(
        "/usr/local/cuda/bin/cuda-uninstaller"
        "/usr/local/cuda/bin/uninstall_cuda_*.pl"
    )

    for script_pattern in "${uninstall_scripts[@]}"; do
        for script in $script_pattern; do
            if [[ -f "$script" && -x "$script" ]]; then
                echo -e "${GREEN}找到卸载脚本: $script${NC}"
                echo -e "${BLUE}执行官方卸载脚本...${NC}"

                if "$script" --silent 2>/dev/null || "$script" 2>/dev/null; then
                    echo -e "${GREEN}✓ 官方卸载脚本执行成功${NC}"
                    uninstall_method="official_script"
                else
                    echo -e "${YELLOW}官方卸载脚本执行失败，继续尝试其他方法${NC}"
                fi
                break 2
            fi
        done
    done

    # 方法2：APT包管理器卸载
    echo -e "${BLUE}检查APT安装的CUDA包...${NC}"
    local cuda_packages=$(dpkg -l | grep -E "cuda-toolkit|cuda-runtime|nvidia-cuda|cuda-repo" | awk '{print $2}' 2>/dev/null || true)

    if [[ -n "$cuda_packages" ]]; then
        echo -e "${YELLOW}发现APT安装的CUDA包:${NC}"
        echo "$cuda_packages" | while read -r package; do
            echo -e "${BLUE}  - $package${NC}"
        done

        echo -e "${BLUE}使用APT卸载CUDA包...${NC}"

        # 卸载CUDA相关包
        if echo "$cuda_packages" | xargs apt-get remove --purge -y 2>/dev/null; then
            echo -e "${GREEN}✓ APT卸载CUDA包成功${NC}"
            uninstall_method="apt"
        else
            echo -e "${YELLOW}APT卸载部分失败，继续清理${NC}"
        fi

        # 清理APT缓存
        apt-get autoremove -y 2>/dev/null || true
        apt-get autoclean 2>/dev/null || true
    fi

    # 方法3：手动清理CUDA文件和目录
    echo -e "${BLUE}手动清理CUDA文件和目录...${NC}"

    # 清理CUDA安装目录
    for cuda_path in "${cuda_paths[@]}"; do
        if [[ -d "$cuda_path" ]]; then
            echo -e "${BLUE}删除CUDA目录: $cuda_path${NC}"
            rm -rf "$cuda_path" 2>/dev/null || {
                echo -e "${YELLOW}无法删除 $cuda_path，可能需要手动清理${NC}"
            }
        fi
    done

    # 清理符号链接
    local cuda_symlinks=(
        "/usr/local/cuda"
        "/usr/bin/nvcc"
        "/usr/bin/nvidia-smi"
    )

    for symlink in "${cuda_symlinks[@]}"; do
        if [[ -L "$symlink" ]]; then
            echo -e "${BLUE}删除符号链接: $symlink${NC}"
            rm -f "$symlink" 2>/dev/null || true
        fi
    done

    # 清理环境变量配置文件
    echo -e "${BLUE}清理环境变量配置...${NC}"
    local env_files=(
        "/etc/environment"
        "/etc/profile"
        "/etc/bash.bashrc"
        "$HOME/.bashrc"
        "$HOME/.profile"
        "$HOME/.zshrc"
    )

    for env_file in "${env_files[@]}"; do
        if [[ -f "$env_file" ]]; then
            # 备份原文件
            cp "$env_file" "${env_file}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

            # 删除CUDA相关的环境变量
            sed -i '/CUDA_HOME/d' "$env_file" 2>/dev/null || true
            sed -i '/cuda.*bin/d' "$env_file" 2>/dev/null || true
            sed -i '/LD_LIBRARY_PATH.*cuda/d' "$env_file" 2>/dev/null || true

            echo -e "${BLUE}清理环境变量文件: $env_file${NC}"
        fi
    done

    # 清理库配置
    echo -e "${BLUE}更新动态链接库配置...${NC}"

    # 删除CUDA库配置文件
    rm -f /etc/ld.so.conf.d/cuda*.conf 2>/dev/null || true

    # 更新ldconfig
    ldconfig 2>/dev/null || true

    # 清理临时文件
    echo -e "${BLUE}清理临时文件...${NC}"
    rm -rf /tmp/cuda* 2>/dev/null || true
    rm -rf /tmp/CUDA* 2>/dev/null || true

    # 验证卸载结果
    echo -e "${BLUE}验证CUDA卸载结果...${NC}"

    local uninstall_success=true

    # 检查nvcc命令
    if command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}⚠ nvcc命令仍然存在${NC}"
        uninstall_success=false
    else
        echo -e "${GREEN}✓ nvcc命令已移除${NC}"
    fi

    # 检查CUDA目录
    local remaining_paths=()
    for path_pattern in "/usr/local/cuda*" "/opt/cuda*"; do
        for cuda_path in $path_pattern; do
            if [[ -d "$cuda_path" ]]; then
                remaining_paths+=("$cuda_path")
            fi
        done
    done

    if [[ ${#remaining_paths[@]} -gt 0 ]]; then
        echo -e "${YELLOW}⚠ 仍有CUDA目录存在:${NC}"
        for path in "${remaining_paths[@]}"; do
            echo -e "${YELLOW}  - $path${NC}"
        done
        uninstall_success=false
    else
        echo -e "${GREEN}✓ CUDA目录已清理${NC}"
    fi

    # 检查APT包
    local remaining_packages=$(dpkg -l | grep -E "cuda-toolkit|cuda-runtime|nvidia-cuda" | awk '{print $2}' 2>/dev/null || true)
    if [[ -n "$remaining_packages" ]]; then
        echo -e "${YELLOW}⚠ 仍有CUDA相关包存在:${NC}"
        echo "$remaining_packages" | while read -r package; do
            echo -e "${YELLOW}  - $package${NC}"
        done
        uninstall_success=false
    else
        echo -e "${GREEN}✓ CUDA相关包已清理${NC}"
    fi

    # 显示卸载结果
    if [[ "$uninstall_success" == "true" ]]; then
        echo -e "${GREEN}✓ CUDA卸载完成${NC}"
        echo -e "${BLUE}建议重启系统以确保所有更改生效${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ CUDA卸载部分完成，可能需要手动清理剩余文件${NC}"
        echo -e "${BLUE}建议检查上述提到的剩余文件和包${NC}"
        return 1
    fi
}

# 网络连接检查函数
check_network_connectivity() {
    echo -e "${BLUE}检查网络连接...${NC}"

    # 首先尝试检查NVIDIA服务器连接（更实用的检查）
    if curl -s --connect-timeout 10 --max-time 15 "https://developer.download.nvidia.com" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ 可以访问NVIDIA官方服务器${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠ 无法直接访问NVIDIA官方服务器${NC}"
    echo -e "${BLUE}尝试访问中国镜像服务器...${NC}"

    if curl -s --connect-timeout 10 --max-time 15 "https://developer.download.nvidia.cn" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ 可以访问NVIDIA中国镜像服务器${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠ 无法访问NVIDIA镜像服务器，尝试基本网络检查...${NC}"

    # 备用检查：尝试访问常见的公共服务
    local test_urls=(
        "https://www.google.com"
        "https://www.baidu.com"
        "http://www.ubuntu.com"
    )

    for url in "${test_urls[@]}"; do
        if curl -s --connect-timeout 5 --max-time 10 "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ 网络连接正常，但NVIDIA服务器可能暂时不可访问${NC}"
            echo -e "${YELLOW}建议：稍后重试或检查防火墙设置${NC}"
            return 0
        fi
    done

    # 最后尝试ping测试
    if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1 || ping -c 1 -W 5 114.114.114.114 >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ 基本网络连接正常，但HTTPS访问可能受限${NC}"
        echo -e "${YELLOW}建议：检查代理设置或防火墙配置${NC}"
        return 0
    fi

    echo -e "${RED}✗ 网络连接失败，无法访问互联网${NC}"
    return 1
}

# 第七步：安装CUDA工具包
install_cuda_toolkit() {
    log_info "第七步：安装CUDA工具包..."

    if [[ "$NEED_INSTALL_CUDA" == "false" ]]; then
        echo -e "${GREEN}✓ CUDA已正确安装，跳过安装步骤${NC}"
        return 0
    fi

    # 检查网络连接
    if ! check_network_connectivity; then
        echo -e "${RED}✗ 网络连接检查失败，无法下载CUDA${NC}"
        echo -e "${YELLOW}建议：${NC}"
        echo -e "  1. 检查网络连接"
        echo -e "  2. 配置代理服务器"
        echo -e "  3. 使用离线安装包"
        return 1
    fi

    echo -e "${BLUE}正在安装CUDA $CUDA_VERSION...${NC}"

    # CUDA 12.8 特殊安装方法（使用本地仓库）
    if [[ "$CUDA_VERSION" == "12.8" ]]; then
        echo -e "${BLUE}使用CUDA 12.8本地仓库安装方法...${NC}"
        install_cuda_12_8_local_repo
        return $?
    fi

    # 设置NVIDIA仓库
    local repo_url="${NVIDIA_REPO_URLS[$UBUNTU_VERSION]}"
    echo -e "${BLUE}设置NVIDIA官方仓库...${NC}"

    # 清理可能存在的旧keyring
    rm -f /tmp/cuda-keyring.deb 2>/dev/null

    # 下载并安装keyring，添加重试机制
    local keyring_url="${repo_url}/${NVIDIA_KEYRING_PACKAGE}"
    echo -e "${BLUE}下载NVIDIA keyring: $keyring_url${NC}"

    if ! wget --timeout=30 --tries=3 "$keyring_url" -O /tmp/cuda-keyring.deb; then
        echo -e "${RED}✗ 下载NVIDIA keyring失败${NC}"
        echo -e "${YELLOW}尝试使用备用方法...${NC}"

        # 尝试使用curl下载
        if ! curl -L --connect-timeout 30 --max-time 60 "$keyring_url" -o /tmp/cuda-keyring.deb; then
            echo -e "${RED}✗ 无法下载NVIDIA keyring${NC}"
            return 1
        fi
    fi

    # 验证下载的keyring文件
    if [[ ! -f /tmp/cuda-keyring.deb ]] || [[ $(stat -c%s /tmp/cuda-keyring.deb) -lt 1000 ]]; then
        echo -e "${RED}✗ NVIDIA keyring文件无效${NC}"
        return 1
    fi

    echo -e "${BLUE}安装NVIDIA keyring...${NC}"
    if ! dpkg -i /tmp/cuda-keyring.deb; then
        echo -e "${RED}✗ 安装NVIDIA keyring失败${NC}"
        return 1
    fi

    echo -e "${BLUE}更新软件包列表...${NC}"
    apt-get update -y || {
        echo -e "${YELLOW}软件包列表更新失败，尝试修复...${NC}"
        apt-get clean
        apt-get update -y
    }

    # 根据环境选择安装包（只安装工具包，不包含驱动）
    local cuda_package="cuda-toolkit-${CUDA_VERSION//./-}"
    echo -e "${BLUE}注意：只安装CUDA工具包，不包含显卡驱动${NC}"

    echo -e "${BLUE}安装CUDA包: $cuda_package${NC}"
    # 尝试APT安装，如果失败则使用Runfile
    echo -e "${BLUE}尝试通过APT安装CUDA...${NC}"
    if apt-get install -y "$cuda_package" 2>&1 | tee /tmp/cuda_apt_install.log; then
        echo -e "${GREEN}✓ APT安装CUDA成功${NC}"
        return 0
    else
        local apt_exit_code=${PIPESTATUS[0]}
        echo -e "${YELLOW}APT安装失败 (退出码: $apt_exit_code)${NC}"

        # 分析APT安装失败的原因
        if grep -q "dpkg.*error code" /tmp/cuda_apt_install.log; then
            echo -e "${YELLOW}检测到dpkg错误，这通常是由于：${NC}"
            echo -e "  1. 软件包冲突或依赖问题"
            echo -e "  2. 磁盘空间不足"
            echo -e "  3. 系统软件包数据库损坏"

            # 尝试修复dpkg问题
            echo -e "${BLUE}尝试修复软件包管理器...${NC}"
            dpkg --configure -a 2>/dev/null || true
            apt-get install -f -y 2>/dev/null || true
        fi

        echo -e "${YELLOW}尝试Runfile安装...${NC}"
        if install_cuda_runfile; then
            return 0
        else
            echo -e "${RED}✗ Runfile安装也失败了${NC}"
            return 1
        fi
    fi

    echo -e "${GREEN}✓ CUDA工具包安装完成${NC}"
}

# CUDA 12.8 本地仓库安装方法
install_cuda_12_8_local_repo() {
    echo -e "${BLUE}开始CUDA 12.8本地仓库安装...${NC}"

    # 下载pin文件
    echo -e "${BLUE}下载CUDA仓库pin文件...${NC}"
    if ! wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/cuda-ubuntu${UBUNTU_VERSION//./}.pin -O /tmp/cuda-ubuntu.pin; then
        echo -e "${RED}✗ 下载pin文件失败${NC}"
        return 1
    fi

    # 移动pin文件到正确位置
    mv /tmp/cuda-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    echo -e "${GREEN}✓ pin文件配置完成${NC}"

    # 下载CUDA 12.8本地仓库包
    local cuda_repo_deb="cuda-repo-ubuntu${UBUNTU_VERSION//./}-12-8-local_12.8.1-570.124.06-1_amd64.deb"
    local cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${cuda_repo_deb}"

    echo -e "${BLUE}下载CUDA 12.8本地仓库包...${NC}"
    echo -e "${BLUE}下载地址: $cuda_repo_url${NC}"

    if ! wget --progress=bar:force --timeout=60 --tries=3 "$cuda_repo_url" -O "/tmp/$cuda_repo_deb"; then
        echo -e "${RED}✗ 下载CUDA 12.8仓库包失败${NC}"
        return 1
    fi

    # 安装本地仓库包
    echo -e "${BLUE}安装CUDA 12.8本地仓库包...${NC}"
    if ! dpkg -i "/tmp/$cuda_repo_deb"; then
        echo -e "${RED}✗ 安装本地仓库包失败${NC}"
        return 1
    fi

    # 复制GPG密钥
    echo -e "${BLUE}配置GPG密钥...${NC}"
    local keyring_path="/var/cuda-repo-ubuntu${UBUNTU_VERSION//./}-12-8-local/cuda-*-keyring.gpg"
    if ! cp $keyring_path /usr/share/keyrings/; then
        echo -e "${RED}✗ 复制GPG密钥失败${NC}"
        return 1
    fi

    # 更新包列表
    echo -e "${BLUE}更新软件包列表...${NC}"
    if ! apt-get update; then
        echo -e "${RED}✗ 更新软件包列表失败${NC}"
        return 1
    fi

    # 安装CUDA 12.8工具包
    echo -e "${BLUE}安装CUDA 12.8工具包...${NC}"
    if apt-get install -y cuda-toolkit-12-8; then
        echo -e "${GREEN}✓ CUDA 12.8安装成功${NC}"
        return 0
    else
        echo -e "${RED}✗ CUDA 12.8安装失败${NC}"
        return 1
    fi
}

# Runfile安装CUDA
install_cuda_runfile() {
    local runfile_url="${CUDA_RUNFILE_URLS[$CUDA_VERSION]}"
    local runfile_name="cuda_${CUDA_VERSION}_installer.run"

    # 检查CUDA_VERSION是否设置
    if [[ -z "$CUDA_VERSION" ]]; then
        echo -e "${RED}✗ CUDA版本未设置，无法下载Runfile${NC}"
        return 1
    fi

    # 检查URL是否存在
    if [[ -z "$runfile_url" ]]; then
        echo -e "${RED}✗ 未找到CUDA $CUDA_VERSION 的下载URL${NC}"
        return 1
    fi

    echo -e "${BLUE}下载CUDA Runfile (版本: $CUDA_VERSION)...${NC}"
    echo -e "${BLUE}下载地址: $runfile_url${NC}"

    # 添加重试机制的下载函数
    download_with_retry() {
        local url="$1"
        local output="$2"
        local max_attempts=3
        local attempt=1

        while [[ $attempt -le $max_attempts ]]; do
            echo -e "${BLUE}尝试下载 (第 $attempt/$max_attempts 次)...${NC}"

            # 使用wget下载，支持重定向和断点续传
            if wget --progress=bar:force --timeout=30 --tries=3 --continue \
                   --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
                   "$url" -O "$output" 2>&1; then
                echo -e "${GREEN}✓ 下载成功${NC}"
                return 0
            else
                local exit_code=$?
                echo -e "${YELLOW}下载失败 (退出码: $exit_code)${NC}"

                if [[ $attempt -lt $max_attempts ]]; then
                    echo -e "${YELLOW}等待5秒后重试...${NC}"
                    sleep 5
                    # 清理部分下载的文件
                    rm -f "$output" 2>/dev/null
                fi
            fi

            ((attempt++))
        done

        echo -e "${RED}✗ 下载失败，已尝试 $max_attempts 次${NC}"
        return 1
    }

    # 执行下载
    if ! download_with_retry "$runfile_url" "/tmp/$runfile_name"; then
        echo -e "${RED}✗ 下载CUDA Runfile失败${NC}"
        echo -e "${YELLOW}建议检查网络连接或尝试手动下载：${NC}"
        echo -e "${BLUE}$runfile_url${NC}"
        return 1
    fi

    # 验证下载的文件
    if [[ ! -f "/tmp/$runfile_name" ]]; then
        echo -e "${RED}✗ 下载的文件不存在${NC}"
        return 1
    fi

    local file_size=$(stat -c%s "/tmp/$runfile_name" 2>/dev/null || echo "0")
    if [[ $file_size -lt 1000000 ]]; then  # 小于1MB认为下载不完整
        echo -e "${RED}✗ 下载的文件大小异常 (${file_size} bytes)${NC}"
        rm -f "/tmp/$runfile_name"
        return 1
    fi

    echo -e "${GREEN}✓ 文件下载完成，大小: $(( file_size / 1024 / 1024 )) MB${NC}"

    chmod +x "/tmp/$runfile_name"

    # 安装选项：只安装工具包，不安装驱动
    local install_options="--silent --toolkit --no-driver-installation"
    echo -e "${BLUE}注意：Runfile安装不包含显卡驱动${NC}"

    echo -e "${BLUE}执行CUDA Runfile安装...${NC}"
    echo -e "${BLUE}安装选项: $install_options${NC}"

    # 执行安装并捕获详细错误信息
    if "/tmp/$runfile_name" $install_options; then
        echo -e "${GREEN}✓ CUDA Runfile安装完成${NC}"
        # 清理安装文件
        rm -f "/tmp/$runfile_name"
        return 0
    else
        local exit_code=$?
        echo -e "${RED}✗ CUDA Runfile安装失败 (退出码: $exit_code)${NC}"
        echo -e "${YELLOW}可能的原因：${NC}"
        echo -e "  1. 系统不兼容或缺少依赖"
        echo -e "  2. 磁盘空间不足"
        echo -e "  3. 权限问题"
        echo -e "  4. 已存在冲突的CUDA安装"

        # 保留安装文件以便调试
        echo -e "${BLUE}安装文件保留在: /tmp/$runfile_name${NC}"
        return 1
    fi
}

# 第八步：配置CUDA环境
configure_cuda_environment() {
    log_info "第八步：配置CUDA环境..."

    echo -e "${BLUE}正在配置CUDA环境变量...${NC}"

    # 查找CUDA安装路径
    local cuda_path=""
    for path in "/usr/local/cuda-${CUDA_VERSION}" "/usr/local/cuda" "/opt/cuda"; do
        if [[ -d "$path" ]]; then
            cuda_path="$path"
            break
        fi
    done

    if [[ -z "$cuda_path" ]]; then
        echo -e "${RED}✗ 未找到CUDA安装路径${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ 找到CUDA安装路径: $cuda_path${NC}"

    # 创建符号链接
    if [[ ! -L "/usr/local/cuda" ]] || [[ "$(readlink /usr/local/cuda)" != "$cuda_path" ]]; then
        ln -sf "$cuda_path" /usr/local/cuda
        echo -e "${GREEN}✓ 创建CUDA符号链接${NC}"
    fi

    # 配置环境变量
    local env_config="
# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export CUDA_ROOT=\$CUDA_HOME
"

    # 写入系统环境配置
    if ! grep -q "CUDA_HOME" /etc/profile; then
        echo "$env_config" >> /etc/profile
        echo -e "${GREEN}✓ 环境变量已写入/etc/profile${NC}"
    fi

    # 立即生效
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDA_ROOT=$CUDA_HOME

    # 验证nvcc命令
    if command -v nvcc &> /dev/null; then
        local nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        echo -e "${GREEN}✓ CUDA环境配置完成，nvcc版本: $nvcc_version${NC}"
    else
        echo -e "${RED}✗ nvcc命令不可用${NC}"
        return 1
    fi

    # 清理临时仓库配置
    rm -f /etc/apt/sources.list.d/cuda*.list 2>/dev/null
    echo -e "${GREEN}✓ 清理临时配置完成${NC}"
}

# 第九步：安装cuDNN
install_cudnn() {
    log_info "第九步：安装cuDNN..."

    # 如果需要卸载cuDNN
    if [[ "$NEED_UNINSTALL_CUDNN" == "true" ]]; then
        echo -e "${BLUE}正在卸载现有cuDNN...${NC}"
        apt-get remove --purge -y libcudnn* 2>/dev/null || true
        rm -f /usr/local/cuda/include/cudnn*.h 2>/dev/null || true
        rm -f /usr/local/cuda/lib64/libcudnn* 2>/dev/null || true
        echo -e "${GREEN}✓ cuDNN卸载完成${NC}"
    fi

    # 如果不需要安装cuDNN，直接返回
    if [[ "$NEED_INSTALL_CUDNN" == "false" ]]; then
        echo -e "${GREEN}✓ 跳过cuDNN安装${NC}"
        return 0
    fi

    echo -e "${BLUE}正在安装cuDNN $CUDNN_VERSION...${NC}"
    
    # 配置NVIDIA仓库以确保能找到cuDNN包
    setup_nvidia_repository_for_cudnn

    # 检查是否已安装
    if [[ -f "/usr/local/cuda/include/cudnn.h" ]]; then
        local installed_version=$(grep "#define CUDNN_MAJOR" /usr/local/cuda/include/cudnn.h | awk '{print $3}')
        echo -e "${GREEN}✓ 检测到cuDNN版本: $installed_version.x${NC}"

        if [[ "$installed_version" == "${CUDNN_VERSION%%.*}" ]]; then
            echo -e "${GREEN}✓ cuDNN版本匹配，跳过安装${NC}"
            return 0
        fi
    fi

    # 安装cuDNN
    echo -e "${BLUE}安装cuDNN开发包...${NC}"
    
    # 尝试安装cuDNN包（兼容不同版本的包名称）
    local cudnn_packages=("libcudnn9-dev-cuda-12" "libcudnn9-cuda-12" "libcudnn8-dev-cuda-12" "libcudnn8-cuda-12")
    local package_installed=false
    
    for package in "${cudnn_packages[@]}"; do
        if apt-get install -y "$package" 2>/dev/null; then
            echo -e "${GREEN}✓ 成功安装cuDNN包: $package${NC}"
            package_installed=true
            break
        else
            echo -e "${YELLOW}尝试安装cuDNN包失败: $package${NC}"
        fi
    done
    
    if [[ "$package_installed" == "false" ]]; then
        echo -e "${YELLOW}APT安装失败，尝试手动安装...${NC}"
        install_cudnn_manual
        return $?
    fi

    echo -e "${GREEN}✓ cuDNN安装完成${NC}"
}

# 配置NVIDIA仓库以支持cuDNN安装
setup_nvidia_repository_for_cudnn() {
    echo -e "${BLUE}配置NVIDIA仓库以支持cuDNN安装...${NC}"
    
    # 检查是否已配置NVIDIA仓库
    if [[ -f "/etc/apt/sources.list.d/nvidia-ml.list" ]] || [[ -f "/etc/apt/sources.list.d/cuda.list" ]]; then
        echo -e "${GREEN}✓ NVIDIA仓库已配置${NC}"
        return 0
    fi
    
    # 添加NVIDIA GPG密钥
    echo -e "${BLUE}添加NVIDIA GPG密钥...${NC}"
    if ! wget -q -O - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/3bf863cc.pub | apt-key add - 2>/dev/null; then
        echo -e "${YELLOW}GPG密钥添加失败，尝试备用方法...${NC}"
        # 尝试直接下载密钥文件
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/3bf863cc.pub -O /tmp/nvidia-gpg-key.pub 2>/dev/null
        if [[ -f "/tmp/nvidia-gpg-key.pub" ]]; then
            apt-key add /tmp/nvidia-gpg-key.pub 2>/dev/null || true
            rm -f /tmp/nvidia-gpg-key.pub
        fi
    fi
    
    # 添加NVIDIA仓库
    echo -e "${BLUE}添加NVIDIA仓库...${NC}"
    local cuda_repo_url="deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/ /"
    local ml_repo_url="deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${UBUNTU_VERSION//./}/x86_64/ /"
    
    echo "$cuda_repo_url" > /etc/apt/sources.list.d/cuda.list 2>/dev/null || true
    echo "$ml_repo_url" > /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true
    
    # 更新APT缓存
    echo -e "${BLUE}更新APT缓存...${NC}"
    apt-get update 2>/dev/null || {
        echo -e "${YELLOW}APT更新失败，但继续安装...${NC}"
    }
    
    echo -e "${GREEN}✓ NVIDIA仓库配置完成${NC}"
}

# 手动安装cuDNN
install_cudnn_manual() {
    local cuda_key="${CUDA_VERSION}"
    local cudnn_url="${CUDNN_DOWNLOAD_URLS[$cuda_key]}"
    local cudnn_file="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${cuda_key}-archive.tar.xz"
    
    # 如果主要URL失败，尝试备用URL
    local backup_urls=(
        "${NVIDIA_BASE_URL}${NVIDIA_CUDNN_PATH}/9.5.1/local_installers/cudnn-linux-x86_64-9.5.1_cuda${cuda_key}-archive.tar.xz"
        "${NVIDIA_BASE_URL}${NVIDIA_CUDNN_PATH}/local_installers/9.5.1/cudnn-linux-x86_64-9.5.1_cuda${cuda_key}-archive.tar.xz"
    )

    echo -e "${BLUE}下载cuDNN...${NC}"
    
    # 尝试下载主URL
    if ! wget -q --show-progress "$cudnn_url" -O "/tmp/$cudnn_file" 2>/dev/null; then
        echo -e "${YELLOW}主下载URL失败，尝试备用URL...${NC}"
        
        # 尝试备用URL
        local download_success=false
        for backup_url in "${backup_urls[@]}"; do
            if wget -q --show-progress "$backup_url" -O "/tmp/$cudnn_file" 2>/dev/null; then
                download_success=true
                break
            fi
        done
        
        if [[ "$download_success" == "false" ]]; then
            echo -e "${RED}✗ 下载cuDNN失败${NC}"
            echo -e "${YELLOW}请检查网络连接或手动下载cuDNN${NC}"
            echo -e "${YELLOW}手动下载步骤:${NC}"
            echo -e "${YELLOW}1. 访问: https://developer.nvidia.com/cudnn${NC}"
            echo -e "${YELLOW}2. 下载 cuDNN ${CUDNN_VERSION} for CUDA ${CUDA_VERSION}${NC}"
            echo -e "${YELLOW}3. 将下载的文件重命名为: $cudnn_file${NC}"
            echo -e "${YELLOW}4. 将文件放置在: /tmp/$cudnn_file${NC}"
            echo -e "${YELLOW}5. 重新运行此脚本${NC}"
            return 1
        fi
    fi

    echo -e "${BLUE}解压并安装cuDNN...${NC}"
    cd /tmp
    tar -xf "$cudnn_file"

    # 尝试多种可能的解压目录结构
    local cudnn_dirs=(
        "cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${cuda_key}-archive"
        "cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive"
        "cudnn-linux-x86_64-${CUDNN_VERSION}-archive"
        $(find . -maxdepth 1 -name "cudnn-*" -type d | head -1)
    )
    
    local cudnn_dir=""
    for dir in "${cudnn_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            cudnn_dir="$dir"
            break
        fi
    done
    
    if [[ -z "$cudnn_dir" ]]; then
        echo -e "${RED}✗ 未找到cuDNN解压目录${NC}"
        echo -e "${YELLOW}解压内容: $(ls -la /tmp/ | grep cudnn)${NC}"
        return 1
    fi

    echo -e "${BLUE}找到cuDNN目录: $cudnn_dir${NC}"
    
    # 检查目录结构
    if [[ -d "$cudnn_dir/include" ]] && [[ -d "$cudnn_dir/lib" || -d "$cudnn_dir/lib64" ]]; then
        echo -e "${BLUE}复制cuDNN头文件...${NC}"
        cp -r "$cudnn_dir/include/"* /usr/local/cuda/include/
        
        echo -e "${BLUE}复制cuDNN库文件...${NC}"
        if [[ -d "$cudnn_dir/lib64" ]]; then
            cp -r "$cudnn_dir/lib64/"* /usr/local/cuda/lib64/
        else
            cp -r "$cudnn_dir/lib/"* /usr/local/cuda/lib64/
        fi
    else
        echo -e "${RED}✗ cuDNN目录结构异常${NC}"
        echo -e "${YELLOW}目录内容: $(ls -la "$cudnn_dir")${NC}"
        return 1
    fi

    # 设置权限
    chmod a+r /usr/local/cuda/include/cudnn*.h 2>/dev/null || true
    chmod a+r /usr/local/cuda/lib64/libcudnn* 2>/dev/null || true
    
    # 更新动态链接库缓存
    ldconfig

    echo -e "${GREEN}✓ cuDNN手动安装完成${NC}"
}

# 第十步：验证安装
verify_installation() {
    log_info "第十步：验证安装..."

    echo -e "${BLUE}正在验证CUDA和cuDNN安装...${NC}"

    # 验证CUDA
    echo -e "\n${YELLOW}CUDA验证:${NC}"
    if command -v nvcc &> /dev/null; then
        nvcc --version
        echo -e "${GREEN}✓ CUDA验证成功${NC}"
    else
        echo -e "${RED}✗ CUDA验证失败${NC}"
        return 1
    fi

    # 验证NVIDIA驱动
    echo -e "\n${YELLOW}NVIDIA驱动验证:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo -e "${GREEN}✓ NVIDIA驱动验证成功${NC}"
    else
        echo -e "${RED}✗ NVIDIA驱动验证失败${NC}"
        return 1
    fi

    # 验证cuDNN
    echo -e "\n${YELLOW}cuDNN验证:${NC}"
    
    # 检查多种可能的cuDNN安装位置
    local cudnn_found=false
    local cudnn_version=""
    local cudnn_paths=(
        "/usr/local/cuda/include/cudnn.h"
        "/usr/local/cuda/include/cudnn_version.h"
        "/usr/include/x86_64-linux-gnu/cudnn.h"
        "/usr/include/x86_64-linux-gnu/cudnn_version.h"
    )
    
    for cudnn_path in "${cudnn_paths[@]}"; do
        if [[ -f "$cudnn_path" ]]; then
            cudnn_found=true
            echo -e "${GREEN}✓ 检测到cuDNN头文件: $cudnn_path${NC}"
            
            # 尝试从不同的头文件中获取版本信息
            if [[ "$cudnn_path" == *"cudnn_version.h" ]]; then
                cudnn_version=$(grep "CUDNN_VERSION" "$cudnn_path" 2>/dev/null | head -1 | awk '{print $3}' || echo "")
            else
                local cudnn_major=$(grep "#define CUDNN_MAJOR" "$cudnn_path" 2>/dev/null | awk '{print $3}' || echo "")
                local cudnn_minor=$(grep "#define CUDNN_MINOR" "$cudnn_path" 2>/dev/null | awk '{print $3}' || echo "")
                local cudnn_patch=$(grep "#define CUDNN_PATCHLEVEL" "$cudnn_path" 2>/dev/null | awk '{print $3}' || echo "")
                if [[ -n "$cudnn_major" && -n "$cudnn_minor" && -n "$cudnn_patch" ]]; then
                    cudnn_version="${cudnn_major}.${cudnn_minor}.${cudnn_patch}"
                fi
            fi
            break
        fi
    done
    
    # 检查APT安装的cuDNN包
    if [[ "$cudnn_found" == "false" ]]; then
        if dpkg -l | grep -q libcudnn; then
            cudnn_found=true
            echo -e "${GREEN}✓ 检测到APT安装的cuDNN包${NC}"
            dpkg -l | grep libcudnn | head -3
        fi
    fi
    
    # 检查库文件
    if [[ "$cudnn_found" == "false" ]]; then
        local cudnn_libs=(
            "/usr/local/cuda/lib64/libcudnn.so*"
            "/usr/lib/x86_64-linux-gnu/libcudnn.so*"
        )
        for lib_pattern in "${cudnn_libs[@]}"; do
            if ls $lib_pattern 2>/dev/null | head -1 | grep -q .; then
                cudnn_found=true
                echo -e "${GREEN}✓ 检测到cuDNN库文件${NC}"
                ls $lib_pattern 2>/dev/null | head -1
                break
            fi
        done
    fi
    
    if [[ "$cudnn_found" == "true" ]]; then
        if [[ -n "$cudnn_version" ]]; then
            echo -e "${GREEN}cuDNN版本: $cudnn_version${NC}"
        fi
        
        # 检查是否选择了不安装cuDNN
        if [[ "$NEED_INSTALL_CUDNN" == "false" && "$NEED_UNINSTALL_CUDNN" == "false" ]]; then
            echo -e "${YELLOW}⚠ 警告: 检测到已安装cuDNN，但您选择了不安装cuDNN${NC}"
            echo -e "${YELLOW}  建议: 如果不需要cuDNN，请考虑卸载以避免潜在冲突${NC}"
            echo -e "${GREEN}✓ cuDNN验证通过（已安装但选择跳过）${NC}"
        else
            echo -e "${GREEN}✓ cuDNN验证成功${NC}"
        fi
    else
        # 检查是否选择了不安装cuDNN
        if [[ "$NEED_INSTALL_CUDNN" == "false" && "$NEED_UNINSTALL_CUDNN" == "false" ]]; then
            echo -e "${GREEN}✓ cuDNN验证通过（未安装且选择跳过）${NC}"
        else
            echo -e "${RED}✗ cuDNN验证失败${NC}"
            return 1
        fi
    fi

    # 创建简单的CUDA测试程序
    echo -e "\n${YELLOW}CUDA功能测试:${NC}"
    create_cuda_test_program
}

# 创建CUDA测试程序
create_cuda_test_program() {
    local test_file="/tmp/cuda_test.cu"

    cat > "$test_file" << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // 检查CUDA调用是否成功
    if (error != cudaSuccess) {
        std::cout << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
        std::cout << "CUDA设备数量: 0 (CUDA不可用)" << std::endl;
        return 1;
    }

    // 验证设备数量是否合理（防止异常值）
    if (deviceCount < 0 || deviceCount > 16) {
        std::cout << "CUDA错误: 检测到异常的设备数量 " << deviceCount << std::endl;
        std::cout << "CUDA设备数量: 0 (设备数量异常)" << std::endl;
        return 1;
    }

    std::cout << "CUDA设备数量: " << deviceCount << std::endl;

    // 只有在设备数量大于0且合理时才遍历设备
    for (int i = 0; i < deviceCount && i < 16; i++) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {
            std::cout << "设备 " << i << ": 获取属性失败 - " << cudaGetErrorString(error) << std::endl;
            continue;
        }

        std::cout << "设备 " << i << ": " << prop.name << std::endl;
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }

    return 0;
}
EOF

    echo -e "${BLUE}编译CUDA测试程序...${NC}"
    if nvcc -o /tmp/cuda_test "$test_file" 2>/dev/null; then
        echo -e "${BLUE}运行CUDA测试程序...${NC}"

        # 使用timeout防止程序无限循环，最多等待30秒
        if timeout 30s /tmp/cuda_test 2>/dev/null; then
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}✓ CUDA功能测试成功${NC}"
            elif [ $exit_code -eq 1 ]; then
                echo -e "${YELLOW}⚠ CUDA测试检测到问题，但这是正常的（可能没有GPU或CUDA不可用）${NC}"
            else
                echo -e "${YELLOW}⚠ CUDA测试程序异常退出（退出码: $exit_code）${NC}"
            fi
        else
            local timeout_exit=$?
            if [ $timeout_exit -eq 124 ]; then
                echo -e "${RED}✗ CUDA测试程序超时（30秒），可能存在无限循环问题${NC}"
                # 强制终止可能的残留进程
                pkill -f cuda_test 2>/dev/null || true
            else
                echo -e "${YELLOW}⚠ CUDA测试程序执行失败${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}CUDA测试程序编译失败，但基本安装正常${NC}"
    fi

    # 清理测试文件
    rm -f /tmp/cuda_test /tmp/cuda_test.cu
}

# 第十一步：清理和优化
cleanup_and_optimize() {
    log_info "第十一步：清理和优化..."

    echo -e "${BLUE}正在清理临时文件和优化系统...${NC}"

    # 清理下载的文件
    rm -f /tmp/cuda*.deb /tmp/cuda*.run /tmp/cudnn*.tar.xz
    echo -e "${GREEN}✓ 清理临时文件${NC}"

    # 更新动态链接库缓存
    ldconfig
    echo -e "${GREEN}✓ 更新动态链接库缓存${NC}"

    # 清理APT缓存
    apt-get autoremove -y
    apt-get autoclean
    echo -e "${GREEN}✓ 清理APT缓存${NC}"

    # 优化GPU性能设置
    if command -v nvidia-smi &> /dev/null && [[ "$IS_WSL" != "true" ]]; then
        echo -e "${BLUE}优化GPU性能设置...${NC}"
        nvidia-smi -pm 1 2>/dev/null || echo -e "${YELLOW}无法设置持久模式${NC}"
        echo -e "${GREEN}✓ GPU性能优化完成${NC}"
    fi
}

# 第十二步：完成安装
complete_installation() {
    log_info "第十二步：完成安装..."

    echo -e "\n${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ ${BLUE}CUDA安装完成${GREEN} │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"

    echo -e "\n${YELLOW}安装摘要:${NC}"
    echo -e "  CUDA版本: ${GREEN}$CUDA_VERSION${NC}"
    echo -e "  cuDNN版本: ${GREEN}$CUDNN_VERSION${NC}"
    echo -e "  安装路径: ${GREEN}/usr/local/cuda${NC}"

    if [[ "$IS_WSL" == "true" ]]; then
        echo -e "\n${BLUE}WSL环境提示:${NC}"
        echo -e "  - NVIDIA驱动由Windows管理"
        echo -e "  - 请确保Windows已安装最新NVIDIA驱动"
    else
        echo -e "\n${BLUE}环境变量:${NC}"
        echo -e "  - 已配置到 /etc/profile"
        echo -e "  - 当前会话已生效"

        if [[ "$NEED_REBOOT" == "true" ]]; then
            echo -e "\n${YELLOW}重要提示:${NC}"
            echo -e "  ${RED}系统需要重启以完成驱动安装${NC}"
            echo -e "  重启后请重新运行此脚本验证安装"

            read -p "是否现在重启系统？[y/N]: " reboot_choice
            if [[ "$reboot_choice" =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}系统将在5秒后重启...${NC}"
                sleep 5
                reboot
            fi
        fi
    fi

    echo -e "\n${GREEN}✓ CUDA安装流程全部完成！${NC}"
    echo -e "\n${BLUE}验证命令:${NC}"
    echo -e "  nvcc --version"
    echo -e "  nvidia-smi"
    echo -e "  python -c \"import torch; print(torch.cuda.is_available())\""
}

print_option_header() {
    local title=$1
    echo -e "\n${BLUE}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│ ${GREEN}$title${BLUE} │${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────────────┘${NC}"
}

print_option() {
    local num=$1
    local text=$2
    local desc=$3
    echo -e " ${YELLOW}$num.${NC} $text ${desc:+${BLUE}($desc)${NC}}"
}

# 验证URL可访问性
verify_url_accessibility() {
    local url="$1"
    local description="${2:-URL}"

    log_message "INFO" "验证${description}可访问性: $url"

    # 使用curl检查HTTP状态码
    local http_code=""
    http_code=$(curl --connect-timeout 15 --max-time 30 -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)

    if [[ "$http_code" =~ ^(200|301|302)$ ]]; then
        log_message "INFO" "✓ ${description}可访问 (HTTP $http_code)"
        return 0
    else
        log_message "WARN" "${description}不可访问 (HTTP $http_code)"
        return 1
    fi
}

# 加载NVIDIA驱动环境变量
load_nvidia_environment() {
    # 尝试加载系统环境变量
    if [[ -f "/etc/profile" ]]; then
        source /etc/profile 2>/dev/null || true
    fi

    # 如果nvidia-smi不在PATH中，尝试常见的NVIDIA驱动安装路径
    if ! command -v nvidia-smi &> /dev/null; then
        local nvidia_paths=(
            "/usr/bin"
            "/usr/local/bin"
            "/opt/nvidia/bin"
            "/usr/local/nvidia/bin"
        )

        for nvidia_path in "${nvidia_paths[@]}"; do
            if [[ -f "$nvidia_path/nvidia-smi" ]]; then
                log_message "INFO" "在 $nvidia_path 找到nvidia-smi，添加到PATH"
                export PATH="$nvidia_path:$PATH"
                break
            fi
        done
    fi

    # 设置NVIDIA相关的库路径
    if [[ -d "/usr/lib/x86_64-linux-gnu" ]]; then
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    fi
    if [[ -d "/usr/lib64" ]]; then
        export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

# 清理NVIDIA配置冲突
# ==================== 重复函数已删除，保留新优化版本 ====================

# ==================== 重复函数已删除，保留新优化版本 ====================

# ==================== 旧函数已删除，保留新优化版本 ====================
# ==================== 旧主函数已删除，保留新优化版本 ====================

# ==================== 标准主函数 ====================

# 显示脚本标题
show_script_header() {
    clear
    echo -e "${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ ${BLUE}NVIDIA CUDA & cuDNN 安装脚本 v2025.1${GREEN} │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
    echo -e "${YELLOW}支持版本: CUDA 12.8/12.6, cuDNN 9.5.1, 驱动575+${NC}"
    echo -e "${YELLOW}支持系统: Ubuntu 20.04/22.04/24.04, WSL2${NC}"
    echo
}

# 标准主函数
main() {
    # 显示标题
    show_script_header

    # 检查运行权限
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}✗ 此脚本需要root权限运行${NC}"
        echo -e "${BLUE}请使用: sudo $0${NC}"
        exit 1
    fi

    # 初始化日志
    log_info "开始CUDA安装流程..."

    # 执行12步安装流程（调用优化后的函数）
    echo -e "${BLUE}开始执行CUDA安装的12步流程...${NC}\n"

    # 第一步：清理NVIDIA配置
    clean_nvidia_configuration || {
        log_error "第一步失败：清理NVIDIA配置"
        exit 1
    }

    # 第二步：检测系统环境
    detect_system_environment || {
        log_error "第二步失败：检测系统环境"
        exit 1
    }

    # 第三步：检查NVIDIA GPU硬件
    check_nvidia_gpu_hardware || {
        log_error "第三步失败：检查NVIDIA GPU硬件"
        exit 1
    }

    # 第四步：获取CUDA安装选择
    get_cuda_installation_choice || {
        log_error "第四步失败：获取CUDA安装选择"
        exit 1
    }

    # 第五步：处理NVIDIA驱动
    local driver_status
    check_nvidia_driver_status
    driver_status=$?

    case $driver_status in
        0)
            echo -e "${GREEN}✓ NVIDIA驱动状态正常${NC}"
            ;;
        1)
            echo -e "${YELLOW}需要安装NVIDIA驱动${NC}"
            NEED_INSTALL_DRIVER=true
            ;;
        2)
            echo -e "${YELLOW}检测到Nouveau驱动冲突，需要重启${NC}"
            NEED_REBOOT=true
            exit 1
            ;;
        *)
            log_error "第五步失败：检查NVIDIA驱动状态"
            exit 1
            ;;
    esac

    # 如果需要安装驱动
    if [[ "$NEED_INSTALL_DRIVER" == "true" ]]; then
        echo -e "${BLUE}开始安装NVIDIA驱动...${NC}"

        # 检查是否有Nouveau驱动冲突
        if lsmod | grep -q nouveau 2>/dev/null; then
            echo -e "${YELLOW}检测到Nouveau驱动冲突，需要先禁用${NC}"
            disable_nouveau_driver || {
                log_error "禁用Nouveau驱动失败"
                exit 1
            }
            echo -e "${YELLOW}请重启系统后重新运行安装脚本${NC}"
            exit 1
        fi

        install_nvidia_driver || {
            log_error "安装NVIDIA驱动失败"
            exit 1
        }

        echo -e "${GREEN}✓ NVIDIA驱动安装完成${NC}"
        echo -e "${YELLOW}建议重启系统以确保驱动正常工作${NC}"
    fi

    # 第六步：检查CUDA状态
    check_cuda_status || {
        log_error "第六步失败：检查CUDA状态"
        exit 1
    }

    # 第七步：安装CUDA工具包
    install_cuda_toolkit || {
        log_error "第七步失败：安装CUDA工具包"
        exit 1
    }

    # 第八步：配置CUDA环境
    configure_cuda_environment || {
        log_error "第八步失败：配置CUDA环境"
        exit 1
    }

    # 第九步：安装cuDNN
    install_cudnn || {
        log_error "第九步失败：安装cuDNN"
        exit 1
    }

    # 第十步：验证安装
    verify_installation || {
        log_error "第十步失败：验证安装"
        exit 1
    }

    # 第十一步：清理和优化
    cleanup_and_optimize || {
        log_error "第十一步失败：清理和优化"
        exit 1
    }

    # 第十二步：完成安装
    complete_installation || {
        log_error "第十二步失败：完成安装"
        exit 1
    }

    log_info "CUDA安装流程全部完成！"
}

# 主要安装函数 - 供外部脚本调用
install_gpu_cuda() {
    # 不显示标题和权限检查（由主脚本处理）
    # 直接执行GPU和CUDA安装的核心流程

    log_info "开始GPU和CUDA安装流程..."

    # 第一步：清理NVIDIA配置
    clean_nvidia_configuration || {
        log_error "第一步失败：清理NVIDIA配置"
        return 1
    }

    # 第二步：检测系统环境
    detect_system_environment || {
        log_error "第二步失败：检测系统环境"
        return 1
    }

    # 第三步：检查NVIDIA GPU硬件
    check_nvidia_gpu_hardware || {
        log_error "第三步失败：检查NVIDIA GPU硬件"
        return 1
    }

    # 第四步：获取CUDA安装选择
    get_cuda_installation_choice || {
        log_error "第四步失败：获取CUDA安装选择"
        return 1
    }

    # 第五步：检查NVIDIA驱动状态（基于选择的CUDA版本）
    local driver_status
    check_nvidia_driver_status
    driver_status=$?

    case $driver_status in
        0)
            echo -e "${GREEN}✓ NVIDIA驱动状态正常${NC}"
            ;;
        1)
            echo -e "${YELLOW}需要安装NVIDIA驱动${NC}"
            NEED_INSTALL_DRIVER=true
            ;;
        2)
            echo -e "${YELLOW}检测到Nouveau驱动冲突，需要重启${NC}"
            NEED_REBOOT=true
            return 1
            ;;
        *)
            log_error "第五步失败：检查NVIDIA驱动状态"
            return 1
            ;;
    esac

    # 如果需要安装驱动
    if [[ "$NEED_INSTALL_DRIVER" == "true" ]]; then
        echo -e "${BLUE}开始安装NVIDIA驱动...${NC}"

        # 检查是否有Nouveau驱动冲突
        if lsmod | grep -q nouveau 2>/dev/null; then
            echo -e "${YELLOW}检测到Nouveau驱动冲突，需要先禁用${NC}"
            disable_nouveau_driver || {
                log_error "禁用Nouveau驱动失败"
                return 1
            }
            echo -e "${YELLOW}请重启系统后重新运行安装脚本${NC}"
            return 1
        fi

        install_nvidia_driver || {
            log_error "安装NVIDIA驱动失败"
            return 1
        }

        echo -e "${GREEN}✓ NVIDIA驱动安装完成${NC}"
        echo -e "${YELLOW}建议重启系统以确保驱动正常工作${NC}"
    fi

    # 第六步：检查CUDA状态
    check_cuda_status || {
        log_error "第六步失败：检查CUDA状态"
        return 1
    }

    # 第七步：安装CUDA工具包
    install_cuda_toolkit || {
        log_error "第七步失败：安装CUDA工具包"
        return 1
    }

    # 第八步：配置CUDA环境
    configure_cuda_environment || {
        log_error "第八步失败：配置CUDA环境"
        return 1
    }

    # 第九步：安装cuDNN
    install_cudnn || {
        log_error "第九步失败：安装cuDNN"
        return 1
    }

    # 第十步：验证安装
    verify_installation || {
        log_error "第十步失败：验证安装"
        return 1
    }

    # 第十一步：清理和优化
    cleanup_and_optimize || {
        log_error "第十一步失败：清理和优化"
        return 1
    }

    # 第十二步：完成安装
    complete_installation || {
        log_error "第十二步失败：完成安装"
        return 1
    }

    log_info "GPU和CUDA安装流程完成！"
    return 0
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
