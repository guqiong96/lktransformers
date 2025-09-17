#!/bin/bash


# 包含系统依赖和基础工具安装

# 迁移过时的trusted.gpg密钥环到现代格式
migrate_trusted_gpg_keys() {
    log_info "检查并迁移过时的trusted.gpg密钥环..."
    
    # 检查是否存在过时的trusted.gpg文件
    if [[ -f "/etc/apt/trusted.gpg" ]]; then
        log_info "发现过时的trusted.gpg文件，正在迁移密钥..."
        
        # 获取trusted.gpg中的所有密钥ID
        local key_ids=()
        if command -v apt-key &> /dev/null; then
            # 提取密钥ID（取最后8位）
            mapfile -t key_ids < <(apt-key list 2>/dev/null | grep "^pub" -A 1 | grep -E "^[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}[[:space:]]+[A-F0-9]{4}" | sed 's/.*\([A-F0-9]\{8\}\)$/\1/' | sort -u)
            
            # 迁移每个密钥到trusted.gpg.d
            for key_id in "${key_ids[@]}"; do
                if [[ -n "$key_id" && "$key_id" =~ ^[A-F0-9]{8}$ ]]; then
                    log_info "迁移密钥: $key_id"
                    sudo apt-key export "$key_id" 2>/dev/null | sudo gpg --dearmor > "/etc/apt/trusted.gpg.d/migrated-$key_id.gpg" 2>/dev/null || true
                fi
            done
            
            # 备份并删除过时的trusted.gpg文件
            if [[ ${#key_ids[@]} -gt 0 ]]; then
                sudo cp "/etc/apt/trusted.gpg" "/etc/apt/trusted.gpg.backup-$(date +%Y%m%d%H%M%S)"
                sudo rm -f "/etc/apt/trusted.gpg"
                log_info "已迁移 ${#key_ids[@]} 个密钥并删除过时的trusted.gpg文件"
            fi
        fi
    else
        log_info "未发现过时的trusted.gpg文件，无需迁移"
    fi
}

# 备份和清理APT源配置
backup_and_clean_apt_sources() {
    # 检查是否已经执行过备份和清理
    if [[ "${APT_SOURCES_CLEANED:-false}" == true ]]; then
        log_info "APT源已清理，跳过重复操作"
        return 0
    fi
    
    log_info "备份和清理APT源配置..."
    
    # 创建备份目录
    local backup_dir="/etc/apt/backup_$(date +%Y%m%d%H%M%S)"
    sudo mkdir -p "$backup_dir"
    
    # 备份主要配置文件
    if [[ -f "/etc/apt/sources.list" ]]; then
        sudo cp "/etc/apt/sources.list" "$backup_dir/"
        log_info "已备份 sources.list 到 $backup_dir"
    fi
    
    # 备份 sources.list.d 目录
    if [[ -d "/etc/apt/sources.list.d" ]]; then
        sudo cp -r "/etc/apt/sources.list.d" "$backup_dir/"
        log_info "已备份 sources.list.d 到 $backup_dir"
    fi
    
    # 备份 trusted.gpg.d 目录
    if [[ -d "/etc/apt/trusted.gpg.d" ]]; then
        sudo cp -r "/etc/apt/trusted.gpg.d" "$backup_dir/"
        log_info "已备份 trusted.gpg.d 到 $backup_dir"
    fi
    
    # 清理可能有问题的第三方源
    log_info "清理可能有问题的第三方源..."
    
    # 移除 Kitware 相关源（常见的GPG问题源）
    if [[ -f "/etc/apt/sources.list.d/kitware.list" ]]; then
        sudo mv "/etc/apt/sources.list.d/kitware.list" "$backup_dir/kitware.list.disabled"
        log_info "已禁用 Kitware 源"
    fi
    
    # 移除 ubuntu-toolchain-r 相关源（导致trusted.gpg警告）
    if [[ -f "/etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list" ]]; then
        sudo mv "/etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test.list" "$backup_dir/ubuntu-toolchain-r-ubuntu-test.list.disabled"
        log_info "已禁用 ubuntu-toolchain-r 源"
    fi
    
    # 清理损坏的GPG密钥
    if [[ -f "/usr/share/keyrings/kitware-archive-keyring.gpg" ]]; then
        sudo mv "/usr/share/keyrings/kitware-archive-keyring.gpg" "$backup_dir/"
        log_info "已移除损坏的 Kitware GPG 密钥"
    fi
    
    # 清理可能的临时GPG文件
    rm -f /tmp/kitware-archive-keyring.gpg 2>/dev/null || true
    
    # 处理过时的trusted.gpg密钥环迁移
    migrate_trusted_gpg_keys
    
    # 清理NVIDIA相关的旧配置和GPG错误
    clean_nvidia_gpg_errors
    
    # 设置状态标记，避免重复执行
    export APT_SOURCES_CLEANED=true
    
    log_info "APT源备份和清理完成，备份位置: $backup_dir"
}

# 清理NVIDIA相关的旧配置和GPG错误
clean_nvidia_gpg_errors() {
    log_message "INFO" "清理NVIDIA相关的旧配置和GPG错误..."
    
    # 清理可能存在的旧NVIDIA仓库配置
    local nvidia_sources=(
        "/etc/apt/sources.list.d/cuda.list"
        "/etc/apt/sources.list.d/nvidia-ml.list"
        "/etc/apt/sources.list.d/cuda-ubuntu*.list"
        "/etc/apt/sources.list.d/nvidia-docker.list"
    )
    
    for source_file in "${nvidia_sources[@]}"; do
        if ls $source_file 2>/dev/null; then
            sudo rm -f $source_file
            log_message "INFO" "已移除旧的NVIDIA源配置: $source_file"
        fi
    done
    
    # 清理可能存在的旧GPG密钥
    local nvidia_keys=(
        "/usr/share/keyrings/cuda-archive-keyring.gpg"
        "/usr/share/keyrings/nvidia-ml-keyring.gpg"
        "/usr/share/keyrings/nvidia-docker-keyring.gpg"
        "/etc/apt/trusted.gpg.d/cuda.gpg"
        "/etc/apt/trusted.gpg.d/nvidia-ml.gpg"
    )
    
    for key_file in "${nvidia_keys[@]}"; do
        if [[ -f "$key_file" ]]; then
            sudo rm -f "$key_file"
            log_message "INFO" "已移除旧的NVIDIA GPG密钥: $key_file"
        fi
    done
    
    # 清理APT优先级配置
    local nvidia_prefs=(
        "/etc/apt/preferences.d/cuda-repository-pin-600"
        "/etc/apt/preferences.d/nvidia-ml-repository-pin-600"
    )
    
    for pref_file in "${nvidia_prefs[@]}"; do
        if [[ -f "$pref_file" ]]; then
            sudo rm -f "$pref_file"
            log_message "INFO" "已移除NVIDIA APT优先级配置: $pref_file"
        fi
    done
    
    # 清理可能存在的apt-key添加的NVIDIA密钥（已弃用但可能存在）
    if command -v apt-key &> /dev/null; then
        # 尝试移除可能的NVIDIA密钥（静默处理，因为可能不存在）
        sudo apt-key del 3bf863cc 2>/dev/null || true
        sudo apt-key del 7fa2af80 2>/dev/null || true
    fi
    
    log_message "INFO" "✓ NVIDIA相关旧配置和GPG错误清理完成"
}

# 清理NVIDIA仓库配置（在CUDA安装完成后调用）
cleanup_nvidia_repository() {
    log_info "清理NVIDIA仓库配置以确保环境纯洁性..."
    
    # 移除NVIDIA仓库源文件
    if [[ -f "/etc/apt/sources.list.d/cuda.list" ]]; then
        sudo rm -f "/etc/apt/sources.list.d/cuda.list"
        log_info "已移除NVIDIA CUDA仓库源文件"
    fi
    
    # 移除APT优先级配置
    if [[ -f "/etc/apt/preferences.d/cuda-repository-pin-600" ]]; then
        sudo rm -f "/etc/apt/preferences.d/cuda-repository-pin-600"
        log_info "已移除CUDA仓库优先级配置"
    fi
    
    # 保留GPG密钥文件，因为已安装的包可能需要验证
    # 但可以选择性移除（如果确定不再需要）
    # rm -f "/usr/share/keyrings/cuda-archive-keyring.gpg"
    
    # 更新软件包列表以反映更改
    log_info "更新软件包列表..."
    sudo apt-get update &> /dev/null || log_warn "更新软件包列表时出现警告，但这是正常的"
    
    log_info "NVIDIA仓库清理完成"
}

# 询问是否使用apt国内源
ask_apt_mirror() {
    print_option_header "APT软件源配置"
    print_option "y" "使用国内APT镜像源" "推荐国内用户选择"
    print_option "n" "使用官方源" "推荐海外用户选择"
    echo -e " 是否使用国内APT镜像源? (y/n) [默认: ${GREEN}n${NC}]: "
    read -r apt_mirror_choice
    
    # 无论选择什么，都先备份和清理源
    backup_and_clean_apt_sources
    
    if [[ "$(echo "$apt_mirror_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        USE_APT_MIRROR="y"
        configure_apt_mirror
    else
        USE_APT_MIRROR="n"
        log_info "继续使用官方APT源"
    fi
    
    # 导出变量以便其他模块使用
    export USE_APT_MIRROR
}

# 配置APT国内源
configure_apt_mirror() {
    # 检查是否已经配置过APT镜像源
    if [[ "${APT_MIRROR_CONFIGURED:-false}" == true ]]; then
        log_info "APT镜像源已配置，跳过重复配置"
        return 0
    fi
    
    log_info "配置APT国内镜像源..."
    
    # 检测Ubuntu版本
    local ubuntu_version=""
    if command -v lsb_release &> /dev/null; then
        ubuntu_version=$(lsb_release -rs)
        ubuntu_codename=$(lsb_release -cs)
    else
        # 从/etc/os-release获取版本代号
        ubuntu_codename=$(grep VERSION_CODENAME /etc/os-release 2>/dev/null | cut -d'=' -f2 || echo "focal")
        ubuntu_version=$(grep VERSION_ID /etc/os-release 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "20.04")
        # 检查是否为Debian系统
        if grep -q "Debian" /etc/os-release 2>/dev/null; then
            log_error "检测到Debian系统，但未安装lsb_release。请先安装lsb_release包"
            log_info "安装命令: apt install -y lsb-release"
            return 1
        fi
    fi
    
    log_info "检测到Ubuntu版本: $ubuntu_version ($ubuntu_codename)"
    
    # 选择镜像源
    print_option_header "选择APT镜像源"
    print_option "1" "清华大学镜像源" "推荐，2025年最新可用"
    print_option "2" "阿里云镜像源" "稳定可靠"
    print_option "3" "中科大镜像源" "高速稳定"
    print_option "4" "华为云镜像源" "企业级稳定"
    echo -e " 请选择 [默认: ${GREEN}1${NC}]: "
    read -r mirror_choice
    
    local mirror_url=""
    local mirror_name=""
    case $mirror_choice in
        2) 
            mirror_url="https://mirrors.aliyun.com/ubuntu/"
            mirror_name="阿里云"
            ;;
        3) 
            mirror_url="https://mirrors.ustc.edu.cn/ubuntu/"
            mirror_name="中科大"
            ;;
        4) 
            mirror_url="https://repo.huaweicloud.com/ubuntu/"
            mirror_name="华为云"
            ;;
        *) 
            mirror_url="https://mirrors.tuna.tsinghua.edu.cn/ubuntu/"
            mirror_name="清华大学"
            ;;
    esac
    
    # 写入新的sources.list
    local sources_file="/etc/apt/sources.list"
    sudo tee "$sources_file" > /dev/null << EOF
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb ${mirror_url} ${ubuntu_codename} main restricted universe multiverse
# deb-src ${mirror_url} ${ubuntu_codename} main restricted universe multiverse
deb ${mirror_url} ${ubuntu_codename}-updates main restricted universe multiverse
# deb-src ${mirror_url} ${ubuntu_codename}-updates main restricted universe multiverse
deb ${mirror_url} ${ubuntu_codename}-backports main restricted universe multiverse
# deb-src ${mirror_url} ${ubuntu_codename}-backports main restricted universe multiverse
deb ${mirror_url} ${ubuntu_codename}-security main restricted universe multiverse
# deb-src ${mirror_url} ${ubuntu_codename}-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb ${mirror_url} ${ubuntu_codename}-proposed main restricted universe multiverse
# deb-src ${mirror_url} ${ubuntu_codename}-proposed main restricted universe multiverse
EOF
    
    log_info "已配置APT国内镜像源: $mirror_name ($mirror_url)"

     # 清理APT缓存
    log_info "清理APT缓存..."
    sudo apt clean || log_warn "清理APT缓存失败，请手动执行 apt clean"
    
    # 更新软件包列表
    log_info "更新软件包列表..."
    sudo apt update || log_warn "更新软件包列表失败，请手动执行 apt update"
    
    # 设置状态标记，避免重复配置
    export APT_MIRROR_CONFIGURED=true
}

# 安装基础依赖工具
install_basic_tools() {
    show_progress "安装基础依赖工具"
    
    # 首先询问用户是否使用国内apt源
    ask_apt_mirror
    
    log_info "检查并安装基础系统工具..."
    log_info "此步骤确保系统具有必要的基本工具，适用于最小化安装的系统环境"
    
    log_info "以root权限执行系统命令"
    
    # 更新软件包列表
    echo -e "${GREEN}更新软件包列表...${NC}"
    if ! sudo apt-get update; then
        log_error "无法更新软件包列表"
        return 1
    fi
    
    # 基础工具列表（仅包含系统级工具，不包含编译工具）
    local basic_tools=(
        "curl"          # 网络下载工具
        "wget"          # 网络下载工具
        "git"           # 版本控制工具
        "software-properties-common" # 软件源管理
        "apt-transport-https" # HTTPS传输支持
        "ca-certificates" # 证书管理
        "gnupg"         # GPG密钥管理
        "lsb-release"   # 系统信息
        "util-linux"    # 系统工具(包含lscpu)
        "pciutils"      # PCI设备工具(包含lspci)
        "net-tools"     # 网络工具
        "iputils-ping"  # 网络测试
        "unzip"         # 解压工具
        "zip"           # 压缩工具
        "tar"           # 归档工具
        "gzip"          # 压缩工具
        "numactl"       # NUMA控制工具
        "bc"            # 计算器工具（用于版本比较）
    )
    
    # 检查并安装基础工具
    local missing_tools=()
    for tool in "${basic_tools[@]}"; do
        if ! dpkg -l | grep -q "^ii  $tool "; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_info "需要安装的基础工具: ${missing_tools[*]}"
        
        # 安装缺失的工具
        if sudo apt-get install -y "${missing_tools[@]}"; then
            log_info "基础工具安装完成"
        else
            log_error "基础工具安装失败"
            return 1
        fi
    else
        log_info "所有基础工具已安装"
    fi
    
    # 验证关键工具是否可用
    local critical_commands=("curl" "wget" "git" "lscpu" "lspci")
    for cmd in "${critical_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "关键工具 $cmd 不可用"
            return 1
        fi
    done
    
    log_info "基础依赖工具安装完成"
    return 0
}

# 安装系统依赖
install_system_deps() {
    show_progress "安装系统依赖"
    
    # 确认conda环境已激活
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_warn "未检测到激活的conda环境，系统依赖安装可能不完整"
    else
        log_info "在conda环境中安装系统依赖: $CONDA_PREFIX"
    fi
    
    log_info "更新系统包列表..."
    echo -e "${GREEN}更新系统包列表...${NC}"
    sudo apt-get update
    
    # 安装Node.js和npm（用于Web服务前端构建，这是KTransformers特有需求）
    log_info "系统依赖安装完成"
    return 0
}