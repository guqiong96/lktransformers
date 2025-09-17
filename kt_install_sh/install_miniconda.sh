#!/bin/bash

# KTransformers Miniconda安装模块
# 包含Miniconda安装和配置

# 获取最新Miniconda版本信息
get_latest_miniconda_version() {
    log_info "获取最新Miniconda版本信息..."
    
    local version_info=""
    local version_source=""
    
    # 尝试从USTC镜像获取版本信息（优先）
    log_info "尝试从USTC镜像获取版本信息..."
    version_info=$(curl -s "https://mirrors.ustc.edu.cn/anaconda/miniconda/" | grep -o 'Miniconda3-latest-Linux-x86_64\.sh' | head -1)
    
    if [[ -n "$version_info" ]]; then
        version_source="USTC镜像"
        log_info "从USTC镜像检测到版本: $version_info"
    else
        # 尝试清华镜像
        log_info "尝试从清华镜像获取版本信息..."
        version_info=$(curl -s "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/" | grep -o 'Miniconda3-latest-Linux-x86_64\.sh' | head -1)
        
        if [[ -n "$version_info" ]]; then
            version_source="清华镜像"
            log_info "从清华镜像检测到版本: $version_info"
        else
            # 回退到官方源
            log_info "尝试从官方源获取版本信息..."
            version_info=$(curl -s "https://repo.anaconda.com/miniconda/" | grep -o 'Miniconda3-latest-Linux-x86_64\.sh' | head -1)
            
            if [[ -n "$version_info" ]]; then
                version_source="官方源"
                log_info "从官方源检测到版本: $version_info"
            fi
        fi
    fi
    
    # 如果都失败，返回默认版本
    if [[ -z "$version_info" ]]; then
        log_info "无法获取最新版本信息，使用默认版本"
        echo "latest"
    else
        echo "$version_info"
    fi
}


# 安装Miniconda并设置环境变量
install_miniconda() {
    show_progress "安装和配置Miniconda"

    log_info "开始Miniconda环境设置..."

    # 标记是否需要重新安装
    local reinstall_needed=false

    # 检查Conda是否已安装并可用
    if command -v conda &> /dev/null; then
        log_info "检测到Conda已安装"

        # 输出Conda版本信息
        local conda_version=$(conda --version 2>/dev/null || echo "无法获取版本")
        if [[ "$conda_version" == "无法获取版本" ]]; then
            log_warn "Conda命令可用但无法获取版本信息，可能已损坏"
            reinstall_needed=true
        else
        log_info "当前Conda版本: $conda_version"

            # 检查Python版本是否满足要求
            if conda info --json 2>/dev/null | grep -q "\"python_version\""; then
                local conda_python_version=$(conda info --json 2>/dev/null | grep "\"python_version\"" | cut -d'"' -f4)
                local conda_python_major=$(echo $conda_python_version | cut -d'.' -f1)
                local conda_python_minor=$(echo $conda_python_version | cut -d'.' -f2)
                local required_python_major=$(echo $PYTHON_VERSION | cut -d'.' -f1)
                local required_python_minor=$(echo $PYTHON_VERSION | cut -d'.' -f2)

                log_info "Conda环境Python版本: $conda_python_version"

                if [[ $conda_python_major -lt $required_python_major || ($conda_python_major -eq $required_python_major && $conda_python_minor -lt $required_python_minor) ]]; then
                    log_warn "Conda环境Python版本($conda_python_version)低于要求的版本($PYTHON_VERSION)"
                    log_info "将更新Conda..."
                    if ! conda update -y conda python; then
                        log_warn "无法更新Conda，可能已损坏，将尝试重新安装"
                        reinstall_needed=true
                    fi
                else
                    log_info "Conda环境Python版本满足要求"
                fi
            else
                log_warn "无法获取Conda的Python版本信息，可能已损坏"
                reinstall_needed=true
            fi

            # 如果不需要重新安装，继续检查并设置环境变量
            if [[ "$reinstall_needed" == "false" ]]; then
                # 检查并设置conda环境变量
                ensure_conda_environment_vars

                # 验证conda功能
                if ! verify_conda_functionality; then
                    log_warn "Conda功能验证失败，将重新安装"
                    reinstall_needed=true
                fi
            fi
        fi
    else
        # Conda命令不可用，检查是否存在Miniconda目录
        if [[ -d "/opt/miniconda3" ]]; then
            log_warn "Miniconda目录存在: /opt/miniconda3，但conda命令不可用"

            # 尝试将Miniconda添加到PATH并使用
            export PATH="/opt/miniconda3/bin:$PATH"

            # 再次检查conda命令
        if command -v conda &> /dev/null; then
            log_info "成功找到conda命令，使用现有安装"

                # 设置conda环境变量
                ensure_conda_environment_vars

                # 验证conda功能
                if ! verify_conda_functionality; then
                    log_warn "虽然找到conda命令，但功能验证失败，将重新安装"
                    reinstall_needed=true
                fi
            else
                log_warn "即使在PATH中添加了Miniconda路径，conda命令仍不可用"
                reinstall_needed=true
            fi
        else
            # 既没有conda命令也没有Miniconda目录，需要安装
            log_info "未检测到Conda，将进行安装"
            reinstall_needed=true
        fi
    fi

    # 如果需要重新安装，先清理现有安装
    if [[ "$reinstall_needed" == "true" ]]; then
        log_info "开始清理旧的Miniconda安装..."

        # 卸载conda包（如果conda命令可用）
        if command -v conda &> /dev/null; then
            log_info "尝试使用conda命令卸载..."
            conda install --yes anaconda-clean 2>/dev/null || true
            anaconda-clean --yes 2>/dev/null || true
        fi

        # 删除Miniconda目录
        if [[ -d "/opt/miniconda3" ]]; then
            log_info "删除Miniconda目录: /opt/miniconda3"
            sudo rm -rf "/opt/miniconda3"
        fi

        # 清理环境变量文件
        log_info "清理环境变量文件..."
        if [[ -f "/etc/profile.d/miniconda.sh" ]]; then
            sudo rm -f "/etc/profile.d/miniconda.sh"
        fi
        if [[ -f "/etc/profile.d/conda_init.sh" ]]; then
            sudo rm -f "/etc/profile.d/conda_init.sh"
        fi

        # 清理bash.bashrc中的相关行
        if grep -q "miniconda3/bin" /etc/bash.bashrc; then
            log_info "从/etc/bash.bashrc中移除Miniconda相关配置"
            sudo sed -i '/# Miniconda环境变量/d' /etc/bash.bashrc
            sudo sed -i '/miniconda3\/bin/d' /etc/bash.bashrc
            sudo sed -i '/conda.sh/d' /etc/bash.bashrc
        fi

        # 清理.bashrc中的相关行
        if [[ -f "$HOME/.bashrc" ]] && grep -q "miniconda3" "$HOME/.bashrc"; then
            log_info "从~/.bashrc中移除Miniconda相关配置"
            sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$HOME/.bashrc"
        fi

        # 现在执行全新安装
        log_info "准备进行全新安装..."
    else
        # 如果不需要重新安装，直接返回成功
        log_info "Miniconda安装和配置正常，无需重新安装"
        return 0
    fi

    # 安装Miniconda
    log_info "准备安装Miniconda..."

    # 创建临时目录
    local tempdir=$(mktemp -d)
    cd "$tempdir" || {
        log_error "无法创建临时目录"
        return 1
    }

    # 获取最新版本信息
    local miniconda_version=$(get_latest_miniconda_version)
    local miniconda_installer=""
    
    if [[ "$miniconda_version" == "latest" ]]; then
        miniconda_installer="Miniconda3-latest-Linux-x86_64.sh"
        log_info "使用最新版本安装包: $miniconda_installer"
    else
        # 尝试使用特定版本，如果失败则回退到latest
        miniconda_installer="Miniconda3-${miniconda_version}-Linux-x86_64.sh"
        log_info "尝试使用特定版本安装包: $miniconda_installer"
    fi

    # 下载Miniconda安装脚本
    local download_success=false
    
    if [[ "${USE_MIRROR}" == "y" ]]; then
        log_info "使用国内镜像下载Miniconda..."
        
        # 尝试USTC镜像（优先）
        log_info "尝试USTC镜像..."
        if curl -L -O "https://mirrors.ustc.edu.cn/anaconda/miniconda/$miniconda_installer" 2>/dev/null; then
            download_success=true
        else
            log_warn "USTC镜像下载失败，尝试清华镜像..."
            
            # 尝试清华镜像
            if curl -L -O "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$miniconda_installer" 2>/dev/null; then
                download_success=true
            else
                log_warn "清华镜像下载失败，尝试官方源..."
                
                # 回退到官方源
                if curl -L -O "https://repo.anaconda.com/miniconda/$miniconda_installer" 2>/dev/null; then
                    download_success=true
                fi
            fi
        fi
    else
        log_info "从官方源下载Miniconda..."
        if curl -L -O "https://repo.anaconda.com/miniconda/$miniconda_installer" 2>/dev/null; then
            download_success=true
        fi
    fi
    
    # 如果特定版本下载失败且不是latest版本，尝试回退到latest
    if [[ "$download_success" == "false" && "$miniconda_version" != "latest" ]]; then
        log_warn "特定版本 $miniconda_installer 下载失败，回退到latest版本..."
        miniconda_installer="Miniconda3-latest-Linux-x86_64.sh"
        log_info "尝试下载latest版本: $miniconda_installer"
        
        if [[ "${USE_MIRROR}" == "y" ]]; then
            # 重新尝试国内镜像
            log_info "使用国内镜像下载latest版本..."
            
            if curl -L -O "https://mirrors.ustc.edu.cn/anaconda/miniconda/$miniconda_installer" 2>/dev/null; then
                download_success=true
            elif curl -L -O "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$miniconda_installer" 2>/dev/null; then
                download_success=true
            elif curl -L -O "https://repo.anaconda.com/miniconda/$miniconda_installer" 2>/dev/null; then
                download_success=true
            fi
        else
            # 官方源
            if curl -L -O "https://repo.anaconda.com/miniconda/$miniconda_installer" 2>/dev/null; then
                download_success=true
            fi
        fi
    fi
    
    if [[ "$download_success" == "false" ]]; then
        log_error "无法下载Miniconda安装程序"
        cd "$ORIGINAL_PWD"
        return 1
    fi

    # 校验下载的文件
    if [[ ! -f "$miniconda_installer" ]]; then
        log_error "未找到Miniconda安装程序"
        cd "$ORIGINAL_PWD"
        return 1
    fi

    # 赋予执行权限
    chmod +x "$miniconda_installer"

    # 执行安装
    log_info "安装Miniconda到 /opt/miniconda3..."
    bash "$miniconda_installer" -b -p "/opt/miniconda3" || {
        log_error "Miniconda安装失败"
        cd "$ORIGINAL_PWD"
        return 1
    }

    # 设置目录权限 - 使用用户友好的权限设置
    log_info "设置Miniconda目录权限..."
    
    # 获取当前用户和组信息
    local current_user=$(whoami)
    local current_group=$(id -gn)
    
    # 如果当前用户不是root，则使用当前用户权限
    if [[ "$current_user" != "root" ]]; then
        log_info "使用当前用户权限设置: $current_user:$current_group"
        sudo chown -R "$current_user:$current_group" "/opt/miniconda3"
    else
        log_info "使用root权限设置"
        sudo chown -R root:root "/opt/miniconda3"
    fi
    
    # 设置合理的权限
    sudo chmod -R 755 "/opt/miniconda3"

    # 为环境和包目录设置适当的权限
    log_info "设置conda环境和包目录权限..."
    sudo mkdir -p "/opt/miniconda3/envs" "/opt/miniconda3/pkgs"

    # 创建conda用户组（如果不存在）并添加当前用户
    if ! getent group conda >/dev/null; then
        sudo groupadd conda
        log_info "创建conda用户组"
    fi
    
    # 将当前用户添加到conda组
    if [[ "$current_user" != "root" ]]; then
        sudo usermod -a -G conda "$current_user"
        log_info "已将用户 $current_user 添加到conda组"
    fi

    # 设置环境和包目录的所有权和权限
    if [[ "$current_user" != "root" ]]; then
        sudo chown -R "$current_user:conda" "/opt/miniconda3/envs" "/opt/miniconda3/pkgs"
    else
        sudo chown -R root:conda "/opt/miniconda3/envs" "/opt/miniconda3/pkgs"
    fi
    sudo chmod -R 775 "/opt/miniconda3/envs" "/opt/miniconda3/pkgs"

    # 设置setgid位以确保新文件继承组权限
    sudo chmod g+s "/opt/miniconda3/envs" "/opt/miniconda3/pkgs"

    log_info "Miniconda目录权限设置完成，使用用户友好的权限模式"

    # 设置环境变量
    ensure_conda_environment_vars

    # 更新Conda - 处理ToS问题
    log_info "更新Conda到最新版本..."
    # 首先尝试接受ToS，然后更新
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true) && \
    conda update -n base -c defaults conda -y || {
        log_warn "Conda更新失败，但将继续使用现有版本"
    }

    # 配置Conda通道
    configure_conda_channels

    # 验证conda功能
    if ! verify_conda_functionality; then
        log_error "即使重新安装后，Conda功能验证仍然失败"
        log_error "可能存在系统兼容性问题，请手动检查"
        cd "$ORIGINAL_PWD"
        return 1
    fi

    # 清理
    cd "$ORIGINAL_PWD"
    rm -rf "$tempdir"

    log_info "Miniconda安装和配置完成"

    # 立即使环境变量生效
    log_info "执行source /etc/profile使环境变量立即生效..."
    source /etc/profile

    # 验证环境变量是否生效
    if command -v conda &> /dev/null; then
        local conda_version=$(conda --version 2>/dev/null)
        log_info "✓ Miniconda环境变量已生效，conda版本: $conda_version"
    else
        log_warn "Miniconda环境变量可能未生效，请手动执行: source /etc/profile"
    fi

    # 添加环境变量生效提示
    echo -e "\n${GREEN}Miniconda环境变量已配置在/etc/profile中并已生效${NC}"
    echo -e "验证Miniconda安装:"
    echo -e "  执行 ${BLUE}conda --version${NC} 查看conda版本"
    echo -e "  执行 ${BLUE}python --version${NC} 查看python版本\n"

    return 0
}

# 确保conda环境变量已正确设置
ensure_conda_environment_vars() {
    log_info "确保conda环境变量已正确设置..."

    # 检查Miniconda安装路径
    local miniconda_path="/opt/miniconda3"
    if [[ ! -d "$miniconda_path" ]]; then
        log_warn "Miniconda目录不存在: $miniconda_path"
        return 1
    fi

    # 检查/etc/profile文件是否存在
    if [[ ! -f "/etc/profile" ]]; then
        log_error "未找到/etc/profile文件，无法设置系统环境变量"
        return 1
    fi

    # 备份/etc/profile
    if ! sudo cp /etc/profile /etc/profile.bak.miniconda; then
        log_warn "无法备份/etc/profile文件，继续操作但不备份"
    else
        log_info "已备份/etc/profile到/etc/profile.bak.miniconda"
    fi

    # 检查是否已经配置了Miniconda环境变量
    if grep -q "# Miniconda Environment" /etc/profile || grep -q "miniconda3/bin" /etc/profile; then
        log_info "Miniconda环境变量已存在于/etc/profile中，将更新配置"
        # 删除已有的Miniconda环境变量配置
        sudo sed -i '/# Miniconda Environment/d' /etc/profile
        sudo sed -i '/miniconda3\/bin/d' /etc/profile
    fi

    # 添加Miniconda环境变量到/etc/profile
    log_info "添加Miniconda环境变量到/etc/profile"
    sudo bash -c "cat >> /etc/profile << 'EOF'

# Miniconda Environment
export PATH=\"/opt/miniconda3/bin:\$PATH\"
export CONDA_PREFIX=\"/opt/miniconda3\"
export CONDA_DEFAULT_ENV=\"base\"
EOF"

    # 创建系统级的conda初始化脚本
    log_info "创建系统级的conda初始化脚本..."
    sudo bash -c "cat > /etc/profile.d/conda.sh << 'EOF'
#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup=\"\$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)\"
if [ \$? -eq 0 ]; then
    eval \"\$__conda_setup\"
else
    if [ -f \"/opt/miniconda3/etc/profile.d/conda.sh\" ]; then
        . \"/opt/miniconda3/etc/profile.d/conda.sh\"
    else
        export PATH=\"/opt/miniconda3/bin:\$PATH\"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
EOF"

    # 设置conda.sh的权限
    sudo chmod 644 /etc/profile.d/conda.sh

    # 为所有用户创建conda配置目录
    log_info "为所有用户创建conda配置目录..."
    sudo mkdir -p /etc/conda
    sudo chmod 755 /etc/conda

    # 创建全局conda配置文件
    log_info "创建全局conda配置文件..."
    sudo bash -c "cat > /etc/conda/condarc << 'EOF'
envs_dirs:
  - /opt/miniconda3/envs
  - ~/.conda/envs
pkgs_dirs:
  - /opt/miniconda3/pkgs
  - ~/.conda/pkgs
EOF"

    # 设置condarc的权限
    sudo chmod 644 /etc/conda/condarc

    # 立即使环境变量生效
    log_info "使Miniconda环境变量立即生效..."
    source /etc/profile

    # 检查conda命令是否可用
    if command -v conda &> /dev/null; then
        log_info "conda命令可用: $(command -v conda)"

        # 初始化conda
        if ! "$miniconda_path/bin/conda" init bash; then
            log_warn "conda初始化失败，可能需要手动初始化"
        fi

        # 确保conda初始化脚本对所有用户可用
        if [[ -f "$HOME/.bashrc" ]]; then
            if ! grep -q "conda initialize" "$HOME/.bashrc"; then
                log_info "为当前用户添加conda初始化配置..."
                if ! "$miniconda_path/bin/conda" init bash; then
                    log_warn "为当前用户添加conda初始化配置失败，可能需要手动配置"
                fi
            fi
        fi
    else
        log_error "conda命令不可用，环境变量可能未正确设置"
        return 1
    fi

    log_info "Miniconda环境变量设置完成"
    return 0
}

# 验证conda功能
verify_conda_functionality() {
    log_info "验证conda功能..."

    # 验证conda命令
    if ! command -v conda &> /dev/null; then
        log_error "conda命令不可用，验证失败"
        return 1
    fi

    # 验证conda版本
    local conda_version
    conda_version=$(conda --version 2>/dev/null) || {
        log_error "无法获取conda版本，验证失败"
        return 1
    }
    log_info "conda版本: $conda_version"

    # 验证conda info命令
    if ! conda info &> /dev/null; then
        log_error "conda info命令失败，验证失败"
        return 1
    fi

    # 验证conda环境列表
    if ! conda info --envs &> /dev/null; then
        log_error "无法获取conda环境列表，验证失败"
        return 1
    fi

    # 验证conda包列表
    if ! conda list &> /dev/null; then
        log_warn "无法获取conda包列表，验证部分失败"
    fi

    # 验证Python
    if command -v python &> /dev/null; then
        local python_version
        python_version=$(python --version 2>&1) || {
            log_warn "python命令可用但无法获取版本，验证部分失败"
        }
        if [[ -n "$python_version" ]]; then
            log_info "Python版本: $python_version"
        fi
    else
        log_warn "python命令不可用，验证部分失败"
    fi

    # 验证用户目录权限
    log_info "验证用户目录权限..."

    return 0
}

# 配置Conda通道
configure_conda_channels() {
    log_info "配置Conda通道..."

    # 检查conda命令是否可用
    if ! command -v conda &> /dev/null; then
        log_warn "conda命令不可用，无法配置通道"
        return 1
    fi

    # 首先清除现有的通道配置，避免冲突
    log_info "清除现有的conda通道配置..."
    conda config --remove-key channels 2>/dev/null || true

    # 使用user_interaction中设置的镜像源配置 (USTC中科大源优先)
    if [[ "$(echo "$USE_CONDA_MIRROR" | tr '[:upper:]' '[:lower:]')" == "y" && -n "$CONDA_MIRROR_URL" ]]; then
        log_info "配置国内conda镜像: $CONDA_MIRROR_URL (USTC中科大源)"

        # 配置USTC中科大镜像源 - 2025年最新地址，完整支持
        # 参考: https://mirrors.ustc.edu.cn/help/anaconda.html
        conda config --add channels ${CONDA_MIRROR_URL}/pkgs/main/
        conda config --add channels ${CONDA_MIRROR_URL}/pkgs/r/
        conda config --add channels ${CONDA_MIRROR_URL}/pkgs/msys2/
        conda config --add channels ${CONDA_MIRROR_URL}/cloud/conda-forge/
        conda config --add channels ${CONDA_MIRROR_URL}/cloud/pytorch/
        conda config --add channels ${CONDA_MIRROR_URL}/cloud/bioconda/
        conda config --add channels ${CONDA_MIRROR_URL}/cloud/menpo/

        # 配置备用镜像源（清华大学TUNA）
        if [[ -n "$CONDA_MIRROR_URL_BACKUP" ]]; then
            log_info "配置备用conda镜像: $CONDA_MIRROR_URL_BACKUP (清华大学TUNA)"
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/pkgs/main/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/pkgs/r/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/pkgs/msys2/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/cloud/conda-forge/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/cloud/pytorch/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/cloud/bioconda/
            conda config --add channels ${CONDA_MIRROR_URL_BACKUP}/cloud/menpo/
        fi

        # 设置搜索时显示通道地址
        conda config --set show_channel_urls yes

        # 设置SSL验证（确保HTTPS连接安全）
        conda config --set ssl_verify yes

        # 设置通道优先级（flexible模式允许回退到备用源）
        conda config --set channel_priority flexible
        
        # 设置非交互模式，避免ToS问题
        conda config --set always_yes yes

        log_info "Conda镜像源配置完成 (USTC中科大 + 清华大学TUNA备用)"
    else
        # 使用官方源
        log_info "使用官方conda源"
        conda config --add channels conda-forge
        conda config --add channels anaconda
        conda config --add channels nvidia
    fi

    # 显示最终配置的通道
    log_info "当前配置的conda通道:"
    conda config --show channels

    # 验证镜像源配置
    verify_conda_mirror_config

    return 0
}

# 验证conda镜像源配置
verify_conda_mirror_config() {
    log_info "验证conda镜像源配置..."
    
    # 检查.condarc文件是否存在
    local condarc_path="$HOME/.condarc"
    if [[ -f "$condarc_path" ]]; then
        log_info "找到用户级.condarc配置文件: $condarc_path"
        echo "=== 用户级.condarc内容 ==="
        cat "$condarc_path"
    fi
    
    # 检查全局.condarc文件
    local global_condarc="/etc/conda/condarc"
    if [[ -f "$global_condarc" ]]; then
        log_info "找到全局级.condarc配置文件: $global_condarc"
        echo "=== 全局级.condarc内容 ==="
        cat "$global_condarc"
    fi
    
    # 测试镜像源连接 - 使用非交互模式避免ToS问题
    log_info "测试conda镜像源连接..."
    if CONDA_ALWAYS_YES=true conda search python --json >/dev/null 2>&1; then
        log_info "✓ conda镜像源连接测试成功"
        
        # 显示实际使用的通道
        log_info "当前conda搜索使用的通道:"
        CONDA_ALWAYS_YES=true conda search python --info 2>/dev/null | grep -E "(channel|url)" | head -5
    else
        log_warn "conda镜像源连接测试失败，将使用备用源或官方源"
        
        # 尝试重置为官方源
        log_info "尝试重置为官方conda源..."
        conda config --remove-key channels 2>/dev/null || true
        conda config --add channels defaults
        conda config --add channels conda-forge
        conda config --add channels anaconda
        conda config --set channel_priority flexible
    fi
    
    return 0
}
