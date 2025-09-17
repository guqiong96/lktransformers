#!/bin/bash

# KTransformers Conda环境管理模块
# 包含Conda环境创建和激活

# 配置pip镜像源
configure_pip_mirror() {
    log_info "配置pip镜像源..."

    # 确保在conda环境中
    if [[ -z "$CONDA_PREFIX" || -z "$CONDA_DEFAULT_ENV" ]]; then
        log_warn "未检测到conda环境，跳过pip镜像源配置"
        return 0
    fi

    # 使用user_interaction中设置的镜像源配置
    if [[ "$(echo "$USE_PIP_MIRROR" | tr '[:upper:]' '[:lower:]')" == "y" && -n "$PIP_MIRROR_URL" && -n "$PIP_MIRROR_HOST" ]]; then
        # 配置pip源 - 使用2025年最新的清华大学镜像
        log_info "配置pip源: $PIP_MIRROR_URL"

        # 设置阿里云pip源（仅在conda环境激活后设置一次）
        log_info "配置pip使用阿里云镜像源..."
        pip config set global.index-url "https://mirrors.aliyun.com/pypi/simple" || log_warn "设置全局index-url失败"
        pip config set install.trusted-host "mirrors.aliyun.com" || log_warn "设置trusted-host失败"

        # 测试pip源连接性
        log_info "测试pip源连接性..."
        if pip install -i https://mirrors.aliyun.com/pypi/simple/ --dry-run --no-deps requests &>/dev/null; then
            log_info "pip镜像源连接正常"
        else
            log_warn "pip镜像源连接测试失败，但配置已完成"
        fi
    else
        # 使用官方源
        log_info "配置pip官方源"
        pip config set global.index-url "https://pypi.org/simple" || log_warn "设置官方源失败"
    fi

    # 验证pip源配置
    local current_index_url=$(pip config get global.index-url 2>/dev/null)
    if [[ -n "$current_index_url" ]]; then
        log_info "当前pip全局源: $current_index_url"
    else
        log_warn "无法获取当前pip源配置，可能配置未生效"
    fi

    # 显示所有pip配置
    log_info "当前pip配置:"
    pip config list 2>/dev/null || log_warn "无法列出pip配置"

    return 0
}

# 创建和管理Conda环境
conda_env_manager() {
    show_progress "创建和激活Conda环境"

    # 确保conda命令可用
    if ! command -v conda &> /dev/null; then
        log_error "conda命令不可用，请先安装Miniconda或Anaconda"
        return 1
    fi

    # 预先处理 Conda Terms of Service
    log_info "预处理 Conda Terms of Service..."
    conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/main" 2>/dev/null || {
        log_warn "无法接受主通道服务条款，将在创建环境时处理"
    }
    conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/r" 2>/dev/null || {
        log_warn "无法接受R通道服务条款，将在创建环境时处理"
    }

    # 加载conda.sh以确保conda activate可用
    local conda_root=$(dirname $(dirname $(which conda)))
    local conda_sh="$conda_root/etc/profile.d/conda.sh"

    if [[ -f "$conda_sh" ]]; then
        log_info "加载conda.sh: $conda_sh"
        source "$conda_sh"
    else
        log_warn "找不到conda.sh，尝试使用conda shell hook"
        # 使用更安全的方法获取conda shell hook
        local conda_hook_output
        if conda_hook_output=$(conda shell.bash hook 2>/dev/null); then
            source /dev/stdin <<< "$conda_hook_output"
        else
            log_error "无法获取conda shell hook，conda环境可能无法正常工作"
            return 1
        fi
    fi

    # 检查环境是否已存在
    if conda info --envs | grep -q "^${CONDA_ENV}"; then
        log_info "Conda环境 '$CONDA_ENV' 已存在"

        # 尝试激活环境以验证其可用性
        log_info "验证环境可用性..."
        if conda activate "$CONDA_ENV" 2>/dev/null && python -c "import sys; print(sys.version)" &>/dev/null; then
            log_info "环境 '$CONDA_ENV' 可正常使用，继续使用该环境"

            # 确保Python版本匹配
            local current_python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$current_python_version" != "$PYTHON_VERSION" ]]; then
                log_warn "当前环境Python版本($current_python_version)与目标版本($PYTHON_VERSION)不匹配"
                log_info "删除现有环境并重新创建"
                conda deactivate
                conda env remove -n "$CONDA_ENV" -y
                create_conda_env
            else
                log_info "环境Python版本正确: $current_python_version"
            fi
        else
            # 如果环境不能正常使用，重新创建
            log_warn "环境 '$CONDA_ENV' 验证失败，将重新创建"
            conda deactivate 2>/dev/null || true
            conda env remove -n "$CONDA_ENV" -y
            create_conda_env
        fi
    else
        # 创建新环境
        log_info "环境 '$CONDA_ENV' 不存在，创建新环境"
        create_conda_env
    fi

    # 激活环境
    log_info "激活Conda环境: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        log_error "无法激活Conda环境: $CONDA_ENV"
        return 1
    }

    # 验证环境
    local active_env=$(conda info --envs | grep '*' | awk '{print $1}')
    if [[ "$active_env" == "$CONDA_ENV" ]]; then
        log_info "成功激活Conda环境: $CONDA_ENV"

        # 获取Python版本
        local python_version=$(python --version 2>&1)
        log_info "当前Python版本: $python_version"

        # 更新基础包
        log_info "更新基础Python包..."
        pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools wheel || {
            log_warn "更新基础包失败，但将继续安装"
        }

        # 配置pip镜像源
        configure_pip_mirror || {
            log_warn "配置pip镜像源失败，但将继续安装"
        }

        # 设置环境变量
        export CONDA_ACTIVE_ENV="$CONDA_ENV"
        return 0
    else
        log_error "环境激活验证失败，当前环境: $active_env"
        return 1
    fi
}

# 创建conda环境
create_conda_env() {
    log_info "创建新的conda环境: $CONDA_ENV 与 Python $PYTHON_VERSION..."

    # 处理 Conda Terms of Service
    log_info "检查并接受 Conda Terms of Service..."

    # 尝试接受主要通道的服务条款
    local channels=("https://repo.anaconda.com/pkgs/main" "https://repo.anaconda.com/pkgs/r")
    for channel in "${channels[@]}"; do
        log_info "接受通道服务条款: $channel"
        conda tos accept --override-channels --channel "$channel" 2>/dev/null || {
            log_warn "无法接受通道 $channel 的服务条款，将尝试移除该通道"
            conda config --remove channels "$channel" 2>/dev/null || true
        }
    done

    # 如果仍然有问题，尝试使用 conda-forge 通道创建环境
    log_info "尝试创建环境..."
    if ! conda create -y -n "$CONDA_ENV" python="$PYTHON_VERSION"; then
        log_warn "使用默认通道创建环境失败，尝试使用 conda-forge 通道"

        # 尝试使用 conda-forge 通道
        if ! conda create -y -n "$CONDA_ENV" -c conda-forge python="$PYTHON_VERSION"; then
            log_error "创建conda环境失败，尝试最后的解决方案"

            # 最后尝试：完全移除有问题的通道并重试
            conda config --remove-key channels 2>/dev/null || true
            conda config --add channels conda-forge
            conda config --add channels defaults

            if ! conda create -y -n "$CONDA_ENV" python="$PYTHON_VERSION"; then
                log_error "所有方法均失败，无法创建conda环境"
                return 1
            fi
        fi
    fi

    # 激活新环境
    log_info "激活新创建的环境..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        log_error "无法激活新创建的Conda环境: $CONDA_ENV"
        return 1
    }

    log_info "conda环境 '$CONDA_ENV' 创建成功"
    return 0
}
