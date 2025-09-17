#!/bin/bash

# 安装PyTorch模块：安装PyTorch和相关依赖

# 安装PyTorch
install_pytorch() {
    show_progress "安装PyTorch"

    log_message "INFO" "开始安装PyTorch $PYTORCH_VERSION..."

    # 检查是否已激活conda环境
    if [[ -z "$CONDA_PREFIX" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        log_message "ERROR" "未正确激活Conda环境，尝试重新激活..."

        # 找到conda.sh位置
        local conda_root=$(dirname $(dirname $(which conda 2>/dev/null || echo "/root/miniconda3/bin/conda")))
        local conda_sh="$conda_root/etc/profile.d/conda.sh"

        # 加载conda.sh并激活环境
        if [[ -f "$conda_sh" ]]; then
            log_message "INFO" "加载conda.sh: $conda_sh"
            source "$conda_sh"
            conda activate "$CONDA_ENV" || {
                log_message "ERROR" "无法激活Conda环境: $CONDA_ENV"
                return 1
            }
        else
            log_message "WARN" "找不到conda.sh，尝试使用conda shell hook"
            # 使用更安全的方法获取conda shell hook
            local conda_hook_output
            if conda_hook_output=$(conda shell.bash hook 2>/dev/null); then
                source /dev/stdin <<< "$conda_hook_output"
            else
                log_message "ERROR" "无法获取conda shell hook，conda环境可能无法正常工作"
                return 1
            fi
            conda activate "$CONDA_ENV" || {
                log_message "ERROR" "无法激活Conda环境: $CONDA_ENV"
                return 1
            }
        fi

        # 验证激活状态
        if [[ -z "$CONDA_PREFIX" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
            log_message "ERROR" "无法激活Conda环境: $CONDA_ENV，无法安装PyTorch"
            return 1
        else
            log_message "INFO" "成功激活conda环境: $CONDA_ENV (CONDA_PREFIX=$CONDA_PREFIX)"
        fi
    else
        log_message "INFO" "已激活Conda环境: $CONDA_ENV (CONDA_PREFIX=$CONDA_PREFIX)"
    fi

    # 检查是否已安装PyTorch及其版本
    if python -c "import torch" &>/dev/null; then
        log_message "INFO" "检测到已安装PyTorch"

        # 获取当前安装的PyTorch版本
        local current_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_message "INFO" "当前PyTorch版本: $current_version"

        # 检查是否是正确的版本
        if [[ "$current_version" == "$PYTORCH_VERSION"* ]]; then
            log_message "INFO" "当前PyTorch版本($current_version)符合要求($PYTORCH_VERSION)"

            # 检查PyTorch是否支持CUDA
            if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                log_message "INFO" "检测到PyTorch支持CUDA"

                # 检查CUDA版本是否符合要求
                local cuda_version_str=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
                if [[ -n "$cuda_version_str" ]]; then
                    log_message "INFO" "当前PyTorch使用的CUDA版本: $cuda_version_str"

                    # 提取CUDA主版本号和次版本号
                    local cuda_major=$(echo "$cuda_version_str" | cut -d'.' -f1)
                    local cuda_minor=$(echo "$cuda_version_str" | cut -d'.' -f2)
                    local required_major=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
                    local required_minor=$(echo "$CUDA_VERSION" | cut -d'.' -f2)

                    # 比较CUDA版本
                    if [[ "$cuda_major" -eq "$required_major" && "$cuda_minor" -ge "$required_minor" ]]; then
                        log_message "SUCCESS" "检测到正确的PyTorch安装，跳过安装步骤"

                        # 验证安装是否正常工作
                        verify_pytorch_installation
                        return 0
                    else
                        log_message "WARN" "PyTorch使用的CUDA版本($cuda_version_str)与要求的版本($CUDA_VERSION)不匹配，需要重新安装"
                    fi
                else
                    log_message "WARN" "无法获取PyTorch使用的CUDA版本，需要重新安装"
                fi
            else
                log_message "WARN" "PyTorch无法使用CUDA，需要重新安装"
            fi
        fi

        # 如果PyTorch版本不正确或CUDA配置不正确，则卸载
        log_message "INFO" "卸载当前PyTorch以进行重新安装..."

        # 使用pip卸载PyTorch相关包
        log_message "INFO" "使用pip卸载PyTorch相关包..."
        echo -e "${GREEN}卸载pip安装的PyTorch...${NC}"
        pip uninstall -y torch torchvision torchaudio || log_message "WARN" "pip卸载PyTorch失败，继续处理"

        # 清理可能的残留
        log_message "INFO" "清理可能的PyTorch残留文件..."
        find "$CONDA_PREFIX/lib/python"*"/site-packages/" -name "*torch*" -type d -exec rm -rf {} + 2>/dev/null || true
    else
        log_message "INFO" "未检测到已安装的PyTorch，将进行全新安装"
    fi

    # 安装PyTorch (仅支持GPU)
    install_pytorch_gpu

    # 验证PyTorch安装
    verify_pytorch_installation

    return 0
}

# 确保基础Python包已安装
ensure_basic_python_packages() {
    log_message "INFO" "确保基础Python包已安装..."

    # 检查并安装/更新setuptools
    log_message "INFO" "检查setuptools..."
    if ! python -c "import setuptools" &>/dev/null; then
        log_message "WARN" "未安装setuptools，现在安装..."
        echo -e "${GREEN}安装setuptools...${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ setuptools || log_message "WARN" "安装setuptools失败，可能影响后续安装"
    else
        log_message "INFO" "setuptools已安装，确保版本最新..."
        echo -e "${GREEN}更新setuptools...${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade setuptools || log_message "WARN" "更新setuptools失败，继续使用现有版本"
    fi

    # 检查并安装wheel
    log_message "INFO" "检查wheel..."
    if ! python -c "import wheel" &>/dev/null; then
        log_message "WARN" "未安装wheel，现在安装..."
        echo -e "${GREEN}安装wheel...${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ wheel || log_message "WARN" "安装wheel失败，可能影响后续安装"
    fi

    # 检查并安装distutils
    log_message "INFO" "检查distutils..."
    if ! python -c "import distutils" &>/dev/null; then
        log_message "WARN" "未安装distutils，现在安装..."
        echo -e "${GREEN}安装distutils...${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ distutils || {
            log_message "WARN" "无法通过pip安装distutils，尝试安装python3-distutils包..."
            apt-get update && apt-get install -y python3-distutils || log_message "WARN" "安装distutils失败，可能影响后续安装"
        }
    fi

    # 检查_distutils_hack问题
    log_message "INFO" "检查_distutils_hack..."
    if ! python -c "import _distutils_hack" &>/dev/null; then
        log_message "WARN" "检测到_distutils_hack模块缺失，尝试修复..."
        # 尝试重新安装setuptools和distutils
        echo -e "${GREEN}重新安装setuptools...${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ --force-reinstall setuptools || log_message "WARN" "重新安装setuptools失败，可能影响后续安装"
    fi

    log_message "INFO" "基础Python包检查完成"
    return 0
}

# 注意：pip配置已在conda_env_manager.sh中统一处理，此处不再重复配置

# 使用GPU安装PyTorch
install_pytorch_gpu() {
    log_message "INFO" "安装PyTorch GPU版本..."

    # 确保基础Python包已安装
    ensure_basic_python_packages

    # 配置PyTorch国内源策略 - 优先阿里云，避免重复配置
    local install_cmd=""
    local mirror_cmd=""
    local aliyun_index="https://mirrors.aliyun.com/pypi/simple/"
    local pytorch_wheel_url=""

    # 根据CUDA版本设置PyTorch wheel源
    case "$CUDA_VERSION" in
        "12.6")
            pytorch_wheel_url="https://download.pytorch.org/whl/cu126"
            ;;
        "12.8")
            pytorch_wheel_url="https://download.pytorch.org/whl/cu128"
            ;;
        *)
            log_message "ERROR" "不支持的CUDA版本: $CUDA_VERSION"
            return 1
            ;;
    esac

    # 配置安装命令 - 使用pip安装官方预编译的二进制包
    case "$PYTORCH_VERSION" in
        "2.6.0")
            # 官方源命令
            install_cmd="pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
            # 阿里云镜像命令 - 优先使用阿里云主源，PyTorch wheel作为额外源
            mirror_cmd="pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu126/ -i https://mirrors.aliyun.com/pypi/simple/"
            ;;
        "2.7.0")
            # 官方源命令
            install_cmd="pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128"
            # 阿里云镜像命令 - 优先使用阿里云主源，PyTorch wheel作为额外源
            mirror_cmd="pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ -i https://mirrors.aliyun.com/pypi/simple/"
            ;;
        *)
            log_message "ERROR" "不支持的PyTorch版本: $PYTORCH_VERSION"
            return 1
            ;;
        esac

    # 尝试多种安装方式 - 优先使用国内源策略
    local success=false
    local retry_count=0
    local max_retries=3

    # 方法1: 优先使用阿里云镜像源（国内源），带重试机制
    while [[ $retry_count -lt $max_retries && $success == false ]]; do
        retry_count=$((retry_count + 1))
        log_message "INFO" "尝试安装PyTorch (阿里云镜像源 - 第 $retry_count 次): $mirror_cmd"

        echo -e "${GREEN}安装PyTorch (阿里云镜像源)...${NC}"
        if $mirror_cmd; then
            success=true
            break
        else
            log_message "WARN" "阿里云镜像源第 $retry_count 次安装失败，等待 10 秒后重试..."
            sleep 10
        fi
    done

    # 方法2: 如果国内源失败，尝试官方源
    if [[ $success == false ]]; then
        log_message "INFO" "阿里云镜像源安装失败，尝试官方源..."
        retry_count=0
        while [[ $retry_count -lt $max_retries && $success == false ]]; do
            retry_count=$((retry_count + 1))
            log_message "INFO" "尝试安装PyTorch (官方源 - 第 $retry_count 次): $install_cmd"

            echo -e "${GREEN}安装PyTorch (官方源)...${NC}"
            if $install_cmd; then
                success=true
                break
            else
                log_message "WARN" "官方源第 $retry_count 次安装失败，等待 15 秒后重试..."
                sleep 15
            fi
        done
    fi

    if [[ $success == false ]]; then
        log_message "ERROR" "所有PyTorch安装方法都失败了"
        return 1
    fi

    # 注意：pip安装包已经包含cuDNN，不需要单独安装
    log_message "INFO" "PyTorch GPU版本安装完成"

    return 0
}

# 验证PyTorch安装
verify_pytorch_installation() {
    log_message "INFO" "验证PyTorch安装..."

    # 创建临时Python脚本
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'EOF'
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
EOF

    # 执行验证脚本
    log_message "INFO" "执行PyTorch验证脚本..."
    if ! python "$temp_script"; then
        log_message "ERROR" "PyTorch验证失败，安装可能不完整"
        rm -f "$temp_script"
        return 1
    fi

    # 清理临时文件
    rm -f "$temp_script"

    log_message "INFO" "PyTorch安装验证成功"
    return 0
}

# 安装PyTorch相关依赖
install_pytorch_dependencies() {
    log_message "INFO" "安装PyTorch相关依赖..."

    # 安装基础依赖
    local dependencies=(
        "transformers"
        "accelerate"
        "bitsandbytes"
        "safetensors"
        "sentencepiece"
        "einops"
        "peft"
        "optimum"
    )

    # 安装依赖 - 使用阿里云镜像源
    for dep in "${dependencies[@]}"; do
        log_message "INFO" "安装依赖: $dep"
        echo -e "${GREEN}安装依赖: $dep${NC}"
        pip install -i https://mirrors.aliyun.com/pypi/simple/ $dep
    done

    # 安装特定版本的依赖 - 使用阿里云镜像源
    log_message "INFO" "安装特定版本的依赖..."
    echo -e "${GREEN}安装 triton...${NC}"
    pip install -i https://mirrors.aliyun.com/pypi/simple/ triton

    log_message "INFO" "PyTorch相关依赖安装完成"
    return 0
}
