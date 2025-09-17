#!/bin/bash

# Flash Attention安装模块 - 优化版本
# 安装策略优先级：
# 方法1：直接下载wheel文件安装（首选，最快最稳定）
# 方法2：pip安装（备选，使用镜像源）
# 方法3：源码编译安装（最后手段）

# 默认配置
DEFAULT_FLASH_ATTENTION_VERSION="2.7.4.post1"
MAX_PARALLEL_JOBS=${MAX_JOBS:-4}

# 日志函数（如果未定义）
if ! command -v log_info &> /dev/null; then
    log_info() { echo "[INFO] $1"; }
    log_warn() { echo "[WARN] $1"; }
    log_error() { echo "[ERROR] $1"; }
fi

# 获取Flash Attention版本
get_flash_attention_version() {
    local version_param="$1"
    local version=""
    
    # 优先使用传入的参数
    if [[ -n "$version_param" && "$version_param" != "latest" ]]; then
        version="$version_param"
    # 其次使用环境变量
    elif [[ -n "$FLASH_ATTN_VERSION" && "$FLASH_ATTN_VERSION" != "latest" ]]; then
        version="$FLASH_ATTN_VERSION"
    fi
    
    if [[ -n "$version" && "$version" != "latest" ]]; then
        echo "$version"
        return
    fi
    
    # 获取最新版本
    local latest_version=""
    
    # 尝试从PyPI获取
    if command -v curl &> /dev/null; then
        latest_version=$(curl -s "https://pypi.org/pypi/flash-attn/json" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data['info']['version'])
except:
    pass
" 2>/dev/null)
    fi
    
    # 如果失败，使用默认版本
    if [[ -z "$latest_version" ]]; then
        latest_version="$DEFAULT_FLASH_ATTENTION_VERSION"
    fi
    
    echo "$latest_version"
}

# 优化pip配置
optimize_pip_config() {
    log_info "优化pip配置..."
    
    # 设置国内镜像源
    local pip_conf_dir="$HOME/.config/pip"
    local pip_conf_file="$pip_conf_dir/pip.conf"
    
    mkdir -p "$pip_conf_dir"
    
    cat > "$pip_conf_file" << EOF
[global]
timeout = 120
retries = 5
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
extra-index-url = https://pypi.org/simple
EOF
    
    log_info "pip配置已优化"
}

# 检查Flash Attention是否已安装
check_flash_attention_installed() {
    local version="$1"
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未找到"
        return 1
    fi
    
    local check_script="
try:
    import flash_attn
    print('Flash Attention已安装')
    print(f'版本: {flash_attn.__version__}')
    exit(0)
except ImportError:
    print('Flash Attention未安装')
    exit(1)
except Exception as e:
    print(f'检查失败: {e}')
    exit(1)
"
    
    if python3 -c "$check_script" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 卸载现有Flash Attention
uninstall_flash_attention() {
    log_info "卸载现有Flash Attention..."
    
    # 使用pip卸载
    pip uninstall -y flash-attn 2>/dev/null || true
    
    # 清理可能的残留文件
    local site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [[ -n "$site_packages" ]]; then
        rm -rf "$site_packages"/flash_attn* 2>/dev/null || true
        rm -rf "$site_packages"/flash_attn 2>/dev/null || true
    fi
    
    # 清理用户目录
    local user_site=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
    if [[ -n "$user_site" ]]; then
        rm -rf "$user_site"/flash_attn* 2>/dev/null || true
        rm -rf "$user_site"/flash_attn 2>/dev/null || true
    fi
    
    log_info "Flash Attention卸载完成"
}

# 清理安装残留
cleanup_flash_attention_install() {
    log_info "清理Flash Attention安装残留..."
    
    # 清理pip缓存
    pip cache purge 2>/dev/null || true
    
    # 清理临时文件
    rm -rf /tmp/flash-attention* 2>/dev/null || true
    rm -rf /tmp/flash_attn* 2>/dev/null || true
    
    # 清理Python缓存
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_info "清理完成"
}

# 获取系统信息用于wheel下载链接
get_system_info_for_wheel() {
    local python_version=""
    local torch_version=""
    local abi_flag="FALSE"
    
    # 获取Python版本
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)
    
    # 获取PyTorch版本
    torch_version=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
    if [[ -n "$torch_version" ]]; then
        # 提取主版本号，例如2.7.0 -> 2.7
        torch_version=$(echo "$torch_version" | cut -d. -f1,2)
    fi
    
    # 检查ABI设置
    local abi_check=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null)
    if [[ "$abi_check" == "TRUE" ]]; then
        abi_flag="TRUE"
    fi
    
    echo "$python_version|$torch_version|$abi_flag"
}

# 方法1：直接下载wheel安装（首选方案）
install_flash_attention_method1() {
    local version="$1"
    log_info "方法1：尝试直接下载wheel安装 Flash Attention $version..."
    
    # 获取系统信息
    local sys_info=$(get_system_info_for_wheel)
    local python_version=$(echo "$sys_info" | cut -d'|' -f1)
    local torch_version=$(echo "$sys_info" | cut -d'|' -f2)
    local abi_flag=$(echo "$sys_info" | cut -d'|' -f3)
    
    if [[ -z "$python_version" || -z "$torch_version" ]]; then
        log_warn "无法获取系统信息，跳过wheel直接下载"
        return 1
    fi
    
    log_info "系统信息: Python $python_version, PyTorch $torch_version, ABI $abi_flag"
    
    # 构建wheel文件名
    local wheel_name="flash_attn-${version}+cu12torch${torch_version}cxx11abi${abi_flag}-cp${python_version}-cp${python_version}-linux_x86_64.whl"
    
    # 首先检查本地包文件夹
    local package_dir="/mnt/c/Users/li_ao/Desktop/Deployment_Kt/packages"
    local local_wheel_path="$package_dir/$wheel_name"
    
    log_info "检查本地包文件夹: $package_dir"
    
    if [[ -f "$local_wheel_path" ]]; then
        log_info "找到本地wheel文件: $wheel_name"
        log_info "文件大小: $(ls -lh "$local_wheel_path" | awk '{print $5}')"
        
        # 验证wheel文件完整性
        if [[ -s "$local_wheel_path" ]]; then
            log_info "开始安装本地wheel文件..."
            if pip install "$local_wheel_path" 2>&1; then
                if check_flash_attention_installed "$version"; then
                    log_info "方法1成功：使用本地wheel文件安装 Flash Attention $version 成功"
                    return 0
                fi
            fi
            log_warn "本地wheel文件安装失败，尝试下载..."
        else
            log_warn "本地wheel文件为空或损坏，尝试下载..."
        fi
    else
        log_info "本地未找到wheel文件: $wheel_name"
        
        # 检查本地包文件夹中的其他wheel文件
        if [[ -d "$package_dir" ]]; then
            log_info "本地包文件夹内容:"
            ls -lh "$package_dir"/flash_attn-*.whl 2>/dev/null | head -10 || log_info "（无wheel文件）"
        else
            log_info "本地包文件夹不存在: $package_dir"
        fi
    fi
    
    # 如果没有找到本地文件，则下载
    local download_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${version}/${wheel_name}"
    
    log_info "尝试下载wheel: $wheel_name"
    log_info "下载链接: $download_url"
    
    # 创建临时目录
    local temp_dir=$(mktemp -d)
    local wheel_path="$temp_dir/$wheel_name"
    
    # 下载wheel文件
    log_info "开始下载wheel文件..."
    if ! curl -L --connect-timeout 30 --max-time 300 --retry 3 -o "$wheel_path" "$download_url" 2>&1; then
        log_warn "wheel文件下载失败"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # 验证文件是否下载成功
    if [[ ! -f "$wheel_path" || ! -s "$wheel_path" ]]; then
        log_warn "wheel文件下载不完整或为空"
        rm -rf "$temp_dir"
        return 1
    fi
    
    log_info "wheel文件下载成功，大小: $(ls -lh "$wheel_path" | awk '{print $5}')"
    
    # 安装下载的wheel文件
    log_info "开始安装wheel文件..."
    if pip install "$wheel_path" 2>&1; then
        if check_flash_attention_installed "$version"; then
            log_info "方法1成功：Flash Attention $version wheel安装成功"
            rm -rf "$temp_dir"
            return 0
        fi
    fi
    
    log_warn "wheel文件安装失败"
    rm -rf "$temp_dir"
    return 1
}

# 方法2：pip安装（备选方案）
install_flash_attention_method2() {
    local version="$1"
    log_info "方法2：尝试pip安装 Flash Attention $version..."
    
    # 安装构建依赖
    log_info "安装构建依赖..."
    pip install -i https://mirrors.aliyun.com/pypi/simple/ packaging ninja --no-cache-dir 2>/dev/null || true
    
    # 尝试不同的pip安装选项组合，包含网络优化
    local install_options=(
        "--timeout 300 --retries 5"
        "--timeout 300 --retries 5 --no-cache-dir"
        "--timeout 300 --retries 5 --no-deps"
        "--timeout 300 --retries 5 --no-cache-dir --no-deps"
        "--timeout 300 --retries 5 --use-pep517"
        "--timeout 300 --retries 5 --no-cache-dir --use-pep517"
        "--timeout 300 --retries 5 --no-build-isolation"
        "--timeout 300 --retries 5 --no-cache-dir --no-build-isolation"
        "--timeout 300 --retries 5 --force-reinstall"
        "--timeout 300 --retries 5 --force-reinstall --no-cache-dir"
        "--timeout 300 --retries 5 --force-reinstall --no-deps"
        "--timeout 300 --retries 5 --force-reinstall --no-cache-dir --no-deps"
        "--timeout 300 --retries 5 --force-reinstall --use-pep517"
        "--timeout 300 --retries 5 --force-reinstall --no-cache-dir --use-pep517"
        "--timeout 300 --retries 5 --force-reinstall --no-build-isolation"
        "--timeout 300 --retries 5 --force-reinstall --no-cache-dir --no-build-isolation"
    )
    
    # 设置环境变量以优化网络连接
    export PIP_TIMEOUT=300
    export PIP_RETRIES=5
    export PIP_DEFAULT_TIMEOUT=300
    
    for option in "${install_options[@]}"; do
        log_info "尝试安装选项: pip install flash-attn==$version $option"
        
        if pip install -i https://mirrors.aliyun.com/pypi/simple/ flash-attn=="$version" $option 2>&1; then
            if check_flash_attention_installed "$version"; then
                log_info "方法2成功：Flash Attention $version 安装成功"
                return 0
            fi
        fi
        
        log_warn "安装选项失败，尝试下一个选项..."
        sleep 2  # 短暂延迟后重试
    done
    
    log_warn "方法2失败：所有pip安装选项都失败"
    return 1
}

# 克隆源码（带重试机制）
clone_flash_attention_source() {
    local repo_dir="$1"
    local version="$2"
    
    log_info "克隆Flash Attention源码..."
    
    # 尝试从GitHub克隆
    if git clone --branch "v$version" --depth 1 https://github.com/Dao-AILab/flash-attention.git "$repo_dir" 2>&1; then
        log_info "成功从GitHub克隆源码"
        return 0
    fi
    
    log_warn "GitHub克隆失败，尝试Gitee镜像..."
    
    # 尝试从Gitee克隆
    if git clone --branch "v$version" --depth 1 https://gitee.com/mirrors/flash-attention.git "$repo_dir" 2>&1; then
        log_info "成功从Gitee克隆源码"
        return 0
    fi
    
    log_error "所有源码克隆尝试都失败"
    return 1
}

# 方法3：源码编译安装（最后手段）
install_flash_attention_method3() {
    local version="$1"
    log_info "方法3：尝试源码编译安装 Flash Attention $version..."
    
    # 创建临时目录
    local temp_dir=$(mktemp -d)
    local repo_dir="$temp_dir/flash-attention"
    
    # 克隆源码
    if ! clone_flash_attention_source "$repo_dir" "$version"; then
        rm -rf "$temp_dir"
        return 1
    fi
    
    # 进入源码目录
    cd "$repo_dir"
    
    # 设置编译环境
    export MAX_JOBS="$MAX_PARALLEL_JOBS"
    export FLASH_ATTENTION_FORCE_BUILD="TRUE"
    
    # 设置CUDA架构（如果已定义）
    if [[ -n "$TORCH_CUDA_ARCH_LIST" && "$TORCH_CUDA_ARCH_LIST" != "auto" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
        log_info "使用指定的CUDA架构: $TORCH_CUDA_ARCH_LIST"
    fi
    
    log_info "开始源码编译安装（使用$MAX_PARALLEL_JOBS个并行任务）..."
    
    # 尝试安装
    if pip install -i https://mirrors.aliyun.com/pypi/simple/ -e . --no-build-isolation 2>&1; then
        if check_flash_attention_installed "$version"; then
            log_info "方法3成功：Flash Attention $version 源码编译安装成功"
            cd - > /dev/null
            rm -rf "$temp_dir"
            return 0
        fi
    fi
    
    log_error "方法3失败：源码编译安装失败"
    cd - > /dev/null
    rm -rf "$temp_dir"
    return 1
}

# 主安装函数
install_flash_attention() {
    local version_param="$1"
    local start_time=$(date +%s)
    
    # 获取版本
    local version=$(get_flash_attention_version "$version_param")
    if [[ -z "$version" ]]; then
        log_error "无法确定Flash Attention版本"
        return 1
    fi
    
    log_info "目标Flash Attention版本: $version"
    
    # 安装前检查：如果已经安装且版本匹配，直接验证功能
    log_info "检查Flash Attention安装状态..."
    if check_flash_attention_installed "$version"; then
        log_info "发现已安装的Flash Attention $version"
        log_info "进行功能验证..."
        
        # 快速验证功能
        local quick_verify_script="
import torch
import flash_attn
print(f'已安装Flash Attention版本: {flash_attn.__version__}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    from flash_attn import flash_attn_func
    print('Flash Attention函数可正常导入')
else:
    print('警告：CUDA不可用')
print('快速验证通过')
"
        
        if python3 -c "$quick_verify_script" 2>/dev/null; then
            log_info "✓ Flash Attention已安装且功能正常，无需重新安装"
            return 0
        else
            log_warn "已安装的Flash Attention功能验证失败，需要重新安装"
        fi
    else
        log_info "未检测到Flash Attention $version 或版本不匹配"
    fi
    
    log_info "开始安装Flash Attention..."
    
    # 注意：pip配置已在conda_env_manager.sh中统一处理，此处不再重复配置
    
    # 卸载现有版本（仅在需要重新安装时）
    log_info "卸载现有Flash Attention..."
    uninstall_flash_attention
    
    # 清理残留
    cleanup_flash_attention_install
    
    # 方法1：直接下载wheel安装（首选方案）
    log_info "执行方法1：直接下载wheel安装..."
    if install_flash_attention_method1 "$version"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "Flash Attention安装成功，耗时 ${duration} 秒"
        return 0
    fi
    
    log_warn "方法1失败，继续执行方法2..."
    
    # 方法2：pip安装（备选方案）
    log_info "执行方法2：pip安装..."
    if install_flash_attention_method2 "$version"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "Flash Attention安装成功，耗时 ${duration} 秒"
        return 0
    fi
    
    log_warn "方法2失败，继续执行方法3..."
    
    # 方法3：源码编译安装
    log_info "执行方法3：源码编译安装..."
    if install_flash_attention_method3 "$version"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "Flash Attention安装成功，耗时 ${duration} 秒"
        return 0
    fi
    
    # 所有方法都失败
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_error "Flash Attention安装失败，耗时 ${duration} 秒"
    
    # 最终清理
    cleanup_flash_attention_install
    
    return 1
}

# 检查Flash Attention是否已安装且版本匹配
check_flash_attention_installed() {
    local target_version="$1"
    
    # 检查Python包是否已安装
    if ! python3 -c "import flash_attn" 2>/dev/null; then
        return 1
    fi
    
    # 获取已安装的版本
    local installed_version=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    if [[ -z "$installed_version" ]]; then
        return 1
    fi
    
    # 版本匹配检查
    if [[ "$installed_version" == "$target_version" ]]; then
        return 0
    fi
    
    # 如果目标版本是latest，接受任何已安装版本
    if [[ "$target_version" == "latest" ]]; then
        return 0
    fi
    
    return 1
}

# 验证安装函数
verify_flash_attention_installation() {
    log_info "验证Flash Attention安装..."
    
    # 检查PyTorch是否可用
    log_info "检查PyTorch环境..."
    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch未安装，无法验证Flash Attention"
        return 1
    fi
    
    # 首先检查基本导入
    log_info "检查Flash Attention基本导入..."
    local basic_import_script="
import torch
import flash_attn
print(f'Flash Attention版本: {flash_attn.__version__}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
    
    if ! python3 -c "$basic_import_script" 2>&1; then
        log_error "Flash Attention基本导入失败"
        return 1
    fi
    
    # 然后检查CUDA环境（如果可用）
    log_info "检查CUDA环境..."
    local cuda_check_script="
import torch
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告：CUDA不可用，将跳过GPU功能测试')
"
    
    python3 -c "$cuda_check_script" 2>&1
    
    # 最后进行功能测试
    log_info "进行Flash Attention功能测试..."
    local verify_script="
import torch
import flash_attn

print(f'Flash Attention版本: {flash_attn.__version__}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    
    # 简单功能测试
    try:
        from flash_attn import flash_attn_func
        print('Flash Attention函数导入成功')
        
        # 创建测试张量
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # 执行flash attention
        out = flash_attn_func(q, k, v, dropout_p=0.1)
        print(f'Flash Attention功能测试成功，输出形状: {out.shape}')
        
    except Exception as e:
        print(f'功能测试失败: {e}')
        exit(1)
else:
    print('警告：CUDA不可用，无法进行完整功能测试')

print('Flash Attention安装验证完成')
"
    
    if python3 -c "$verify_script" 2>&1; then
        log_info "Flash Attention安装验证成功"
        return 0
    else
        log_error "Flash Attention安装验证失败"
        return 1
    fi
}

# 主函数
main() {
    local version="${1:-latest}"
    
    log_info "Flash Attention安装模块启动"
    log_info "目标版本: $version"
    log_info "并行编译任务数: $MAX_PARALLEL_JOBS"
    
    # 执行安装
    if install_flash_attention "$version"; then
        # 验证安装
        if verify_flash_attention_installation; then
            log_info "Flash Attention安装和验证全部完成"
            return 0
        else
            log_warn "安装成功但验证失败"
            return 1
        fi
    else
        log_error "Flash Attention安装失败"
        return 1
    fi
}

# 如果直接执行此脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi