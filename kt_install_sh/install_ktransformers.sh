#!/bin/bash

# KTransformers 模块化自动安装脚本
# 
# 功能说明：
# 此脚本是KTransformers项目的自动化安装工具，支持在Ubuntu系统上部署多种大语言模型。
# 脚本采用模块化设计，将不同功能拆分为独立子模块，提高代码可维护性和扩展性。
# 
# 更新日志：
# - 新增Flash Attention源码编译选项，支持更多硬件配置
# - 调整安装流程，将用户交互部分移动到GPU和CUDA安装之后
# - 新增apt国内源配置选项
# - 优化各模块功能位置，确保配置在正确的流程中执行
# - 集中所有用户交互到user_interaction.sh中，避免重复询问
# - 优化系统检测流程，确保CPU架构和NUMA检测只在check_system阶段进行，避免重复检测

# 检查sudo权限可用性
check_sudo_available() {
    if ! command -v sudo &> /dev/null; then
        echo "错误: sudo命令不可用，请先安装sudo"
        echo "使用命令: apt update && apt install -y sudo"
        exit 1
    fi
    
    # 测试sudo权限
    if ! sudo -n true 2>/dev/null; then
        echo "提示: 脚本需要sudo权限来安装系统软件包"
        echo "请输入密码以获取sudo权限..."
        if ! sudo true; then
            echo "错误: 无法获取sudo权限"
            exit 1
        fi
    fi
    echo "sudo权限检查通过"
}

# 安全执行需要sudo权限的命令
sudo_execute() {
    local cmd="$*"
    echo "执行需要权限的命令: $cmd"
    if ! sudo $cmd; then
        echo "错误: 命令执行失败: $cmd"
        return 1
    fi
    return 0
}

# 保存当前工作目录
ORIGINAL_PWD="$(pwd)"

# 设置脚本目录
# 原生Ubuntu路径处理（已移除WSL转换）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULES_DIR="$SCRIPT_DIR/kt_install_sh"

# 加载初始化设置模块（首先加载，设置基础变量和函数）
source "$MODULES_DIR/init_settings.sh"

# 加载各个功能模块
source "$MODULES_DIR/setup_env_vars.sh"      # 环境变量设置模块（新增）
source "$MODULES_DIR/user_interaction.sh"    # 用户交互模块（包含所有用户配置收集功能）
source "$MODULES_DIR/check_system.sh"        # 系统检查模块
source "$MODULES_DIR/install_basic_tools.sh" # 基础工具安装模块（包含apt源配置）
source "$MODULES_DIR/install_gpu_cuda.sh"    # GPU和CUDA安装模块
source "$MODULES_DIR/install_miniconda.sh"   # Miniconda安装模块（使用用户配置的conda源）
source "$MODULES_DIR/conda_env_manager.sh"   # Conda环境管理模块（使用用户配置的pip源）
source "$MODULES_DIR/install_pytorch.sh"     # PyTorch安装模块
source "$MODULES_DIR/install_build_tools.sh" # 编译工具安装模块
source "$MODULES_DIR/check_build_env.sh"     # 编译环境检查模块
source "$MODULES_DIR/check_install_verification.sh" # 安装验证模块
source "$MODULES_DIR/install_kt.sh"          # KTransformers安装模块
source "$MODULES_DIR/install_flash_attention.sh" # Flash Attention安装模块

# Flash Attention安装验证函数（复用现有代码）
verify_flash_attention_installation() {
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_warn "未检测到激活的conda环境，跳过Flash Attention验证"
        return 0
    fi
    
    log_info "验证Flash Attention安装..."
    
    # 复用check_install_verification.sh中的验证逻辑
    if python -c "import flash_attn" &> /dev/null; then
        local flash_attn_version=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
        log_info "Flash Attention版本: $flash_attn_version"
        
        # 基本功能测试
        local flash_test_script="
import torch
import flash_attn
from flash_attn import flash_attn_func

print('Flash Attention基本功能测试:')
print(f'  版本: {flash_attn.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    # 创建简单测试数据
    batch_size, seq_len, num_heads, head_dim = 1, 64, 4, 32
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    
    # 执行flash attention
    out = flash_attn_func(q, k, v, dropout_p=0.0)
    print(f'  功能测试: 成功 (输出形状: {out.shape})')
else:
    print('  警告: CUDA不可用，跳过GPU功能测试')

print('Flash Attention功能验证完成')
"
        
        if python -c "$flash_test_script" &> /dev/null; then
            log_info "✓ Flash Attention功能正常"
            return 0
        else
            log_warn "✗ Flash Attention功能异常"
            return 1
        fi
    else
        log_warn "Flash Attention未安装或导入失败"
        return 1
    fi
}

# 主函数
main() {
    # 注册清理函数
    trap cleanup_on_exit EXIT
    trap 'log_error "脚本被中断"; exit 130' INT
    trap 'log_error "脚本被终止"; exit 143' TERM
    
    # 1. 检查sudo权限可用性（不立即获取root权限）
    check_sudo_available
    
    # 2. 显示欢迎信息
    show_welcome_message
    export WELCOME_SHOWN="true"
    
    # 3. 安装基础工具（包含询问apt源配置）
    install_basic_tools || {
        handle_error $? "安装基础工具失败，无法继续安装" "true"
    }
    
    # 4. 系统环境检查（自动检测CPU架构和NUMA配置）
    check_system || {
        handle_error $? "系统环境与依赖检查失败，无法继续安装" "true"
    }
    
    # 5. 安装GPU驱动和CUDA（包含询问CUDA版本选择）
    install_gpu_cuda || {
        handle_error $? "安装GPU和CUDA失败，无法继续安装" "true"
    }
    
    # 导出关键环境变量，确保后续模块可以使用
    export CUDA_VERSION
    export CUDA_VERSION_SHORT
    export NEED_INSTALL_CUDA
    export FORCE_CUDA_REINSTALL
    
    # 6. 设置环境变量（基于系统检测结果）
    setup_env_vars || {
        handle_error $? "环境变量设置失败，无法继续安装" "true"
    }
    
    # 7. 用户交互：收集用户配置并确认（融合阶段）
    # 注意：CPU架构和NUMA配置已在系统检查中自动设置
    # 此阶段包含：自动检测配置显示、高级配置选择、编译环境设置、参数确认
    user_interaction || {
        handle_error $? "用户配置收集失败，无法继续安装" "true"
    }
    
    # 配置GitHub hosts加速（如果用户选择启用）
    if [[ "$USE_GIT_HOSTS" == "y" ]]; then
        setup_git_hosts "y" || {
            log_warn "GitHub hosts配置失败，但将继续安装"
        }
    fi
    
    # 8. 安装Miniconda（使用用户配置的conda源）
    install_miniconda || {
        handle_error $? "安装Miniconda失败，无法继续安装" "true"
    }
    
    # 9. 创建Conda环境（使用用户配置的pip源）
    conda_env_manager || {
        handle_error $? "创建Conda环境失败，无法继续安装" "true"
    }
    
    # 10. 安装PyTorch
    install_pytorch || {
        handle_error $? "安装PyTorch失败，无法继续安装" "true"
    }
    
    # 11. 安装编译工具和conda环境包
    install_build_tools || {
        handle_error $? "安装编译工具和conda环境包失败，无法继续安装" "true"
    }
    
    # 11.5. 设置全局编译环境以解决兼容性问题
    setup_global_compile_environment || {
        log_warn "设置全局编译环境失败，但将继续安装"
    }
    
    # 12. 编译前的环境检测
    check_build_env || {
        log_warn "编译环境检查失败，但将继续安装"
    }
    
    # 13. 安装Flash Attention
    install_flash_attention "$FLASH_ATTN_VERSION" || {
        log_warn "安装Flash Attention失败，但将继续安装"
    }
    
    # 13.5. 验证Flash Attention安装
    verify_flash_attention_installation || {
        log_warn "Flash Attention验证失败，但将继续安装"
    }
    
    # 14. 安装KTransformers
    install_kt || {
        handle_error $? "安装KTransformers失败，无法继续安装" "true"
    }
    
    # 15. 验证安装并显示完成信息
    check_install_verification || {
        handle_error $? "安装验证失败，但安装过程已完成" "false"
    }
    
    # 显示完成信息
    show_completion_message
    
    return 0
}

# 执行主函数
main "$@"