#!/bin/bash

# KTransformers 安装验证模块
# 包含安装验证和完成信息显示功能

# 验证安装并显示完成信息
check_install_verification() {
    show_progress "验证安装并显示完成信息"
    
    log_info "验证KTransformers安装..."
    
    # 确保conda环境已激活
    if [[ -z "$CONDA_PREFIX" ]]; then
        log_error "未检测到激活的conda环境，无法验证安装"
        return 1
    fi
    
    # 验证KTransformers是否可导入
    log_info "检查KTransformers是否可导入..."
    if ! python -c "import ktransformers" &> /dev/null; then
        log_error "KTransformers无法导入，安装可能有问题"
        return 1
    fi
    
    # 获取KTransformers版本
    local kt_version=$(python -c "import ktransformers; print(ktransformers.__version__)" 2>/dev/null)
    if [[ -n "$kt_version" ]]; then
        log_info "KTransformers版本: $kt_version"
    else
        log_warn "无法获取KTransformers版本"
    fi
    
    # 验证Python环境
    log_info "验证Python环境..."
    local python_version=$(python --version 2>&1)
    log_info "Python版本: $python_version"
    
    # 验证PyTorch
    log_info "验证PyTorch安装..."
    if ! python -c "import torch" &> /dev/null; then
        log_error "PyTorch无法导入，安装可能有问题"
    else
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_info "PyTorch版本: $torch_version"
        
        # 检查CUDA是否可用
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_info "CUDA可用: 是"
            local cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            log_info "CUDA版本: $cuda_version"
            
            # 获取GPU信息
            local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            log_info "GPU数量: $gpu_count"
            
            if [[ "$gpu_count" -gt 0 ]]; then
                for ((i=0; i<gpu_count; i++)); do
                    local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
                    log_info "GPU $i: $gpu_name"
                done
            fi
        else
            log_info "CUDA可用: 否 (使用CPU模式)"
        fi
    fi
    
    # 显示安装完成信息
    echo -e "\n${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│      KTransformers 安装完成!       │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
    echo -e "\n${BLUE}安装信息:${NC}"
    echo -e " ${GREEN}•${NC} 安装目录:      ${BLUE}$KT_ROOT${NC}"
    echo -e " ${GREEN}•${NC} Python环境:    ${BLUE}$python_version${NC}"
    echo -e " ${GREEN}•${NC} PyTorch版本:   ${BLUE}$torch_version${NC}"
    echo -e " ${GREEN}•${NC} KT版本:        ${BLUE}$kt_version${NC}"
    
    # 显示启动命令
    echo -e "\n${BLUE}启动服务:${NC}"
    echo -e " ${GREEN}1.${NC} 激活环境:     ${YELLOW}conda activate $CONDA_ENV${NC}"
    echo -e " ${GREEN}2.${NC} 启动服务:     ${YELLOW}python -m ktransformers.serve --model MODEL_NAME${NC}"
    
    echo -e "\n${GREEN}感谢使用KTransformers!${NC}"
    
    return 0
}

# 最终环境检查
final_environment_check() {
    show_progress "最终环境检查"
    
    # 检查Python环境
    log_info "Python环境:"
    local python_version=$(python --version 2>&1)
    log_info "  Python版本: $python_version"
    
    # 检查PyTorch
    log_info "PyTorch环境:"
    if python -c "import torch" &> /dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        local cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        log_info "  PyTorch版本: $torch_version"
        log_info "  CUDA可用: $cuda_available"
        [[ "$cuda_available" == "True" ]] && log_info "  CUDA版本: $cuda_version"
    else
        log_error "  PyTorch未安装或导入失败"
    fi
    
    # 检查Flash Attention
    log_info "Flash Attention环境:"
    
    # 首先检查PyTorch
    if python -c "import torch" &> /dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        log_info "  PyTorch版本: $torch_version"
        log_info "  CUDA可用: $cuda_available"
        
        if [[ "$cuda_available" == "True" ]]; then
            local cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            log_info "  CUDA版本: $cuda_version"
        fi
        
        # 然后检查Flash Attention
        if python -c "import flash_attn" &> /dev/null; then
            local flash_attn_version=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
            log_info "  Flash Attention版本: $flash_attn_version"
            
            # 验证Flash Attention功能
            log_info "  验证Flash Attention功能..."
            local flash_test_script="
import torch
import flash_attn
from flash_attn import flash_attn_func

# 基本功能测试
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
                log_info "  ✓ Flash Attention功能正常"
            else
                log_warn "  ✗ Flash Attention功能异常"
            fi
        else
            log_warn "  Flash Attention未安装或导入失败"
        fi
    else
        log_error "  PyTorch未安装，无法检查Flash Attention"
    fi
    
    # 检查KTransformers
    log_info "KTransformers环境:"
    if python -c "import ktransformers" &> /dev/null; then
        local kt_version=$(python -c "import ktransformers; print(ktransformers.__version__)" 2>/dev/null)
        log_info "  KTransformers版本: $kt_version"
    else
        log_error "  KTransformers未安装或导入失败"
    fi
    
    log_info "最终环境检查完成"
    return 0
}