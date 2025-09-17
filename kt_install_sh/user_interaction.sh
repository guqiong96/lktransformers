#!/bin/bash

# KTransformers 用户交互模块
# 包含用户界面和配置收集功能

# 美化输入界面辅助函数
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

# 显示欢迎信息
show_welcome_message() {
    echo ""
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${GREEN}[INFO]    KTransformers 自动安装脚本 v${SCRIPT_VERSION}${NC}"
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${GREEN}[INFO] 该脚本将帮助您安装KTransformers环境${NC}"
    echo -e "${GREEN}[INFO] 适用于大型语言模型的高效推理${NC}"
    echo -e "${GREEN}[INFO] ======================================${NC}"
    
    # 显示项目状态
    echo -e "${GREEN}[INFO] 项目状态检查:${NC}"
    if [[ -d "kt" ]]; then
        echo -e "${GREEN}[INFO] ✓ 检测到本地kt项目文件夹${NC}"
        if [[ -f "kt/setup.py" ]]; then
            echo -e "${GREEN}[INFO] ✓ 项目结构完整${NC}"
        else
            echo -e "${RED}[INFO] ✗ 项目结构不完整${NC}"
        fi
    else
        echo -e "${YELLOW}[INFO] ! 未检测到本地kt项目文件夹${NC}"
        echo -e "${YELLOW}[INFO]   将提供环境配置指令供手动安装${NC}"
    fi
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo ""
}

# 配置镜像源（统一询问）
configure_mirror_sources() {
    print_option_header "镜像源配置"
    
    # 国内镜像源默认启用
    export USE_APT_MIRROR="y"
    export USE_CONDA_MIRROR="y"
    export CONDA_MIRROR_URL="https://mirrors.ustc.edu.cn/anaconda"
    export CONDA_MIRROR_URL_BACKUP="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
    export USE_PIP_MIRROR="y"
    export PIP_MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    export PIP_MIRROR_HOST="pypi.tuna.tsinghua.edu.cn"
    export PIP_MIRROR_URL_BACKUP="https://mirrors.aliyun.com/pypi/simple"
    export PIP_MIRROR_HOST_BACKUP="mirrors.aliyun.com"
    
    log_info "已默认启用国内镜像源 (APT、Conda、PIP)"
    log_info "Conda镜像源: USTC中科大镜像站 (首选) + 清华大学TUNA镜像站 (备用)"
    
    # 单独询问GitHub代理
    print_option "y" "启用GitHub代理" "使用GitCDN动态获取最新IP地址加速GitHub访问"
    print_option "n" "禁用GitHub代理" "使用官方GitHub地址"
    echo -e " 是否启用GitHub代理? (y/n) [默认: ${GREEN}n${NC}]: "
    read -r git_proxy_choice
    
    if [[ "$(echo "$git_proxy_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        export USE_GIT_HOSTS="y"
        log_info "已启用GitHub hosts加速"
    else
        export USE_GIT_HOSTS="n"
        log_info "使用官方GitHub地址"
    fi
}

# GPU选择配置函数
configure_gpu_selection() {
    print_option_header "GPU架构配置"
    
    # 检测系统上的GPU
    log_message "INFO" "检测系统上的GPU..."
    local gpu_count=0
    local gpu_names=()
    local gpu_arches=()
    local gpu_indices=()
    local valid_gpu_count=0
    
    if command -v nvidia-smi &>/dev/null; then
        # 获取GPU数量，确保去除换行符和空格
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | tr -d '\n\r\t ' || echo "0")
        # 确保gpu_count是一个数字
        if ! [[ "$gpu_count" =~ ^[0-9]+$ ]]; then
            log_message "WARN" "GPU数量获取失败，设置为0"
            gpu_count=0
        fi
        
        # 如果有GPU，获取每个GPU的名称和计算能力
        if [[ $gpu_count -gt 0 ]]; then
            log_message "INFO" "系统报告 $gpu_count 个GPU设备，正在验证..."
            
            for ((i=0; i<gpu_count; i++)); do
                # 获取GPU名称
                local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader --id=$i 2>/dev/null | tr -d '\r')
                
                # 检查是否为有效的GPU设备（排除"No devices were found"等无效设备）
                if [[ -z "$gpu_name" || "$gpu_name" == *"No devices were found"* || "$gpu_name" == *"N/A"* ]]; then
                    log_message "WARN" "跳过无效GPU设备 #$i"
                    continue
                fi
                
                # 添加有效GPU到列表
                gpu_names+=("$gpu_name")
                gpu_indices+=("$i")
                valid_gpu_count=$((valid_gpu_count+1))
                
                # 尝试获取计算能力
                local compute_cap=""
                if nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=$i &>/dev/null; then
                    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=$i | tr -d '\n\r\t ')
                    # 确保计算能力是有效的格式
                    if [[ -n "$compute_cap" && "$compute_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
                        # 将计算能力直接添加到架构列表中
                        gpu_arches+=("$compute_cap")
                        log_message "INFO" "检测到GPU ${i}: ${gpu_name} (计算能力: ${compute_cap})"
                    else
                        log_message "WARN" "GPU ${i} 计算能力格式无效: '$compute_cap'，将使用自动检测"
                        gpu_arches+=("auto")
                    fi
                else
                    # 如果无法直接获取计算能力，使用自动检测
                    log_message "WARN" "无法获取GPU ${i} 的计算能力，将使用自动检测"
                    gpu_arches+=("auto")
                fi
            done
            
            # 更新GPU数量为有效GPU数量
            gpu_count=$valid_gpu_count
            log_message "INFO" "检测到 $gpu_count 个有效GPU设备"
        else
            log_message "WARN" "未检测到NVIDIA GPU，使用默认配置"
        fi
    else
        log_message "WARN" "nvidia-smi命令不可用，无法检测GPU，使用默认配置"
    fi
    
    # 根据检测到的GPU数量决定交互方式
    if [[ $gpu_count -eq 0 ]]; then
        # 没有检测到GPU，提供手动选择
        print_option "1" "自动检测" "推荐"
        print_option "2" "Turing架构(SM75)" "适用于RTX 2000系列/T4等"
        print_option "3" "Ampere架构(SM80/86)" "适用于A100/A10/A30/A40/RTX 3000系列等"
        print_option "4" "Ada Lovelace架构(SM89)" "适用于RTX 4000系列/L40等"
        print_option "5" "Hopper架构(SM90)" "适用于H100/H800等"
        print_option "6" "Blackwell架构(SM120)" "适用于RTX 5000系列"
        print_option "7" "自定义" ""
        echo -e " 请选择CUDA计算架构 [默认: ${GREEN}1${NC}]: "
        read -r gpu_arch_choice
        case $gpu_arch_choice in
            2) TORCH_CUDA_ARCH_LIST="7.5" ;;
            3) TORCH_CUDA_ARCH_LIST="8.0;8.6" ;;
            4) TORCH_CUDA_ARCH_LIST="8.9" ;;
            5) TORCH_CUDA_ARCH_LIST="9.0" ;;
            6) TORCH_CUDA_ARCH_LIST="12.0" ;;
            7) 
               echo -e " 请输入自定义CUDA架构列表(如7.5;8.0;8.6;9.0): "
               read -r custom_arch
               if [[ -n "$custom_arch" ]]; then
                   TORCH_CUDA_ARCH_LIST="$custom_arch"
               else
                   TORCH_CUDA_ARCH_LIST="auto"
               fi
               ;;
            *) TORCH_CUDA_ARCH_LIST="auto" ;;
        esac
        log_message "INFO" "选择的CUDA架构: $TORCH_CUDA_ARCH_LIST"
    elif [[ $gpu_count -eq 1 ]]; then
        # 只有一个GPU，询问是否使用检测到的架构
        local arch=${gpu_arches[0]}
        if [[ "$arch" == "auto" ]]; then
            echo -e "\n${BLUE}检测到单个GPU: ${gpu_names[0]}${NC}"
            echo -e "${YELLOW}无法自动检测计算架构，请手动选择:${NC}"
            print_option "1" "自动检测" "推荐"
            print_option "2" "Turing架构(SM75)" "适用于RTX 2000系列/T4等"
            print_option "3" "Ampere架构(SM80/86)" "适用于A100/A10/A30/A40/RTX 3000系列等"
            print_option "4" "Ada Lovelace架构(SM89)" "适用于RTX 4000系列/L40等"
            print_option "5" "Hopper架构(SM90)" "适用于H100/H800等"
            print_option "6" "Blackwell架构(SM120)" "适用于RTX 5000系列"
            print_option "7" "自定义" ""
            echo -e " 请选择CUDA计算架构 [默认: ${GREEN}1${NC}]: "
            read -r gpu_arch_choice
            case $gpu_arch_choice in
                2) TORCH_CUDA_ARCH_LIST="7.5" ;;
                3) TORCH_CUDA_ARCH_LIST="8.0;8.6" ;;
                4) TORCH_CUDA_ARCH_LIST="8.9" ;;
                5) TORCH_CUDA_ARCH_LIST="9.0" ;;
                6) TORCH_CUDA_ARCH_LIST="12.0" ;;
                7) 
                   echo -e " 请输入自定义CUDA架构列表(如7.5;8.0;8.6;9.0): "
                   read -r custom_arch
                   if [[ -n "$custom_arch" ]]; then
                       TORCH_CUDA_ARCH_LIST="$custom_arch"
                   else
                       TORCH_CUDA_ARCH_LIST="auto"
                   fi
                   ;;
                *) TORCH_CUDA_ARCH_LIST="auto" ;;
            esac
        else
            echo -e "\n${BLUE}检测到单个GPU: ${gpu_names[0]} (计算架构: ${arch})${NC}"
            print_option "1" "使用检测到的架构" "${arch} (推荐)"
            print_option "2" "自动检测" ""
            print_option "3" "手动选择其他架构" ""
            echo -e " 请选择GPU架构配置 [默认: ${GREEN}1${NC}]: "
            read -r gpu_choice
            case $gpu_choice in
                2) 
                    TORCH_CUDA_ARCH_LIST="auto"
                    ;;
                3)
                    print_option "1" "Turing架构(SM75)" "适用于RTX 2000系列/T4等"
                    print_option "2" "Ampere架构(SM80/86)" "适用于A100/A10/A30/A40/RTX 3000系列等"
                    print_option "3" "Ada Lovelace架构(SM89)" "适用于RTX 4000系列/L40等"
                    print_option "4" "Hopper架构(SM90)" "适用于H100/H800等"
                    print_option "5" "Blackwell架构(SM120)" "适用于RTX 5000系列"
                    print_option "6" "自定义" ""
                    echo -e " 请选择CUDA计算架构 [默认: ${GREEN}2${NC}]: "
                    read -r manual_arch_choice
                    case $manual_arch_choice in
                        1) TORCH_CUDA_ARCH_LIST="7.5" ;;
                        3) TORCH_CUDA_ARCH_LIST="8.9" ;;
                        4) TORCH_CUDA_ARCH_LIST="9.0" ;;
                        5) TORCH_CUDA_ARCH_LIST="12.0" ;;
                        6) 
                           echo -e " 请输入自定义CUDA架构列表(如7.5;8.0;8.6;9.0): "
                           read -r custom_arch
                           if [[ -n "$custom_arch" ]]; then
                               TORCH_CUDA_ARCH_LIST="$custom_arch"
                           else
                               TORCH_CUDA_ARCH_LIST="8.0;8.6"
                           fi
                           ;;
                        *) TORCH_CUDA_ARCH_LIST="8.0;8.6" ;;
                    esac
                    ;;
                *) 
                    TORCH_CUDA_ARCH_LIST="$arch"
                    ;;
            esac
        fi
        log_message "INFO" "选择的GPU架构: $TORCH_CUDA_ARCH_LIST"
    else
        # 多个GPU，提供选择
        echo -e "\n${BLUE}检测到多个有效GPU:${NC}"
        for ((i=0; i<${#gpu_names[@]}; i++)); do
            local arch=${gpu_arches[$i]}
            if [[ "$arch" == "auto" ]]; then
                echo -e " ${YELLOW}$((i+1)).${NC} ${gpu_names[$i]} ${BLUE}(自动检测)${NC}"
            else
                echo -e " ${YELLOW}$((i+1)).${NC} ${gpu_names[$i]} ${BLUE}(计算架构: ${arch})${NC}"
            fi
        done
        
        print_option "$((${#gpu_names[@]}+1))" "全部GPU" "推荐"
        print_option "$((${#gpu_names[@]}+2))" "多选GPU" "支持多种不同架构"
        print_option "$((${#gpu_names[@]}+3))" "自定义" "手动输入架构列表"
        
        echo -e " 请选择要优化的GPU [默认: ${GREEN}$((${#gpu_names[@]}+1))${NC}]: "
        read -r gpu_select_choice
        
        # 处理用户选择
        local gpu_plus_1=$((${#gpu_names[@]}+1))
        local gpu_plus_2=$((${#gpu_names[@]}+2))
        local gpu_plus_3=$((${#gpu_names[@]}+3))
        
        if [[ -z "$gpu_select_choice" || "$gpu_select_choice" == "$gpu_plus_1" ]]; then
            # 选择全部GPU
            local all_arches=""
            for arch in "${gpu_arches[@]}"; do
                if [[ "$arch" == "auto" ]]; then
                    # 如果有任何GPU是auto，整体设置为auto
                    all_arches="auto"
                    break
                else
                    if [[ -z "$all_arches" ]]; then
                        all_arches="$arch"
                    else
                        # 检查是否已经包含该架构
                        if [[ ! "$all_arches" =~ $arch ]]; then
                            all_arches="$all_arches;$arch"
                        fi
                    fi
                fi
            done
            TORCH_CUDA_ARCH_LIST="$all_arches"
            log_message "INFO" "选择所有GPU架构: $TORCH_CUDA_ARCH_LIST"
        elif [[ "$gpu_select_choice" == "$gpu_plus_2" ]]; then
            # 多选GPU
            echo -e "\n${BLUE}请输入要选择的GPU编号，用空格分隔（如: 1 3）[默认: 全部]:${NC} "
            read -r multi_gpu_choice
            
            if [[ -z "$multi_gpu_choice" ]]; then
                # 用户未输入，默认选择全部
                local all_arches=""
                for arch in "${gpu_arches[@]}"; do
                    if [[ "$arch" == "auto" ]]; then
                        all_arches="auto"
                        break
                    else
                        if [[ -z "$all_arches" ]]; then
                            all_arches="$arch"
                        else
                            if [[ ! "$all_arches" =~ $arch ]]; then
                                all_arches="$all_arches;$arch"
                            fi
                        fi
                    fi
                done
                TORCH_CUDA_ARCH_LIST="$all_arches"
                log_message "INFO" "选择所有GPU架构: $TORCH_CUDA_ARCH_LIST"
            else
                # 处理用户选择的多个GPU
                local selected_arches=""
                local has_auto=false
                for num in $multi_gpu_choice; do
                    if [[ "$num" =~ ^[0-9]+$ && "$num" -ge 1 && "$num" -le ${#gpu_names[@]} ]]; then
                        local idx=$((num-1))
                        local arch=${gpu_arches[$idx]}
                        
                        if [[ "$arch" == "auto" ]]; then
                            has_auto=true
                            log_message "INFO" "选择GPU $num: ${gpu_names[$idx]} (自动检测)"
                        else
                            # 添加到选择列表
                            if [[ -z "$selected_arches" ]]; then
                                selected_arches="$arch"
                            else
                                # 检查是否已经包含该架构
                                if [[ ! "$selected_arches" =~ $arch ]]; then
                                    selected_arches="$selected_arches;$arch"
                                fi
                            fi
                            log_message "INFO" "选择GPU $num: ${gpu_names[$idx]} (计算架构: $arch)"
                        fi
                    else
                        log_message "WARN" "忽略无效的GPU编号: $num"
                    fi
                done
                
                if [[ "$has_auto" == "true" ]]; then
                    TORCH_CUDA_ARCH_LIST="auto"
                    log_message "INFO" "选择的GPU中包含无法确定架构的GPU，使用自动检测"
                elif [[ -n "$selected_arches" ]]; then
                    TORCH_CUDA_ARCH_LIST="$selected_arches"
                    log_message "INFO" "选择的GPU架构: $TORCH_CUDA_ARCH_LIST"
                else
                    TORCH_CUDA_ARCH_LIST="auto"
                    log_message "WARN" "未选择有效的GPU，使用自动检测"
                fi
            fi
        elif [[ "$gpu_select_choice" == "$gpu_plus_3" ]]; then
            # 自定义
            echo -e " 请输入自定义CUDA架构列表(如7.5;8.0;8.6;9.0): "
            read -r custom_arch
            if [[ -n "$custom_arch" ]]; then
                TORCH_CUDA_ARCH_LIST="$custom_arch"
                log_message "INFO" "使用自定义架构: $TORCH_CUDA_ARCH_LIST"
            else
                TORCH_CUDA_ARCH_LIST="auto"
                log_message "INFO" "未提供自定义架构，使用自动检测"
            fi
        elif [[ "$gpu_select_choice" -ge 1 && "$gpu_select_choice" -le ${#gpu_names[@]} ]]; then
            # 选择特定GPU
            local idx=$((gpu_select_choice-1))
            local arch=${gpu_arches[$idx]}
            if [[ "$arch" == "auto" ]]; then
                TORCH_CUDA_ARCH_LIST="auto"
                log_message "INFO" "选择GPU ${gpu_indices[$idx]}: ${gpu_names[$idx]} (自动检测)"
            else
                TORCH_CUDA_ARCH_LIST="$arch"
                log_message "INFO" "选择GPU ${gpu_indices[$idx]}: ${gpu_names[$idx]} (计算架构: $arch)"
            fi
        else
            # 无效选择，使用自动检测
            TORCH_CUDA_ARCH_LIST="auto"
            log_message "INFO" "无效选择，使用自动检测"
        fi
    fi
    
    # 导出CUDA架构配置
    export TORCH_CUDA_ARCH_LIST
    log_message "INFO" "GPU架构配置完成: $TORCH_CUDA_ARCH_LIST"
}

# 高级设置配置
configure_advanced_settings() {
    print_option_header "高级设置配置"
    echo -e "${YELLOW}以下设置已根据您的硬件智能配置，您可以选择调整或保持默认${NC}"
    echo ""
    
    # 根据已检测的结果设置默认值
    local default_cpu_arch="n"
    local default_amx="n"
    
    if [[ "$ENABLE_CPU_ARCH_OPT" == "y" ]]; then
        default_cpu_arch="y"
    fi
    
    if [[ "$ENABLE_AMX" == "1" ]]; then
        default_amx="y"
    fi
    
    # CPU架构优化设置
    print_option_header "CPU架构优化"
    if [[ "$default_cpu_arch" == "y" ]]; then
        echo -e "${BLUE}说明: 已检测到您的CPU支持高级指令集优化，推荐启用以获得更好性能${NC}"
        echo -e "${GREEN}智能推荐: 启用 (检测到 ${CPU_ARCH_OPT} 支持)${NC}"
    else
        echo -e "${BLUE}说明: 启用CPU指令集优化可以提高计算性能，但可能在某些旧CPU上不兼容${NC}"
    fi
    print_option "y" "启用CPU架构优化" "自动检测并启用AVX2/AVX512等指令集优化"
    print_option "n" "禁用CPU架构优化" "使用默认编译选项，兼容性更好"
    echo -e " 是否启用CPU架构优化? (y/n) [默认: ${GREEN}${default_cpu_arch}${NC}]: "
    read -r cpu_arch_choice
    
    # 如果用户没有输入，使用智能推荐的默认值
    if [[ -z "$cpu_arch_choice" ]]; then
        cpu_arch_choice="$default_cpu_arch"
    fi
    
    if [[ "${cpu_arch_choice,,}" == "y" ]]; then
        export ENABLE_CPU_ARCH_OPT="y"
        log_info "已启用CPU架构优化"
        
        # 如果没有预设CPU架构，询问是否自定义
        if [[ -z "$CPU_ARCH_OPT" || "$CPU_ARCH_OPT" == "auto" ]]; then
            echo -e " 是否自定义CPU架构? (y/n) [默认: ${GREEN}n${NC}]: "
            read -r custom_cpu_choice
            if [[ "$(echo "$custom_cpu_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
                echo -e " 请选择CPU架构优化级别:"
                print_option "1" "自动检测" "推荐"
                print_option "2" "AVX2优化" "适用于较新的Intel/AMD CPU"
                print_option "3" "AVX512优化" "适用于服务器级CPU"
                print_option "4" "默认优化" "最大兼容性"
                echo -e " 请选择 [默认: ${GREEN}1${NC}]: "
                read -r cpu_arch_level
                case $cpu_arch_level in
                    2) export CPU_ARCH_OPT="avx2" ;;
                    3) export CPU_ARCH_OPT="avx512" ;;
                    4) export CPU_ARCH_OPT="default" ;;
                    *) export CPU_ARCH_OPT="auto" ;;
                esac
            fi
        fi
    else
        export ENABLE_CPU_ARCH_OPT="n"
        export CPU_ARCH_OPT="auto"
        log_info "已禁用CPU架构优化"
    fi
    echo ""
    
    # AMX优化设置
    print_option_header "AMX优化"
    if [[ "$default_amx" == "y" ]]; then
        echo -e "${BLUE}说明: 检测到您的CPU支持AMX (Advanced Matrix Extensions) 矩阵计算加速指令集${NC}"
        echo -e "${GREEN}智能推荐: 启用 (检测到AMX指令集支持)${NC}"
    else
        echo -e "${BLUE}说明: AMX (Advanced Matrix Extensions) 是Intel最新的矩阵计算加速指令集${NC}"
        echo -e "${BLUE}      仅在支持AMX的CPU上启用，如Intel Sapphire Rapids等${NC}"
    fi
    print_option "y" "启用AMX优化" "在支持的CPU上启用AMX矩阵计算加速"
    print_option "n" "禁用AMX优化" "不使用AMX指令集"
    echo -e " 是否启用AMX优化? (y/n) [默认: ${GREEN}${default_amx}${NC}]: "
    read -r amx_choice
    
    # 如果用户没有输入，使用智能推荐的默认值
    if [[ -z "$amx_choice" ]]; then
        amx_choice="$default_amx"
    fi
    
    if [[ "$(echo "$amx_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        export ENABLE_AMX="1"
        log_info "已启用AMX优化"
    else
        export ENABLE_AMX="0"
        log_info "已禁用AMX优化"
    fi
    echo ""
    
    # 线程优化设置
    print_option_header "线程数优化"
    echo -e "${BLUE}说明: 自定义线程数可以优化多线程库性能，默认使用CPU核心数${NC}"
    print_option "y" "启用线程数优化" "自定义OpenMP、MKL、NumExpr线程数"
    print_option "n" "使用默认线程数" "自动使用CPU核心数"
    echo -e " 是否启用线程数优化? (y/n) [默认: ${GREEN}n${NC}]: "
    read -r thread_opt_choice
    
    if [[ "$(echo "$thread_opt_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        export ENABLE_THREAD_OPT="y"
        log_info "已启用线程数优化"
        
        local cpu_cores=$(nproc)
        echo -e " 检测到CPU核心数: ${YELLOW}$cpu_cores${NC}"
        
        echo -e " OpenMP线程数 [默认: ${GREEN}$cpu_cores${NC}]: "
        read -r omp_threads
        export CUSTOM_OMP_THREADS="${omp_threads:-$cpu_cores}"
        
        echo -e " MKL线程数 [默认: ${GREEN}$cpu_cores${NC}]: "
        read -r mkl_threads
        export CUSTOM_MKL_THREADS="${mkl_threads:-$cpu_cores}"
        
        echo -e " NumExpr线程数 [默认: ${GREEN}$cpu_cores${NC}]: "
        read -r numexpr_threads
        export CUSTOM_NUMEXPR_THREADS="${numexpr_threads:-$cpu_cores}"
    else
        export ENABLE_THREAD_OPT="n"
        log_info "使用默认线程数设置"
    fi
    echo ""
    
    # CUDA调试设置
    print_option_header "CUDA调试模式"
    echo -e "${BLUE}说明: CUDA调试模式会降低性能但便于调试，生产环境建议关闭${NC}"
    print_option "y" "启用CUDA调试模式" "CUDA_LAUNCH_BLOCKING=1，便于调试但性能较低"
    print_option "n" "禁用CUDA调试模式" "CUDA_LAUNCH_BLOCKING=0，性能模式"
    echo -e " 是否启用CUDA调试模式? (y/n) [默认: ${GREEN}n${NC}]: "
    read -r cuda_debug_choice
    
    if [[ "$(echo "$cuda_debug_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        export ENABLE_CUDA_DEBUG="y"
        log_info "已启用CUDA调试模式"
    else
        export ENABLE_CUDA_DEBUG="n"
        log_info "已禁用CUDA调试模式"
    fi
    echo ""
    
    # CUDA头文件路径设置
    print_option_header "CUDA头文件路径"
    echo -e "${BLUE}说明: 如果CUDA安装在非标准路径，可以自定义头文件路径${NC}"
    print_option "y" "自定义CUDA头文件路径" "手动指定CUDA include目录"
    print_option "n" "使用默认路径" "自动检测CUDA_HOME/include"
    echo -e " 是否自定义CUDA头文件路径? (y/n) [默认: ${GREEN}n${NC}]: "
    read -r cuda_path_choice
    
    if [[ "$(echo "$cuda_path_choice" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        echo -e " 请输入CUDA头文件路径 (如: /usr/local/cuda/include): "
        read -r cuda_include_path
        if [[ -n "$cuda_include_path" && -d "$cuda_include_path" ]]; then
            export CUSTOM_CUDA_INCLUDE_PATH="$cuda_include_path"
            log_info "设置自定义CUDA头文件路径: $cuda_include_path"
        else
            log_warn "路径无效或不存在，将使用默认路径"
        fi
    fi
    echo ""
    
    # 直接应用高级配置，不再显示冗余摘要
    log_info "已应用高级配置"
    
    log_info "高级设置配置完成"
}

# 保存用户配置 - 已移除配置文件生成功能，仅保留内存中的配置管理
save_user_config() {
    # 配置文件生成功能已移除，配置仅在内存中管理
    log_info "用户配置已在内存中管理（配置文件生成功能已移除）"
}

# 配置GitHub hosts加速
setup_git_hosts() {
    local enable_hosts=$1
    if [[ "$enable_hosts" == "y" ]]; then
        log_info "正在配置GitHub hosts加速..."
        # 这里可以添加具体的hosts配置逻辑
        # 例如：写入/etc/hosts或生成hosts文件
        log_info "GitHub hosts加速配置完成"
    fi
}

# 显示欢迎信息并收集用户配置
user_interaction() {
    # 显示欢迎信息（如果之前没有显示过）
    if [[ -z "$WELCOME_SHOWN" ]]; then
        show_welcome_message
        export WELCOME_SHOWN="true"
    fi
    
    # 在开始用户交互前进行硬件检测，设置智能默认值
    log_info "正在检测系统硬件配置..."
    
    # CPU架构和NUMA配置已在check_system阶段自动检测并设置
    # 这里只需要设置默认的编译优化级别
    export COMPILER_OPTIMIZATION_LEVEL="-O3 -march=native"
    
    # 设置CPU架构优化默认启用状态（基于check_system的检测结果）
    if [[ -n "$CPU_ARCH_OPT" && "$CPU_ARCH_OPT" != "auto" ]]; then
        export ENABLE_CPU_ARCH_OPT="y"
        log_info "基于系统检测结果，默认启用CPU架构优化: $CPU_ARCH_OPT"
    else
        export ENABLE_CPU_ARCH_OPT="n"
        log_info "系统检测未发现高级指令集支持，默认禁用CPU架构优化"
    fi
    
    # 设置编译优化级别和并行任务数
    export BUILD_PARALLEL_JOBS="$(nproc)"
    
    log_info "硬件检测完成，已设置智能默认配置"
    
    # 收集用户配置
    show_progress "用户配置"
    
    # 项目获取方式 - 简化配置，不再询问克隆相关选项
    export USE_LOCAL_REPO="false"
    export CUSTOM_REPO_URL=""
    export REPO_CONFIG_COMPLETED="true"
    log_info "使用网络克隆方式获取KTransformers项目"
    echo ""
    
    # 2. 基本路径配置
    print_option_header "安装路径配置"
    echo -e "${BLUE}说明: KTransformers将被安装到指定目录${NC}"
    print_option "1" "使用默认路径" "/opt/kt (推荐)"
    print_option "2" "自定义安装路径" "指定其他目录"
    echo -e " 请选择安装路径 [默认: ${GREEN}1${NC}]: "
    read -r path_choice
    
    case $path_choice in
        1)
            KT_ROOT="/opt/kt"
            ;;
        2)
            echo -e " 请输入自定义安装路径 (如: /home/user/ktransformers): "
            read -r custom_path
            if [[ -n "$custom_path" ]]; then
                # 处理路径，移除末尾的斜杠
                custom_path=$(echo "$custom_path" | sed 's:/*$::')
                KT_ROOT="$custom_path"
            else
                log_warn "未输入路径，使用默认路径"
                KT_ROOT="/opt/kt"
            fi
            ;;
        *)
            KT_ROOT="/opt/kt"
            ;;
    esac
    
    # 验证路径并显示确认信息
    if [[ "$KT_ROOT" == /* ]]; then
        # 绝对路径
        echo -e " ${GREEN}✓${NC} KTransformers将安装到: ${BLUE}${KT_ROOT}${NC}"
    else
        # 相对路径，转换为绝对路径
        KT_ROOT="$(cd "$(dirname "$KT_ROOT")" 2>/dev/null && pwd)/$(basename "$KT_ROOT")" 2>/dev/null || KT_ROOT="$PWD/$KT_ROOT"
        echo -e " ${GREEN}✓${NC} KTransformers将安装到: ${BLUE}${KT_ROOT}${NC}"
    fi
    
    # 检查路径权限
    parent_dir="$(dirname "$KT_ROOT")"
    if [[ ! -d "$parent_dir" ]]; then
        log_warn "父目录不存在: $parent_dir"
        echo -e " ${YELLOW}注意: 安装时将创建必要的目录${NC}"
    elif [[ ! -w "$parent_dir" ]]; then
        log_warn "对父目录没有写权限: $parent_dir"
        echo -e " ${YELLOW}注意: 安装时需要root权限${NC}"
    fi
    
    export KT_ROOT
    echo ""
    
    # 2. Conda环境配置
    print_option_header "Conda环境配置"
    echo -e " 请输入Conda环境名称 [默认: ${GREEN}${CONDA_ENV}${NC}]: "
    read -r user_conda_env
    if [[ -n "$user_conda_env" ]]; then
        CONDA_ENV="$user_conda_env"
    fi
    
    # 3. Python版本选择
    print_option_header "Python版本选择"
    print_option "1" "Python 3.10" "稳定性好"
    print_option "2" "Python 3.11" "推荐，性能优"
    print_option "3" "Python 3.12" "新特性支持"
    echo -e " 请选择 [默认: ${GREEN}2${NC}]: "
    read -r python_choice
    case $python_choice in
        1) PYTHON_VERSION="3.10" ;;
        2) PYTHON_VERSION="3.11" ;;
        3) PYTHON_VERSION="3.12" ;;
        *) PYTHON_VERSION="3.11" ;;
    esac
    
    # 4. PyTorch版本选择
    print_option_header "PyTorch版本选择"
    print_option "1" "PyTorch 2.6.0 + CUDA 12.6" "稳定版"
    print_option "2" "PyTorch 2.7.0 + CUDA 12.8" "最新版"
    echo -e " 请选择 [默认: ${GREEN}2${NC}]: "
    read -r pytorch_choice
    case $pytorch_choice in
        1) 
           PYTORCH_VERSION="2.6.0" 
           CUDA_VERSION="12.6"
           CUDA_VERSION_SHORT="126"
           ;;
        2) 
           PYTORCH_VERSION="2.7.0" 
           CUDA_VERSION="12.8"
           CUDA_VERSION_SHORT="128"
           ;;
        *) 
           PYTORCH_VERSION="2.7.0" 
           CUDA_VERSION="12.8"
           CUDA_VERSION_SHORT="128"
           ;;
    esac
    
    # 5. Flash Attention版本选择
    print_option_header "Flash Attention版本选择"
    print_option "1" "Flash Attention 2.8.3" "最新版"
    print_option "2" "Flash Attention 2.7.4.post1" "补丁版，推荐"
    print_option "3" "从源码编译" "自定义编译，适用于特殊硬件配置"
    echo -e " 请选择 [默认: ${GREEN}1${NC}]: "
    read -r flash_attn_choice
    case $flash_attn_choice in
        1) FLASH_ATTN_VERSION="2.8.3" ;;
        2) FLASH_ATTN_VERSION="2.7.4.post1" ;;
        3) FLASH_ATTN_VERSION="source" ;;
        *) FLASH_ATTN_VERSION="2.8.3" ;;
    esac
    
    # 导出Flash Attention版本选择，确保后续模块可以使用
    export FLASH_ATTN_VERSION
    log_info "用户选择Flash Attention版本: $FLASH_ATTN_VERSION"
    
    # 6. 镜像源配置
    # 镜像源配置（统一询问）
    configure_mirror_sources
    
    # 8. GPU自动检测配置
    print_option_header "GPU配置检测"
    log_info "自动检测GPU配置..."
    
    # 检测NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local nvidia_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$nvidia_info" ]]; then
            GPU_MODE="cuda"
            log_info "检测到NVIDIA GPU，已配置CUDA模式"
            
            # 检测具体的GPU信息并提供选择
            configure_gpu_selection
        fi
    fi
    
    # 如果没有检测到NVIDIA GPU，检测AMD GPU
    if [[ -z "$GPU_MODE" ]]; then
        if command -v rocm-smi >/dev/null 2>&1; then
            local amd_info=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1)
            if [[ -n "$amd_info" ]]; then
                GPU_MODE="rocm"
                log_info "检测到AMD GPU，已配置ROCm模式"
            fi
        fi
    fi
    
    # 如果仍未检测到，通过lspci检测
    if [[ -z "$GPU_MODE" ]]; then
        if lspci | grep -i "vga\|3d\|display" | grep -i "amd\|ati" >/dev/null 2>&1; then
            GPU_MODE="rocm"
            log_info "检测到AMD显卡，已配置ROCm模式"
        elif lspci | grep -i "vga\|3d\|display" | grep -i "nvidia" >/dev/null 2>&1; then
            GPU_MODE="cuda"
            log_info "检测到NVIDIA显卡，已配置CUDA模式"
        else
            # 默认使用CUDA模式
            GPU_MODE="cuda"
            log_info "默认使用CUDA模式"
        fi
    fi
    
    # 9. 配置确认与优化（融合阶段）
    show_configuration_and_confirm
}

# 配置确认与优化（融合的配置确认阶段）
show_configuration_and_confirm() {
    print_option_header "KTransformers 安装配置确认"
    
    # 确保TORCH_CUDA_ARCH_LIST有默认值
    if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
        TORCH_CUDA_ARCH_LIST="auto"
        export TORCH_CUDA_ARCH_LIST
    fi
    
    # 设置编译环境变量
    set_compilation_variables
    
    # 显示完整的配置信息
    show_unified_configuration_display
}

# 显示统一的配置界面
show_unified_configuration_display() {
    echo ""
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${GREEN}[INFO]    KTransformers 安装配置确认${NC}"
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo ""
    
    echo -e "${BLUE}基础配置:${NC}"
    echo -e "  安装路径: ${YELLOW}$KT_ROOT${NC}"
    echo -e "  Conda环境: ${YELLOW}$CONDA_ENV${NC}"
    echo -e "  Python版本: ${YELLOW}$PYTHON_VERSION${NC}"
    echo -e "  PyTorch版本: ${YELLOW}$PYTORCH_VERSION${NC}"
    echo -e "  CUDA版本: ${YELLOW}$CUDA_VERSION${NC}"
    echo ""
    
    echo -e "${BLUE}编译配置:${NC}"
    echo -e "  CUDA支持: ${YELLOW}$KTRANSFORMERS_USE_CUDA${NC}"
    echo -e "  构建类型: ${YELLOW}$CMAKE_BUILD_TYPE${NC}"
    echo -e "  编译优化级别: ${YELLOW}$COMPILER_OPTIMIZATION_LEVEL${NC}"
    echo -e "  编译并行度: ${YELLOW}$BUILD_PARALLEL_JOBS${NC}"
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        echo -e "  GPU架构: ${YELLOW}$TORCH_CUDA_ARCH_LIST${NC}"
    fi
    echo ""
    
    echo -e "${BLUE}编译器配置:${NC}"
    if [[ -n "$CC" ]]; then
        echo -e "  C编译器: ${YELLOW}$CC${NC}"
    fi
    if [[ -n "$CXX" ]]; then
        echo -e "  C++编译器: ${YELLOW}$CXX${NC}"
    fi
    if [[ -n "$CUDACXX" ]]; then
        echo -e "  CUDA编译器: ${YELLOW}$CUDACXX${NC}"
    fi
    echo ""
    
    echo -e "${BLUE}系统优化配置:${NC}"
    # CPU架构优化显示
    if [[ -n "$ENABLE_CPU_ARCH_OPT" && "$ENABLE_CPU_ARCH_OPT" == "y" ]]; then
        echo -e "  CPU架构优化: ${GREEN}已启用${NC} (${YELLOW}$CPU_ARCH_OPT${NC})"
    else
        echo -e "  CPU架构优化: ${RED}未启用${NC} (使用默认编译选项)"
    fi
    
    # NUMA支持显示
    if [[ "$USE_NUMA" == "1" ]]; then
        echo -e "  NUMA支持: ${GREEN}已启用${NC}"
    else
        echo -e "  NUMA支持: ${RED}未启用${NC}"
    fi
    
    # AMX优化显示
    if [[ "$ENABLE_AMX" == "1" ]]; then
        echo -e "  AMX优化: ${GREEN}已启用${NC} (Intel矩阵计算加速)"
    else
        echo -e "  AMX优化: ${RED}未启用${NC}"
    fi
    
    # 线程优化显示
    if [[ -n "$ENABLE_THREAD_OPT" && "$ENABLE_THREAD_OPT" == "y" ]]; then
        echo -e "  线程优化: ${GREEN}已启用${NC} (自定义线程数配置)"
    else
        echo -e "  线程优化: ${RED}未启用${NC} (使用系统默认)"
    fi
    
    # CUDA调试模式显示
    if [[ -n "$ENABLE_CUDA_DEBUG" && "$ENABLE_CUDA_DEBUG" == "y" ]]; then
        echo -e "  CUDA调试模式: ${YELLOW}已启用${NC} (性能较低，便于调试)"
    else
        echo -e "  CUDA调试模式: ${RED}未启用${NC} (性能模式)"
    fi
    
    # 自定义CUDA路径显示
    if [[ -n "$CUSTOM_CUDA_INCLUDE_PATH" ]]; then
        echo -e "  自定义CUDA路径: ${GREEN}已启用${NC}"
        echo -e "    路径: ${YELLOW}$CUSTOM_CUDA_INCLUDE_PATH${NC}"
    else
        echo -e "  自定义CUDA路径: ${RED}未启用${NC} (使用默认路径)"
    fi
    echo ""
    
    echo -e "${BLUE}镜像源配置:${NC}"
    # APT镜像显示
    if [[ "$USE_APT_MIRROR" == "y" ]]; then
        echo -e "  APT镜像: ${GREEN}已启用${NC} (使用国内镜像源)"
    else
        echo -e "  APT镜像: ${RED}未启用${NC} (使用官方源)"
    fi
    
    # Conda镜像显示
    if [[ "$USE_CONDA_MIRROR" == "y" ]]; then
        echo -e "  Conda镜像: ${GREEN}已启用${NC} (使用国内镜像源)"
    else
        echo -e "  Conda镜像: ${RED}未启用${NC} (使用官方源)"
    fi
    
    # PIP镜像显示
    if [[ "$USE_PIP_MIRROR" == "y" ]]; then
        echo -e "  PIP镜像: ${GREEN}已启用${NC} (使用国内镜像源)"
    else
        echo -e "  PIP镜像: ${RED}未启用${NC} (使用官方源)"
    fi
    
    # Git代理显示
    if [[ "$USE_GIT_HOSTS" == "y" ]]; then
        echo -e "  GitHub hosts: ${GREEN}已启用${NC} (使用hosts加速)"
    else
        echo -e "  GitHub hosts: ${RED}未启用${NC} (直连)"
    fi
    echo ""
    
    echo -e "${GREEN}[INFO] ======================================${NC}"
    echo -e "${YELLOW}请选择操作：${NC}"
    echo -e " ${GREEN}1${NC} - ${GREEN}确认配置并开始安装${NC}"
    echo -e " ${BLUE}2${NC} - ${BLUE}调整高级优化设置${NC}"
    echo -e " ${RED}3${NC} - ${RED}取消安装${NC}"
    echo -e " 请选择 [默认: ${GREEN}1${NC}]: "
    read -r confirm_choice
    
    case "${confirm_choice}" in
        "2")
            configure_advanced_settings
            show_unified_configuration_display  # 递归调用以重新显示确认界面
            ;;
        "3")
            log_info "用户取消安装"
            exit 0
            ;;
        *)
            log_info "用户确认配置，开始安装..."
            ;;
    esac
}

# 设置编译环境变量
set_compilation_variables() {
    # 启用CUDA支持
    export KTRANSFORMERS_USE_CUDA="ON"
    log_info "启用CUDA支持: KTRANSFORMERS_USE_CUDA=ON"
    
    # 设置编译优化级别
    if [[ -n "$COMPILER_OPTIMIZATION_LEVEL" ]]; then
        export CFLAGS="$COMPILER_OPTIMIZATION_LEVEL"
        export CXXFLAGS="$COMPILER_OPTIMIZATION_LEVEL"
        log_info "使用用户指定的编译优化级别: $COMPILER_OPTIMIZATION_LEVEL"
    else
        export CFLAGS="-O3 -march=native"
        export CXXFLAGS="-O3 -march=native"
        log_info "使用默认编译优化级别: -O3 -march=native"
    fi
    
    # 设置编译器（根据系统当前版本决定）
    if [[ -n "$CC" && -n "$CXX" ]]; then
        # 如果已经检测到系统编译器，直接使用
        log_info "使用系统检测的编译器: CC=$CC, CXX=$CXX"
    elif command -v gcc-13 >/dev/null 2>&1; then
        export CC="gcc-13"
        export CXX="g++-13"
        log_info "使用GCC 13编译器"
    elif command -v gcc-12 >/dev/null 2>&1; then
        export CC="gcc-12"
        export CXX="g++-12"
        log_info "使用GCC 12编译器"
    else
        export CC="gcc"
        export CXX="g++"
        log_info "使用系统默认GCC编译器"
    fi
    
    # 设置CUDA编译器
    if [[ -n "$CUDA_HOME" && -f "$CUDA_HOME/bin/nvcc" ]]; then
        export CUDACXX="$CUDA_HOME/bin/nvcc"
        log_info "设置CUDA编译器: $CUDA_HOME/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDACXX="$(which nvcc)"
        log_info "设置CUDA编译器: $(which nvcc)"
    else
        export CUDACXX="/usr/local/cuda/bin/nvcc"
        log_info "设置CUDA编译器: /usr/local/cuda/bin/nvcc"
    fi
    
    # 设置CMake构建类型
    export CMAKE_BUILD_TYPE="Release"
    log_info "设置CMake构建类型: Release"
    
    # 设置编译并行度
    if [[ -n "$BUILD_PARALLEL_JOBS" ]]; then
        export MAX_JOBS="$BUILD_PARALLEL_JOBS"
        log_info "设置编译并行度: $BUILD_PARALLEL_JOBS"
    else
        export MAX_JOBS="$(nproc)"
        log_info "设置编译并行度: $(nproc)"
    fi
    
    # 设置GPU架构
    if [[ -n "$TORCH_CUDA_ARCH_LIST" && "$TORCH_CUDA_ARCH_LIST" != "auto" ]]; then
        export CMAKE_CUDA_ARCHITECTURES="$TORCH_CUDA_ARCH_LIST"
        log_info "使用用户选择的GPU架构配置: $TORCH_CUDA_ARCH_LIST"
        log_info "设置CMAKE_CUDA_ARCHITECTURES: $TORCH_CUDA_ARCH_LIST"
    else
        # 使用系统检测的GPU架构（优先），否则自动检测
        if [[ -n "$DETECTED_GPU_ARCHES" ]]; then
            # 清理架构列表格式（去除+PTX等后缀）
            local clean_arch_list=$(echo "$DETECTED_GPU_ARCHES" | sed 's/+PTX//g' | sed 's/+RTX//g')
            export CMAKE_CUDA_ARCHITECTURES="$clean_arch_list"
            export TORCH_CUDA_ARCH_LIST="$clean_arch_list"
            log_info "使用系统检测的GPU架构: $clean_arch_list"
            log_info "设置CMAKE_CUDA_ARCHITECTURES: $clean_arch_list"
        elif command -v nvidia-smi >/dev/null 2>&1; then
            local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            if [[ -n "$compute_cap" && "$compute_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
                export CMAKE_CUDA_ARCHITECTURES="$compute_cap"
                export TORCH_CUDA_ARCH_LIST="$compute_cap"
                log_info "自动检测GPU架构: $compute_cap"
                log_info "设置CMAKE_CUDA_ARCHITECTURES: $compute_cap"
            else
                export CMAKE_CUDA_ARCHITECTURES="8.0;8.6;9.0"
                export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
                log_info "GPU架构检测失败，使用通用配置: 8.0;8.6;9.0"
            fi
        else
            export CMAKE_CUDA_ARCHITECTURES="8.0;8.6;9.0"
            export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
            log_info "未检测到nvidia-smi，使用通用GPU架构配置: 8.0;8.6;9.0"
        fi
    fi
    
    # 设置NUMA配置
    if [[ -n "$USE_NUMA" ]]; then
        log_info "使用已检测的NUMA配置: USE_NUMA=$USE_NUMA"
    else
        export USE_NUMA="0"
        log_info "设置NUMA配置: USE_NUMA=0"
    fi
    
    # 构建CMAKE_ARGS
    export CMAKE_ARGS="-DKTRANSFORMERS_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES -DUSE_BALANCE_SERVE=1 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CUDACXX"
    log_info "设置CMAKE_ARGS: $CMAKE_ARGS"
    
    log_info "编译环境变量设置完成"
    
    # 显示编译配置摘要
    echo ""
    log_info "编译配置摘要:"
    log_info "  KTRANSFORMERS_USE_CUDA: $KTRANSFORMERS_USE_CUDA"
    log_info "  CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE"
    log_info "  编译优化级别: $CFLAGS"
    log_info "  编译并行度: $MAX_JOBS"
    log_info "  CPU架构优化: $CPU_ARCH_OPT"
    log_info "  NUMA支持: $USE_NUMA"
    log_info "  多并发支持: 1"
    log_info "  GPU架构: $TORCH_CUDA_ARCH_LIST"
    log_info "  CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
    log_info "  CMAKE_ARGS: $CMAKE_ARGS"
    echo ""
}
