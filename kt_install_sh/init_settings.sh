#!/bin/bash

# KTransformers 初始化设置模块
# 包含全局变量、日志功能和基础函数

# 脚本版本
SCRIPT_VERSION="1.0.0"

# 日志和配置目录设置（仅适用于Linux系统）
LOG_DIR="$HOME/.ktransformers/logs"
CONFIG_DIR="$HOME/.ktransformers/config"

# 创建日志目录，配置目录创建功能已移除
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    echo "警告: 无法在 $LOG_DIR 创建日志目录，尝试使用当前目录"
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR" || {
        echo "警告: 无法创建日志目录，使用临时目录"
        LOG_DIR="/tmp/ktransformers_logs_$$"
        mkdir -p "$LOG_DIR"
    }
fi

# 配置目录功能已移除，CONFIG_DIR仅用于向后兼容
CONFIG_DIR="/tmp/ktransformers_config_disabled"

LOG_FILE="$LOG_DIR/kt_install_$(date +%Y%m%d_%H%M%S).log"

# 创建日志文件，增强错误处理
if ! touch "$LOG_FILE" 2>/dev/null; then
    echo "警告: 无法创建日志文件 $LOG_FILE"
    # 尝试在临时目录创建日志文件
    LOG_FILE="/tmp/kt_install_$(date +%Y%m%d_%H%M%S)_$$.log"
    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "警告: 无法创建临时日志文件，使用标准输出"
        LOG_FILE="/dev/stdout"
    else
        echo "日志文件已创建: $LOG_FILE"
    fi
else
    echo "日志文件已创建: $LOG_FILE"
fi

# GitHub hosts配置标志（移除旧的Git代理配置）
GIT_HOSTS_CONFIGURED=false  # hosts配置状态
USE_GIT_HOSTS="n"  # 是否启用GitHub hosts配置

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 恢复默认颜色

# 默认配置变量
KT_ROOT="/opt/kt"
CONDA_ENV="kt311"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.6.0"
CUDA_VERSION="12.8"
CUDA_VERSION_SHORT="128"
USE_MIRROR="n"
USE_NUMA="0"
GPU_MODE="cuda"
FLASH_ATTN_VERSION="2.8.2"
FORCE_CUDA_REINSTALL="false"        # 是否强制重新安装CUDA
USE_BALANCE_SERVE="1"               # 是否启用负载均衡服务（默认启用多并发）

# KTransformers官方推荐的环境变量
CMAKE_ARGS=""                       # CMake编译参数
NVTE_CUDA_INCLUDE_PATH=""          # CUDA头文件路径（如果CUDA不在标准路径）
KTRANSFORMERS_USE_CUDA="ON"        # 启用CUDA支持
KTRANSFORMERS_USE_NUMA="0"         # NUMA支持（与USE_NUMA保持一致）
KTRANSFORMERS_USE_BALANCE_SERVE="1" # 负载均衡服务（与USE_BALANCE_SERVE保持一致）

# 编译优化相关环境变量
CC=""                              # C编译器路径
CXX=""                             # C++编译器路径
CUDACXX=""                         # CUDA编译器路径
CMAKE_BUILD_TYPE="Release"         # CMake构建类型
CMAKE_CUDA_ARCHITECTURES=""        # CUDA架构（与TORCH_CUDA_ARCH_LIST对应）

# 性能优化环境变量
OMP_NUM_THREADS=""                 # OpenMP线程数
MKL_NUM_THREADS=""                 # MKL线程数
NUMEXPR_NUM_THREADS=""             # NumExpr线程数
CUDA_LAUNCH_BLOCKING="0"           # CUDA启动阻塞模式（调试用）

# Intel CPU优化环境变量（仅适用于Intel CPU）
DNNL_MAX_CPU_ISA=""               # Intel DNNL最大指令集
MKL_ENABLE_INSTRUCTIONS=""        # MKL启用的指令集
ONEDNN_DEFAULT_FPMATH_MODE=""     # OneDNN浮点数学模式

# 日志级别配置
# 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=FATAL
LOG_LEVEL=${LOG_LEVEL:-1}  # 默认INFO级别
LOG_TO_FILE=${LOG_TO_FILE:-true}  # 默认写入文件
LOG_TO_CONSOLE=${LOG_TO_CONSOLE:-true}  # 默认输出到控制台

# 日志级别映射函数
get_log_level_value() {
    case "$1" in
        "DEBUG") echo 0 ;;
        "INFO") echo 1 ;;
        "WARN") echo 2 ;;
        "ERROR") echo 3 ;;
        "FATAL") echo 4 ;;
        *) echo 1 ;;  # 默认INFO级别
    esac
}

# 增强的日志函数
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local caller_info=""

    # 获取调用者信息 (函数名和行号)
    if [[ "${level}" == "DEBUG" ]] || [[ "${level}" == "ERROR" ]] || [[ "${level}" == "FATAL" ]]; then
        local caller_line="unknown"
        local caller_func="main"
        if [[ -n "${BASH_LINENO[1]}" ]]; then
            caller_line="${BASH_LINENO[1]}"
        fi
        if [[ -n "${FUNCNAME[2]}" ]]; then
            caller_func="${FUNCNAME[2]}"
        fi
        caller_info=" [${caller_func}:${caller_line}]"
    fi

    # 日志级别数值转换
    local current_level=1  # 默认INFO级别
    case $level in
        "DEBUG") current_level=0 ;;
        "INFO") current_level=1 ;;
        "WARN") current_level=2 ;;
        "ERROR") current_level=3 ;;
        "FATAL") current_level=4 ;;
    esac

    # 检查日志级别
    if [[ $current_level -lt $LOG_LEVEL ]]; then
        # 只写入文件，不输出到控制台
        if [[ "$LOG_TO_FILE" == "true" ]] && [[ -w "$LOG_FILE" ]]; then
            echo "[$timestamp] [$level]$caller_info $message" >> "$LOG_FILE" 2>/dev/null || true
        fi
        return 0
    fi

    # 输出到终端
    if [[ "$LOG_TO_CONSOLE" == "true" ]]; then
        case $level in
            "DEBUG") echo -e "${BLUE}[DEBUG]$caller_info $message${NC}" ;;
            "INFO") echo -e "${GREEN}[INFO] $message${NC}" ;;
            "WARN") echo -e "${YELLOW}[WARN] $message${NC}" ;;
            "ERROR") echo -e "${RED}[ERROR]$caller_info $message${NC}" ;;
            "FATAL")
                echo -e "${RED}[FATAL]$caller_info $message${NC}"
                # FATAL级别同时写入stderr
                echo -e "[FATAL]$caller_info $message" >&2
                ;;
        esac
    fi

    # 写入日志文件
    if [[ "$LOG_TO_FILE" == "true" ]] && [[ -w "$LOG_FILE" ]]; then
        echo "[$timestamp] [$level]$caller_info $message" >> "$LOG_FILE" 2>/dev/null || {
            # 如果写入失败，尝试输出到stderr
            echo "[LOG_ERROR] 无法写入日志文件: $LOG_FILE" >&2
        }
    fi

    # FATAL级别退出程序
    if [[ "$level" == "FATAL" ]]; then
        cleanup_on_exit
        exit 1
    fi
}

# 便捷的日志函数
log_debug() { log_message "DEBUG" "$1"; }
log_info() { log_message "INFO" "$1"; }
log_warn() { log_message "WARN" "$1"; }
log_error() { log_message "ERROR" "$1"; }
log_fatal() { log_message "FATAL" "$1"; }

# 统一的错误处理函数
handle_error() {
    local exit_code=$1
    local error_msg="$2"
    local is_fatal=${3:-false}

    if [[ $exit_code -ne 0 ]]; then
        if [[ "$is_fatal" == "true" ]]; then
            log_fatal "$error_msg (退出码: $exit_code)"
        else
            log_error "$error_msg (退出码: $exit_code)"
        fi
        return $exit_code
    fi
    return 0
}



# 设置日志级别的函数
set_log_level() {
    local level_name="$1"
    local level_value=$(get_log_level_value "$level_name")

    if [[ "$level_name" == "DEBUG" || "$level_name" == "INFO" || "$level_name" == "WARN" || "$level_name" == "ERROR" || "$level_name" == "FATAL" ]]; then
        LOG_LEVEL=$level_value
        log_info "日志级别设置为: $level_name"
    else
        log_error "无效的日志级别: $level_name，支持的级别: DEBUG, INFO, WARN, ERROR, FATAL"
        return 1
    fi
}

# Git代理配置函数
test_proxy_connection() {
    local proxy_url="$1"
    local timeout=10

    # 提取域名进行连接测试
    local domain=$(echo "$proxy_url" | sed 's|https\?://||' | cut -d'/' -f1)

    if timeout "$timeout" curl -s --head "$proxy_url" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 配置GitHub hosts加速 - 使用GitCDN动态获取最新hosts
setup_git_hosts() {
    local use_git_hosts="$1"

    # 检查是否已经配置过hosts（内存状态检查）
    if [[ "${GIT_HOSTS_CONFIGURED:-false}" == true ]]; then
        log_message "INFO" "GitHub hosts配置已存在（内存状态），跳过重复配置"
        return 0
    fi

    # 检查hosts文件是否已存在我们的配置（文件系统检查）
    local hosts_file=""
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        hosts_file="/c/Windows/System32/drivers/etc/hosts"
    else
        hosts_file="/etc/hosts"
    fi

    local github_hosts_marker="# GitHub Hosts Start - Added by KTransformers Installer"
    local github_hosts_end_marker="# GitHub Hosts End"
    
    # 如果hosts文件存在，检查是否已有我们的配置
    if [[ -f "$hosts_file" ]]; then
        if grep -q "$github_hosts_marker" "$hosts_file" && grep -q "$github_hosts_end_marker" "$hosts_file"; then
            log_message "INFO" "检测到已存在的KTransformers GitHub hosts配置，将清理后重新配置"
            
            # 清理旧的配置（包括开始和结束标记之间的所有内容）
            sed -i "/$github_hosts_marker/,/$github_hosts_end_marker/d" "$hosts_file" 2>/dev/null || true
            log_message "INFO" "已清理旧的KTransformers GitHub hosts配置"
        fi
    fi

    if [[ "$(echo "$use_git_hosts" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        log_message "INFO" "配置GitHub hosts加速访问..."

        # 获取最新的GitHub hosts配置
        local hosts_content=""
        local gitcdn_hosts_url="https://hosts.gitcdn.top/hosts.txt"
        
        log_message "INFO" "从GitCDN获取最新的GitHub hosts配置..."
        log_message "INFO" "使用URL: $gitcdn_hosts_url"
        
        # 从GitCDN获取hosts配置
        hosts_content=$(curl -s --connect-timeout 10 --max-time 30 "$gitcdn_hosts_url" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[[:space:]]+(github|api\.github|raw\.githubusercontent|githubassets|githubusercontent)' | head -30)
        
        if [[ -n "$hosts_content" ]]; then
            log_message "INFO" "成功从GitCDN获取hosts配置"
            log_message "INFO" "获取到的hosts条目数量: $(echo "$hosts_content" | wc -l)"
        else
            log_message "WARN" "从GitCDN获取hosts配置失败，使用备用hosts配置"
            # 备用hosts配置
            hosts_content="140.82.113.26                 alive.github.com
20.205.243.168                api.github.com
185.199.111.133               avatars.githubusercontent.com
140.82.113.4                  github.com
140.82.112.21                 central.github.com
185.199.111.133               raw.githubusercontent.com
185.199.111.133               user-images.githubusercontent.com
185.199.111.133               avatars0.githubusercontent.com
185.199.111.133               avatars1.githubusercontent.com
185.199.111.133               avatars2.githubusercontent.com"
        fi

        # 备份原始hosts文件（只有在需要修改时才备份）
        if [[ -f "$hosts_file" ]]; then
            # 检查是否需要备份（如果文件将被修改）
            local needs_backup=false
            
            # 如果hosts文件不为空且没有我们的标记，或者我们要添加新配置
            if [[ -s "$hosts_file" ]]; then
                if ! grep -q "$github_hosts_marker" "$hosts_file"; then
                    needs_backup=true
                fi
            else
                # 空文件也需要备份
                needs_backup=true
            fi
            
            if [[ "$needs_backup" == true ]]; then
                cp "$hosts_file" "${hosts_file}.bak.$(date +%Y%m%d_%H%M%S)"
                log_message "INFO" "已备份原始hosts文件到 ${hosts_file}.bak.$(date +%Y%m%d_%H%M%S)"
            fi
        fi

        # 检查并清理可能存在的冲突条目（其他程序添加的GitHub相关条目）
        log_message "INFO" "检查并清理可能存在的冲突GitHub hosts条目..."
        
        # 获取我们要添加的域名列表
        local new_domains=$(echo "$hosts_content" | awk '{print $2}' | sort -u)
        
        # 清理hosts文件中这些域名的现有条目（不包括我们的标记块）
        if [[ -f "$hosts_file" ]] && [[ -n "$new_domains" ]]; then
            while IFS= read -r domain; do
                if [[ -n "$domain" ]]; then
                    # 清理指定域名的条目，但保留我们的配置块内的内容
                    sed -i "/^[^#]*[[:space:]]*$domain[[:space:]]*$/d" "$hosts_file" 2>/dev/null || true
                fi
            done <<< "$new_domains"
            log_message "INFO" "已清理冲突的GitHub hosts条目"
        fi

        # 添加新的GitHub hosts条目
        echo -e "\n$github_hosts_marker" >> "$hosts_file"
        echo "$hosts_content" >> "$hosts_file"
        echo -e "$github_hosts_end_marker\n" >> "$hosts_file"

        log_message "INFO" "GitHub hosts配置已更新"
        log_message "INFO" "添加的hosts配置："
        echo "$hosts_content" | while read -r line; do
            log_message "INFO" "  $line"
        done

        # 刷新DNS缓存
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            ipconfig /flushdns 2>/dev/null || true
            log_message "INFO" "Windows DNS缓存已刷新"
        elif [[ "$OSTYPE" == "darwin" ]]; then
            sudo dscacheutil -flushcache 2>/dev/null || true
            sudo killall -HUP mDNSResponder 2>/dev/null || true
            log_message "INFO" "macOS DNS缓存已刷新"
        else
            sudo systemctl restart systemd-resolved 2>/dev/null || true
            sudo service nscd restart 2>/dev/null || true
            log_message "INFO" "Linux DNS缓存已刷新"
        fi

        # 测试GitHub连接
        log_message "INFO" "测试GitHub连接..."
        if curl -s --connect-timeout 10 --max-time 15 "https://github.com" > /dev/null; then
            log_message "SUCCESS" "GitHub连接测试成功，hosts配置生效"
        else
            log_message "WARN" "GitHub连接测试失败，可能需要重启网络或等待DNS生效"
        fi

        # 设置环境变量和状态标记
        export USE_GIT_HOSTS="y"
        GIT_HOSTS_CONFIGURED=true
        
        log_message "SUCCESS" "GitHub hosts配置完成"
        
    else
        log_message "INFO" "跳过GitHub hosts配置"
    fi
}

# 清理GitHub hosts配置（hosts配置通常不需要清理，保留备份即可）
cleanup_git_hosts() {
    log_message "INFO" "GitHub hosts配置已备份，无需清理"
    # hosts配置通过备份文件管理，不自动清理
}

# 清理函数 - 在脚本退出时执行
cleanup_on_exit() {
    local exit_code=$?
    log_debug "执行清理操作，退出码: $exit_code"

    # 清理GitHub hosts配置
    cleanup_git_hosts

    # 清理临时文件
    if [[ -n "$TEMP_DIR" ]] && [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR" 2>/dev/null || true
        log_debug "已清理临时目录: $TEMP_DIR"
    fi

    # 恢复原始工作目录
    if [[ -n "$ORIGINAL_PWD" ]] && [[ -d "$ORIGINAL_PWD" ]]; then
        cd "$ORIGINAL_PWD" 2>/dev/null || true
    fi

    # 如果是异常退出，记录错误信息
    if [[ $exit_code -ne 0 ]]; then
        log_error "脚本异常退出，退出码: $exit_code"
        log_info "详细日志请查看: $LOG_FILE"
    fi
}

# 错误处理函数
handle_error() {
    local exit_code=$1
    local line_number=$2
    local command="$3"

    log_error "命令执行失败 [行号: $line_number, 退出码: $exit_code]"
    log_error "失败的命令: $command"

    # 提供错误恢复建议
    case $exit_code in
        1) log_warn "建议: 检查命令语法或权限问题" ;;
        2) log_warn "建议: 检查文件或目录是否存在" ;;
        126) log_warn "建议: 检查文件执行权限" ;;
        127) log_warn "建议: 检查命令是否存在或PATH设置" ;;
        130) log_warn "用户中断操作" ;;
        *) log_warn "建议: 查看详细日志分析具体原因" ;;
    esac

    return $exit_code
}





# 显示进度
show_progress() {
    local step_name=$1
    echo -e "\n${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ 步骤: $step_name${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
}

# 显示完成信息
show_completion_message() {
    echo -e "\n${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ 安装完成                                        │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
    echo -e "\n${YELLOW}安装已完成，请根据需要手动设置环境变量${NC}"
    echo -e "如需使用环境变量，请执行: source /etc/profile\n"
}

# 显示欢迎信息
show_welcome_message() {
    echo -e "\n${GREEN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ KTransformers 安装程序 v${SCRIPT_VERSION}                  │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
    echo -e "\n${BLUE}此脚本将自动安装KTransformers及其依赖项${NC}"
    echo -e "${BLUE}包括: CUDA, PyTorch, Conda环境和相关工具${NC}\n"
}

# 打印选项标题
print_option_header() {
    local header=$1
    echo -e "\n${BLUE}$header:${NC}"
}

# 打印选项
print_option() {
    local option=$1
    local description=$2
    local note=${3:-""}

    echo -e " ${GREEN}$option${NC}. $description ${YELLOW}${note}${NC}"
}

# 保存配置状态 - 这是全局函数
save_config_state() {
    # 导出所有配置变量
    export_all_variables

    log_message "INFO" "配置状态已设置完成"
}

# 加载配置状态 - 已移除配置文件依赖，仅使用内存配置
load_config_state() {
    # 配置文件功能已移除，配置仅在内存中管理
    log_message "INFO" "配置文件功能已移除，使用默认配置"
    return 1  # 始终返回1以使用默认配置
}

# 设置默认配置
set_default_config() {
    # 基础配置
    SCRIPT_VERSION="1.0.1"
    CONDA_ENV="kt"
    PYTHON_VERSION="3.11"
    PYTORCH_VERSION="2.6.0"
    CUDA_VERSION="12.8"
    CUDA_VERSION_SHORT="128"
    # 如果FLASH_ATTN_VERSION已经设置，不要覆盖它
    if [[ -z "${FLASH_ATTN_VERSION}" ]]; then
        FLASH_ATTN_VERSION="2.8.3"
    fi

    # 镜像源配置
    USE_APT_MIRROR="n"        # apt源配置
    USE_CONDA_MIRROR="n"      # conda源配置
    USE_PIP_MIRROR="n"        # GitHub hosts配置
    USE_GIT_HOSTS="n"         # GitHub hosts配置
    GIT_HOSTS_CONFIGURED=false # hosts配置状态

    # GPU配置
    GPU_MODE="cuda"
    FORCE_CUDA_REINSTALL="false"

    # 高级配置
    CPU_ARCH_OPT="auto"
    USE_NUMA="0"

    # 仓库配置
    USE_LOCAL_REPO="false"
    LOCAL_REPO_PATH=""

    # Web UI配置

    # 配置目录创建功能已移除
    CONFIG_DIR="/tmp/ktransformers_config_disabled"
}

# 初始化设置
init_settings() {
    # 设置颜色代码
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[0;37m'
    NC='\033[0m' # No Color

    # 设置默认配置
    set_default_config

    # 加载已保存的配置（如果存在）
    load_config_state

    # 导出所有变量
    export_all_variables
}

# 导出所有配置变量
export_all_variables() {
    # 导出所有配置变量，但保留已设置的值
    export SCRIPT_VERSION
    export KT_ROOT
    export CONDA_ENV
    export PYTHON_VERSION
    export PYTORCH_VERSION
    export CUDA_VERSION
    export CUDA_VERSION_SHORT
    # 如果FLASH_ATTN_VERSION已经设置，不要覆盖它
    if [[ -n "${FLASH_ATTN_VERSION}" ]]; then
        # 变量已设置，直接使用当前值
        log_debug "FLASH_ATTN_VERSION已设置为: ${FLASH_ATTN_VERSION}，保留用户选择"
    else
        # 变量未设置，导出默认值
        export FLASH_ATTN_VERSION
    fi
    export USE_APT_MIRROR
    export USE_CONDA_MIRROR
    export USE_PIP_MIRROR
    export USE_GIT_HOSTS
    export USE_NUMA
    export GPU_MODE
    export FORCE_CUDA_REINSTALL
    export CPU_ARCH_OPT
    export CONFIG_DIR
    export LOG_DIR
    export LOG_FILE
    export USE_MIRROR
    export USE_BALANCE_SERVE
}

# 注意：环境变量设置功能已移至setup_env_vars.sh模块
# 此处保留setup_global_compile_environment函数名以保持兼容性，但实际功能由setup_env_vars.sh处理
setup_global_compile_environment() {
    log_info "全局编译环境设置由setup_env_vars.sh模块处理"

    # 如果setup_env_vars模块未加载，则提供基本的兼容性设置
    if [[ -z "$CMAKE_ARGS" ]]; then
        log_warn "setup_env_vars模块未执行，提供基本环境变量设置"
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    fi

    log_info "全局编译环境设置完成"
}

# 模块注册系统
declare -A KT_MODULES
declare -A KT_MODULE_DEPENDENCIES
declare -A KT_MODULE_INITIALIZED

# 模块注册函数
register_module() {
    local module_name="$1"
    local module_description="$2"
    local dependencies="$3"

    KT_MODULES["$module_name"]="$module_description"
    KT_MODULE_DEPENDENCIES["$module_name"]="$dependencies"
    KT_MODULE_INITIALIZED["$module_name"]="false"

    log_debug "注册模块: $module_name - $module_description"
    if [[ -n "$dependencies" ]]; then
        log_debug "  依赖: $dependencies"
    fi
}

# 模块初始化函数
initialize_module() {
    local module_name="$1"

    if [[ "${KT_MODULE_INITIALIZED[$module_name]}" == "true" ]]; then
        log_debug "模块 $module_name 已初始化，跳过"
        return 0
    fi

    # 检查依赖
    local dependencies="${KT_MODULE_DEPENDENCIES[$module_name]}"
    if [[ -n "$dependencies" ]]; then
        log_debug "初始化模块 $module_name 的依赖: $dependencies"
        for dep in $dependencies; do
            if [[ "${KT_MODULE_INITIALIZED[$dep]}" != "true" ]]; then
                log_debug "初始化依赖模块: $dep"
                initialize_module "$dep"
            fi
        done
    fi

    # 执行模块初始化
    local init_func="init_${module_name}"
    if declare -f "$init_func" > /dev/null 2>&1; then
        log_info "初始化模块: $module_name"
        if "$init_func"; then
            KT_MODULE_INITIALIZED["$module_name"]="true"
            log_info "模块 $module_name 初始化成功"
        else
            log_error "模块 $module_name 初始化失败"
            return 1
        fi
    else
        log_warn "模块 $module_name 没有初始化函数，标记为已初始化"
        KT_MODULE_INITIALIZED["$module_name"]="true"
    fi

    return 0
}

# 模块接口验证函数
validate_module_interface() {
    local module_name="$1"
    local required_functions="$2"

    for func in $required_functions; do
        if ! declare -f "$func" > /dev/null 2>&1; then
            log_error "模块 $module_name 缺少必需函数: $func"
            return 1
        fi
    done

    return 0
}

# 安全的模块函数调用
safe_module_call() {
    local module_name="$1"
    local function_name="$2"
    shift 2
    local args="$@"

    if [[ "${KT_MODULE_INITIALIZED[$module_name]}" != "true" ]]; then
        log_error "模块 $module_name 未初始化，无法调用函数 $function_name"
        return 1
    fi

    local full_function_name="${module_name}_${function_name}"
    if declare -f "$full_function_name" > /dev/null 2>&1; then
        log_debug "调用模块函数: $full_function_name $args"
        "$full_function_name" "$@"
        return $?
    else
        log_error "模块 $module_name 没有函数: $function_name"
        return 1
    fi
}

# 模块配置管理 - 已移除配置文件生成功能，仅保留内存中的配置管理
save_module_config() {
    local module_name="$1"
    # 不再生成配置文件，仅在内存中管理配置
    log_debug "模块 $module_name 配置已在内存中管理（配置文件生成功能已移除）"
}

load_module_config() {
    local module_name="$1"
    # 配置文件加载功能已移除，配置仅在内存中管理
    log_debug "模块 $module_name 配置加载功能已移除（配置文件生成功能已移除）"
}

# 模块间通信接口
module_call() {
    local target_module="$1"
    local function_name="$2"
    shift 2

    if [[ -z "${KT_MODULES[$target_module]}" ]]; then
        log_error "目标模块未注册: $target_module"
        return 1
    fi

    if [[ "${KT_MODULE_INITIALIZED[$target_module]}" != "true" ]]; then
        log_error "目标模块未初始化: $target_module"
        return 1
    fi

    local full_function_name="${target_module}_${function_name}"
    if declare -f "$full_function_name" > /dev/null 2>&1; then
        log_debug "模块间调用: $target_module.$function_name"
        "$full_function_name" "$@"
        return $?
    else
        log_error "模块 $target_module 没有函数: $function_name"
        return 1
    fi
}

# 初始化核心模块
init_core() {
    log_info "初始化核心模块..."

    # 设置模块特定的配置变量前缀
    local core_prefix="core"

    # 导出核心配置
    export core_log_dir="$LOG_DIR"
    export core_config_dir="$CONFIG_DIR"  # 配置目录功能已移除，仅保留向后兼容
    export core_script_version="$SCRIPT_VERSION"

    return 0
}

# 注册核心模块
register_module "core" "核心功能模块" ""

# 注意：初始化函数init_settings现在需要手动调用
# 不再自动执行，以避免覆盖用户配置
