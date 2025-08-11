# LKtransformers - 开启NUMA内存占用不翻倍
[2025-08-10 prefill提升，32节点为最优速度，同时优化了超线程支持]
[2025-08-02 修复sh install.sh在shell为dash时的兼容posix问题，导致未能安装自带flashinfer,出现的一些莫名问题
RuntimeError: pidfd_getfd: Operation not permitted，使用PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True后报错等
更新代码重新运行USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh 或者 USE_BALANCE_SERVE=1 USE_NUMA=1 sh install.sh]

[2025-08-01 解决 16G 显卡加载 Kimi K2 的显存峰值问题]

[2025-07-25 开启NUMA内存占用不翻倍]

本分支提供 NUMA 优化的稳定版本，包含内存/显存优化和一些问题修复。

## ✨ 核心特性

- **NUMA 内存优化**：多个 NUMA 节点共享单份内存，32节点负载均衡，解码速度不降略有提升
- **线程精细控制**：通过 `THREADS_PER_NODE` 环境变量管理每个节点的计算线程数
- **显存优化**：新增 `VLinearMarlin16` 支持，解决 16G 显卡加载 Kimi K2 的显存峰值问题
- **已验证模型**：
  - `KVCache-ai/Kimi-K2-Instruct-GGUF`
  - `deepseek-ai/DeepSeek-R1-0528`
  - `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF`

## ⚠️ 已知问题

1. 支持 AMX 的 CPU 使用 amx 配置文件会报错（AMX 后台NUMA改造未完成）
2. Prefill 性能下降（已解决）
3. 报错 `RuntimeError: pidfd_getfd: Operation not permitted`可以去掉PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 运行，速度变慢，提交问题到这里(已解决， 重新运行USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh 或者 USE_BALANCE_SERVE=1 USE_NUMA=1 sh install.sh)
4. Intel 至强平台，或者开启超线程运行速度慢（已解决）

## 🚀 快速开始

### 安装步骤

git clone https://github.com/guqiong96/ktransformers.git

git checkout full-support-numa

git submodule update --init --recursive --verbose

USE_BALANCE_SERVE=1 USE_NUMA=1 sh install.sh

### 运行示例
THREADS_PER_NODE=8 python ~/Downloads/KTransformers/ktransformers/server/main.py \
    --gguf_path ~/Models/Kimi-K2-Instruct-GGUF  \
    --model_path ~/Models/Kimi-K2-Instruct \
    --model_name Kimi-K2  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 16384 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/KTransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
    --force_think \
    --use_cuda_graph \
    --host 0.0.0.0 \
    --port 8070 \
    --max_batch_size 4 \
    --backend_type balance_serve

## 🔧 配置技巧

- **NUMA 线程配置**：
  - 8节点 × 8线程 = 64计算线程
  - 32节点 × 2线程 = 64计算线程
- **显存优化**：在 YAML 配置中使用 `VLinearMarlin16` 防止 16G 显卡显存溢出

## 📌 注意事项

1. 更新后出现疑难问题，运行 USE_BALANCE_SERVE=1 USE_NUMA=1 sh install.sh
2. 更多安装问题请参考主线文档
3. 定期合并主线获取最新特性

![Weixin Image_2025-08-02_183944_421](https://github.com/user-attachments/assets/8f227407-2e0c-48b1-a677-ff0545b5b7a8)
 
