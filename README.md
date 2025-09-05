# LKtransformers - 开启NUMA内存占用不翻倍！AMX支持GGUF模型，内存占用不翻倍, 支持NUMA加速！

[2025-09-05 AMX预览版：支持GGUF模型，内存占用不翻倍，支持NUMA加速，预览版q4量化测试通过， 优化了编译参数]

[2025-08-29 将计算进程放主进程运行，接口、调度放子进程，方便调试及及时发现错误提示] 

[2025-08-22 prefill使用一次多tonken矩阵计算的numa最优化版本， 增加环境变量LK_POWER_SAVING，默认为关闭，开启时使用环境变量LK_POWER_SAVING=1 可以缓解内存过热降速]

[2025-08-21 降低线程占用CPU，避免内存温度上升过快，prefill切换至numa最优化版本]

[2025-08-18 prefill提速10%, 使用环境变量LK_THREADS调节性能]

[2025-08-14 AVX2 激活函数]

[2025-08-12 decode提升速度，32节点为最优速度]

[2025-08-10 prefill提升速度，32节点为最优速度，同时优化了超线程支持]

[2025-08-02 修复sh install.sh在shell为dash时的兼容posix问题，导致未能安装自带flashinfer,出现的一些莫名问题
RuntimeError: pidfd_getfd: Operation not permitted，使用PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True后报错等
更新代码重新运行USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh]

[2025-08-01 解决 16G 显卡加载 Kimi K2 的显存峰值问题]

[2025-07-25 开启NUMA内存占用不翻倍]

本分支提供 NUMA 优化的稳定版本，包含内存/显存优化和一些问题修复。

## ✨ 核心特性

- **NUMA 内存优化**：多个 NUMA 节点共享单份内存，32节点负载均衡，解码速度不降略有提升
- **线程精细控制**：通过 `LK_THREADS` 环境变量管理计算线程数
- **显存优化**：新增 `VLinearMarlin16` 支持，解决 16G 显卡加载 Kimi K2 的显存峰值问题
- **已验证模型**：
  - `KVCache-ai/Kimi-K2-Instruct-GGUF`
  - `deepseek-ai/DeepSeek-R1-0528`
  - `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf `
    <img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/0eaef315-71a4-4371-9cac-a54e3e640f80" />

  - `unsloth/DeepSeek-V3.1-GGUF/UD-Q4_K_XL`
    <img width="1000" height="1364" alt="image" src="https://github.com/user-attachments/assets/493eb4f6-b807-4408-841b-da0d8046e94d" />


## ⚠️ 已知问题

1. 支持 AMX 的 CPU 使用 amx 配置文件会报错（已解决：AMX开启无需特别配置）
2. Prefill 性能下降（已解决）
3. 报错 `RuntimeError: pidfd_getfd: Operation not permitted`可以去掉PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True(已解决， 重新运行USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh)
4. Intel 至强平台，或者开启超线程运行速度慢（已解决）
5. 推理测试多轮以后或使用长输入会话出现错误，检查并确保cache_lens = max_new_tokens + 2048， 并且cache_lens、max_new_tokens为 1024 的倍数， 如要设置max_batch_size， 显存足够的情况下增加相应倍数的cache_lens。
6. Eingine error: The size of tensor a (4)must match the size of tensor b (5) at non-singleton dimension 0 ,

## 🚀 快速开始

### 安装步骤

git clone https://github.com/guqiong96/lktransformers.git

git checkout full-support-numa

git submodule update --init --recursive

USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh

### 更新源码

git pull

USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh

### 运行示例
LK_POWER_SAVING=1 LK_THREADS=62 python ~/Downloads/KTransformers/ktransformers/server/main.py \
    --gguf_path ~/Models/Kimi-K2-Instruct-GGUF  \
    --model_path ~/Models/Kimi-K2-Instruct-GGUF \
    --model_name Kimi-K2-Instruct-GGUF  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 18432 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/KTransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
    --force_think \
    --use_cuda_graph \
    --host 0.0.0.0 \
    --port 8070 \
    --backend_type balance_serve \
    --chunk_size 1024

## 🔧 配置技巧

- **NUMA 线程配置**：
  - 开启最多NUMA节点，AMD EPYC 开启 L3 As NUMA Domain 

- **显存优化**：在 YAML 配置中使用 `VLinearMarlin16` 防止 16G 显卡显存溢出

## 📌 注意事项

1. 更新后出现疑难问题，在主目录运行 
  git pull
  USE_BALANCE_SERVE=1 USE_NUMA=1 bash install.sh
2. 更多安装问题请参考主线文档
3. 定期合并主线获取最新特性


![a42193311f0cf149b9204ea8f3ad37d6](https://github.com/user-attachments/assets/874b2c38-0edb-42b0-a017-0e5f7ec1294e)


 
