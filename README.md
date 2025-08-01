2025.8.1  16G显卡运行Kimi K2加载阶段不会突破峰值显存!!!
2025.7.25 开启多个NUMA节点不会占用双倍内存!!!

通过测试的模型：
KVCache-ai/Kimi-K2-Instruct-GGUF 
deepseek-ai/DeepSeek-R1-0528 
unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF

已知问题：
1、带AMX支持的cpu使用带amx的yaml配置文件运行会出错，amx后台还没改好
2、prefill性能下降，优化未完成
3、50系显卡可能出现 RuntimeError: pidfd_getfd: Operation not permitted  原因未知
4、在 Intel 至强平台上运行慢， 原因未知


基本介绍：

一、这个分支提供一个NUMA优化的稳定版本，以及部分BUG处理和一些内存、显存优化支持，根据情况与合并主线新特性。

二、优化了numa内存使用，多个numa节点只用一份内存，32个节点上均衡，解码速度不下降，略有提升。
增加一个环境变量THREADS_PER_NODE控制每个numa节点用几个线程计算，nsp4 8个节点设置THREADS_PER_NODE=8意味着64个计算线程，如果开了L3 Cache NUMA Domain 32个节点, 设置THREADS_PER_NODE=2也是64个计算线程。

三、增加一个VLinearMarlin16，16显卡运行Kimi K2遇到启动爆显存时可以在yaml中配置使用

!!!更多安装类问题参考主线，没有改变!!!

git clone https://github.com/guqiong96/ktransformers.git

git checkout full-support-numa

git submodule update --init --recursive --verbose

USE_BALANCE_SERVE=1 USE_NUMA=1 sh install.sh


THREADS_PER_NODE=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ~/Downloads/ktransformers/ktransformers/server/main.py \
    --gguf_path ~/Downloads/DeepSeek-R1-0528-GGUF  \
    --model_path ~/Models/DeepSeek-R1-0528 \
    --model_name DeepSeek-R1  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 16384 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/ktransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
    --force_think \
    --use_cuda_graph \
    --host :: \
    --port 8070 \
    --max_batch_size 4 \
    --backend_type balance_serve

