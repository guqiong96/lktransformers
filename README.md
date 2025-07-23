运行命令没改变，增加一个环境变量THREADS_PER_NODE控制每个numa节点用几个线程计算，nsp4 8个节点设置THREADS_PER_NODE=8意味着64个计算线程，如果开了L3 Cache NUMA Domain 32个节点, 设置THREADS_PER_NODE=2也是64个计算线程。
编译命令：USE_NUMA=1 MAX_JOBS=64 KTRANSFORMERS_FORCE_BUILD=TRUE python -m build --no-isolation --wheel  
         pip install dist/ktransformers-0.2.3.post2+torch26avx2-cp311-cp311-linux_x86_64.whl
运行命令：THREADS_PER_NODE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ~/Downloads/ktransformers/ktransformers/server/main.py \
    --gguf_path ~/Downloads/DeepSeek-R1-0528-GGUF  \
    --model_path ~/Models/DeepSeek-R1-0528 \
    --model_name DeepSeek-R1  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 64768 \
    --total_context 144703488 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/ktransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    --force_think \
    --use_cuda_graph \
    --host :: \
    --port 8070 \
    --web True

