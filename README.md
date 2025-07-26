git clone https://github.com/guqiong96/ktransformers.git

git checkout full-support-numa

git submodule update --init --recursive --verbose

export  USE_NUMA=1 

sh install.sh

npm config set registry https://mirrors.cloud.tencent.com/npm/ 

cd ktransformers/website/

npm install @vue/cli --verbose

npm run build --verbose

THREADS_PER_NODE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ~/Downloads/KTransformers/ktransformers/server/main.py \
    --gguf_path ~/Downloads/DeepSeek-R1-0528-GGUF  \
    --model_path ~/Models/DeepSeek-R1-0528 \
    --model_name DeepSeek-R1  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 64768 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/KTransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    --force_think \
    --use_cuda_graph \
    --host :: \
    --port 8070 \
    --web True

从主线最新代码（2025.7.26）更新的full-support-numa版本已经提交，测试deepseek r1 0528 q4_k_m通过，Qwen3-235B测试如有结果可以提交到这里。
目前这个版本amx那部分的numa优化未经测试（没机器），可能出错
