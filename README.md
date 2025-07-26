git clone https://github.com/guqiong96/ktransformers.git
git checkout full-support-numa
git submodule update --init --recursive --verbose
export  USE_NUMA=1 
sh install.sh

THREADS_PER_NODE=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ~/Downloads/ktransformers/ktransformers/server/main.py \
    --gguf_path ~/Downloads/DeepSeek-R1-0528-GGUF  \
    --model_path ~/Models/DeepSeek-R1-0528 \
    --model_name DeepSeek-R1  \
    --cpu_infer 28 \
    --max_new_tokens 16384 \
    --cache_lens 64768 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ~/Downloads/ktransformers/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    --force_think \
    --use_cuda_graph \
    --host :: \
    --port 8070 \
    --web True
