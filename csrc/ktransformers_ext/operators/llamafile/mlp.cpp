/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:44:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "mlp.h"

MLP::MLP(MLPConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj; 
    
    #ifdef USE_NUMA
    gate_numa_.resize(numa_nodes_);
    up_numa_.resize(numa_nodes_);
    down_numa_.resize(numa_nodes_); 
    gate_numa_size_.resize(numa_nodes_);
    up_numa_size_.resize(numa_nodes_);
    down_numa_size_.resize(numa_nodes_); 
    #endif
    int nth = config_.intermediate_size / config_.stride;
    stride_gate_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    stride_up_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;

    gate_up_blocks_.resize(numa_nodes_);
    int current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) {   
        int n_blocks = (base + (nid < remain));
        gate_numa_size_[nid] = n_blocks * stride_gate_bytes_;
        up_numa_size_[nid] = n_blocks * stride_up_bytes_;
        gate_numa_[nid] = allocate_aligned_numa(gate_numa_size_[nid], nid);
        up_numa_[nid] = allocate_aligned_numa(up_numa_size_[nid], nid);
        gate_up_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        
        current_block += n_blocks; 

    }
   
    nth = config_.hidden_size / config_.stride;
    stride_down_bytes_ = config_.stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    size_t expert_down_bytes = nth * stride_down_bytes_;
    base = nth / numa_nodes_;
    remain = nth % numa_nodes_;
    down_blocks_.resize(numa_nodes_);
    current_block = 0;
    for (int nid = 0; nid < numa_nodes_; nid++) {   
        int n_blocks = (base + (nid < remain));
        down_numa_size_[nid] = n_blocks * stride_down_bytes_;
        down_numa_[nid] = allocate_aligned_numa(down_numa_size_[nid], nid);
        down_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        
        current_block += n_blocks; 
        
    }   
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int start_block = gate_up_blocks_[nid].start_block;
        int n_blocks = gate_up_blocks_[nid].num_blocks;
        for (int ib = 0; ib < n_blocks; ib++) {
            int ith = start_block + ib;
    
            void* gate_ptr = (uint8_t*)gate_proj_ + ith * stride_gate_bytes_;
            void* up_ptr = (uint8_t*)up_proj_ + ith * stride_up_bytes_;
    
            uint8_t* local_gate_ptr = (uint8_t*)gate_numa_[nid]  + ib * stride_gate_bytes_;
            uint8_t* local_up_ptr = (uint8_t*)up_numa_[nid] + ib * stride_up_bytes_;
            
            memcpy(local_gate_ptr, gate_ptr, stride_gate_bytes_);
            memcpy(local_up_ptr, up_ptr, stride_up_bytes_);
           
        }
    }
 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int start_block = down_blocks_[nid].start_block;
        int n_blocks = down_blocks_[nid].num_blocks;
        for (int ib = 0; ib < n_blocks; ib++) {
            int ith = start_block + ib; 
            void* down_ptr = (uint8_t*)down_proj_ + ith * stride_down_bytes_;
            uint8_t* local_down_ptr = (uint8_t*)down_numa_[nid] + ib * stride_down_bytes_;
            memcpy(local_down_ptr, down_ptr, stride_down_bytes_);
          
        }
    } 
    input_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.hidden_size);
    gate_input_ = (uint8_t*) allocate_aligned(config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    up_input_ = (uint8_t*) allocate_aligned(config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    gate_output_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.intermediate_size);
    up_output_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.intermediate_size);
    intermediate_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.intermediate_size);
    down_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
    down_output_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.hidden_size);
}

MLP::~MLP() {
    for (int nid = 0; nid < numa_nodes_; nid++) {  
        free_aligned_numa(gate_numa_[nid], gate_numa_size_[nid]);
        free_aligned_numa(up_numa_[nid], up_numa_size_[nid]);
        free_aligned_numa(down_numa_[nid], down_numa_size_[nid]);
    }
    free_aligned(input_fp32_ , sizeof(float) * config_.group_max_len * config_.hidden_size);
    free_aligned(gate_input_ , config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    free_aligned(up_input_ , config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    free_aligned(gate_output_ , sizeof(float) * config_.group_max_len * config_.intermediate_size);
    free_aligned(up_output_,sizeof(float) * config_.group_max_len * config_.intermediate_size);
    free_aligned(intermediate_fp32_, sizeof(float) * config_.group_max_len * config_.intermediate_size);
    free_aligned(down_input_, config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
    free_aligned(down_output_ ,sizeof(float) * config_.group_max_len * config_.hidden_size);
 
}

void MLP::warm_up(Backend *backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size *
                               ggml_type_size(config_.hidden_type) /
                               ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size *
                                ggml_type_size(config_.hidden_type) /
                                ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    forward_many(1, input.data(), output.data(), backend);
}

static float act_fn(float x) { return x / (1.0f + expf(-x)); }

void MLP::forward_many(int qlen, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, input_fp32_, qlen * config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(input_fp32_, gate_input_, qlen * config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(input_fp32_, gate_input_, qlen * config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(input_fp32_, up_input_, qlen * config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) { 
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - gate_up_blocks_[nid].start_block;
        int offset = x % gate_up_blocks_[nid].num_blocks; 
        int ith = gate_up_blocks_[nid].start_block + offset;
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  offset * config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + ith * config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif
        float* gate_output_ptr = gate_output_ + ith * config_.stride;
        llamafile_sgemm(config_.stride, qlen, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  offset * config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + ith * config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif
        float* up_output_ptr = up_output_ + ith * config_.stride;
        llamafile_sgemm(config_.stride, qlen, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < qlen; i++) {
            for (int j = ith * config_.stride; j < (ith + 1) * config_.stride; j++) {
                intermediate_fp32_[i * config_.intermediate_size + j] = act_fn(gate_output_[i * config_.intermediate_size + j]) * up_output_[i * config_.intermediate_size + j];
            }
            if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
                float* intermediate_fp32_ptr = intermediate_fp32_ + i * config_.intermediate_size + ith * config_.stride;
                void* down_input_ptr = (uint8_t*)down_input_ + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
                from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            }
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        from_float(intermediate_fp32_, down_input_, qlen * config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
    }
    nth = config_.hidden_size / config_.stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - down_blocks_[nid].start_block;
        int offset = x % down_blocks_[nid].num_blocks; 
        int ith = down_blocks_[nid].start_block + offset; 
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_numa_[nid] + offset * config_.stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + ith * config_.stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif
        float* down_output_ptr = down_output_ + ith * config_.stride;
        llamafile_sgemm(config_.stride, qlen, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            for (int i = 0; i < qlen; i++) {
                float* output_fp32_ptr = down_output_ + i * config_.hidden_size + ith * config_.stride;
                void* output_ptr = (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type) + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
            }
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(down_output_, output, qlen * config_.hidden_size, config_.hidden_type);
    }
}

void MLP::forward(int qlen, const void* input, void* output, Backend* backend) {
    if (qlen <= 0) {
        return;
    }
    int forward_len = std::min(qlen, config_.group_max_len);
    forward_many(forward_len, input, output, backend);
    forward(qlen - forward_len, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}