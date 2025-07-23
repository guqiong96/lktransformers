/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:45:18
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "linear.h"

Linear::Linear(LinearConfig config) {
    config_ = config;
    proj_ = config_.proj;
    #ifdef USE_NUMA
    proj_numa_.resize(numa_nodes_); 
    proj_numa_size_.resize(numa_nodes_); 
    #endif

    int nth = config_.output_size / config_.stride;
    stride_bytes_ = config_.stride * config_.input_size * ggml_type_size(config_.proj_type) / ggml_blck_size(config_.proj_type);
 
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;
    proj_blocks_.resize(numa_nodes_);
    int current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) { 
        int n_blocks = (base + (nid < remain));
        proj_numa_size_[nid] = n_blocks * stride_bytes_;
        proj_numa_[nid] = allocate_aligned_numa(proj_numa_size_[nid], nid); 
        proj_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        
        current_block += n_blocks; 

    }    
 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int start_block = proj_blocks_[nid].start_block;
        int n_blocks = proj_blocks_[nid].num_blocks;
        for (int ib = 0; ib < n_blocks; ib++) {
            int ith = start_block + ib; 
            void* proj_ptr = (uint8_t*)proj_ + ith * stride_bytes_;
            uint8_t* local_ptr = (uint8_t*)proj_numa_[nid] + ib * stride_bytes_;
            memcpy(local_ptr, proj_ptr, stride_bytes_);
          
        }
    } 
    input_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.input_size);
    proj_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.input_size * ggml_type_size(ggml_internal_get_type_traits(config_.proj_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.proj_type).vec_dot_type));
    proj_output_ = (float*)allocate_aligned(sizeof(float) * config_.group_max_len * config_.output_size);

}

Linear::~Linear() {
    for (int nid = 0; nid < numa_nodes_; nid++) {  
        free_aligned_numa(proj_numa_[nid], proj_numa_size_[nid]);
    }
    free_aligned(input_fp32_ , sizeof(float) * config_.group_max_len * config_.input_size);
    free_aligned(proj_input_ , config_.group_max_len * config_.input_size * ggml_type_size(ggml_internal_get_type_traits(config_.proj_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.proj_type).vec_dot_type));
    free_aligned(proj_output_ , sizeof(float) * config_.group_max_len * config_.output_size);

}

void Linear::warm_up(Backend *backend) {
    std::vector<float> input_fp32(config_.input_size);
    std::vector<uint8_t> input(config_.input_size *
                               ggml_type_size(config_.hidden_type) /
                               ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.output_size *
                                ggml_type_size(config_.hidden_type) /
                                ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.input_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.input_size, config_.hidden_type);
    forward_many(1, input.data(), output.data(), backend);
}

void Linear::forward_many(int qlen, const void* input, void* output, Backend* backend) {
    const void* proj_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.proj_type).vec_dot_type) {
        proj_input_ptr = input;
    } else {
        to_float(input, input_fp32_, qlen * config_.input_size, config_.hidden_type);
        from_float(input_fp32_, proj_input_, qlen * config_.input_size, ggml_internal_get_type_traits(config_.proj_type).vec_dot_type);
        proj_input_ptr = proj_input_;
    }
    int nth = config_.output_size / config_.stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - proj_blocks_[nid].start_block;
        int offset = x % proj_blocks_[nid].num_blocks; 
        int ith = proj_blocks_[nid].start_block + offset; 
        #ifdef USE_NUMA
        void* proj_ptr = (uint8_t*)proj_numa_[nid] + offset * config_.stride * config_.input_size * ggml_type_size(config_.proj_type) / ggml_blck_size(config_.proj_type);
        #else
        void* proj_ptr = (uint8_t*)proj_ + ith * config_.stride * config_.input_size * ggml_type_size(config_.proj_type) / ggml_blck_size(config_.proj_type);
        #endif
        float* proj_output_ptr = proj_output_ + ith * config_.stride;
        llamafile_sgemm(config_.stride, qlen, config_.input_size / ggml_blck_size(config_.proj_type), proj_ptr, config_.input_size / ggml_blck_size(config_.proj_type), proj_input_ptr, config_.input_size / ggml_blck_size(config_.proj_type), proj_output_ptr, config_.output_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.proj_type, ggml_internal_get_type_traits(config_.proj_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            for (int i = 0; i < qlen; i++) {
                float* output_fp32_ptr = proj_output_ + i * config_.output_size + ith * config_.stride;
                void* output_ptr = (uint8_t*)output + i * config_.output_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type) + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
            }
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(proj_output_, output, qlen * config_.output_size, config_.hidden_type);
    }
}

void Linear::forward(int qlen, const void* input, void* output, Backend* backend) {
    if (qlen <= 0) {
        return;
    }
    int forward_len = std::min(qlen, config_.group_max_len);
    forward_many(forward_len, input, output, backend);
    forward(qlen - forward_len, (uint8_t*)input + forward_len * config_.input_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.output_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}