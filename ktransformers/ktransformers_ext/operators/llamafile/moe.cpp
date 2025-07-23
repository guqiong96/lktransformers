/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>
 
#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif
#include <unistd.h>

MOE::MOE(MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    //config_.stride = QK_K;
    
    #ifdef USE_NUMA
    gate_numa_.resize(numa_nodes_);
    up_numa_.resize(numa_nodes_);
    down_numa_.resize(numa_nodes_); 

    gate_numa_size_.resize(numa_nodes_);
    up_numa_size_.resize(numa_nodes_);
    down_numa_size_.resize(numa_nodes_); 
 

    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    
   
    
    int nth = config_.intermediate_size / config_.stride;
    stride_gate_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    stride_up_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
    size_t expert_gate_bytes = nth * stride_gate_bytes_;
    size_t expert_up_bytes = nth * stride_up_bytes_;
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;

    gate_up_blocks_.resize(numa_nodes_);
    int current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = (base + (nid < remain));
        gate_numa_size_[nid] = config_.expert_num * n_blocks * stride_gate_bytes_;
        up_numa_size_[nid] = config_.expert_num * n_blocks * stride_up_bytes_;
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
        down_numa_size_[nid] = config_.expert_num * n_blocks * stride_down_bytes_;
        down_numa_[nid] = allocate_aligned_numa(down_numa_size_[nid], nid);
        down_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        
        current_block += n_blocks; 
        
    } 
    
    //int m_nth = config_.intermediate_size / QK_K;
    int m_nth = config_.intermediate_size / config_.stride;
    base = m_nth / numa_nodes_;
    remain = m_nth % numa_nodes_;
    m_gate_up_blocks_.resize(numa_nodes_);
    current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) { 
        int n_blocks = (base + (nid < remain));
        m_gate_up_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        current_block += n_blocks; 
    }
    //m_nth = config_.hidden_size / QK_K;
    m_nth = config_.hidden_size / config_.stride;
    base = m_nth / numa_nodes_;
    remain = m_nth % numa_nodes_;
    m_down_blocks_.resize(numa_nodes_);
    current_block = 0;
    for (int nid = 0; nid < numa_nodes_; nid++) { 
        int n_blocks = (base + (nid < remain));
        m_down_blocks_[nid] = NumaBlock{
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
            for (int e = 0; e < config_.expert_num; e++) {
 
                void* gate_ptr = (uint8_t*)gate_proj_ + e * expert_gate_bytes + ith * stride_gate_bytes_;
                void* up_ptr = (uint8_t*)up_proj_ + e * expert_up_bytes + ith * stride_up_bytes_;
      
                uint8_t* local_gate_ptr = (uint8_t*)gate_numa_[nid]  + (e * n_blocks + ib) * stride_gate_bytes_;
                uint8_t* local_up_ptr = (uint8_t*)up_numa_[nid] + (e * n_blocks + ib) * stride_up_bytes_;
                
                memcpy(local_gate_ptr, gate_ptr, stride_gate_bytes_);
                memcpy(local_up_ptr, up_ptr, stride_up_bytes_);
            }
        }
    }
 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int start_block = down_blocks_[nid].start_block;
        int n_blocks = down_blocks_[nid].num_blocks;
        for (int ib = 0; ib < n_blocks; ib++) {
            int ith = start_block + ib;
            for (int e = 0; e < config_.expert_num; e++) {
                void* down_ptr = (uint8_t*)down_proj_ + e * expert_down_bytes + ith * stride_down_bytes_;
                uint8_t* local_down_ptr = (uint8_t*)down_numa_[nid] + (e * n_blocks + ib) * stride_down_bytes_;
                memcpy(local_down_ptr, down_ptr, stride_down_bytes_);
            }
        }
    }
 
    s_output_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    s_input_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    s_gate_input_ = (uint8_t*)allocate_aligned(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    s_up_input_ = (uint8_t*)allocate_aligned(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));

    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_gate_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_up_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_intermediate_fp32_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_down_input_[i] = (uint8_t*)allocate_aligned(config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
        s_down_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    } 
     
    #endif 

    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_input_fp32_[i] = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
        m_gate_input_[i] = (uint8_t*)allocate_aligned(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
        m_up_input_[i] =  (uint8_t*)allocate_aligned(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    }
    m_local_gate_input_ = (uint8_t*)allocate_aligned(config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    m_local_up_input_ = (uint8_t*)allocate_aligned(config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    m_local_gate_output_ = (float*)allocate_aligned(sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    m_local_up_output_ = (float*)allocate_aligned(sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    m_local_intermediate_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    m_local_down_input_ = (uint8_t*)allocate_aligned(config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
    m_local_down_output_ = (float*)allocate_aligned( sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size);
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_output_fp32_[i] = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    } 

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num); 

    std::cout <<  "MOE init success ." << std::endl;
}

MOE::~MOE() {
    
    for (int nid = 0; nid < numa_nodes_; nid++) {  
        free_aligned_numa(gate_numa_[nid], gate_numa_size_[nid]);
        free_aligned_numa(up_numa_[nid], up_numa_size_[nid]);
        free_aligned_numa(down_numa_[nid], down_numa_size_[nid]);
    }
    free_aligned(s_input_fp32_, sizeof(float) * config_.hidden_size);
    free_aligned(s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    free_aligned(s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    for (int i = 0; i < config_.routed_expert_num; i++) { 
        free_aligned(s_gate_output_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_up_output_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
        free_aligned(s_down_output_[i], sizeof(float) * config_.hidden_size);
    } 

    for (int i = 0; i < config_.group_max_len; i++) {
        free_aligned(m_input_fp32_[i], sizeof(float) * config_.hidden_size);
        free_aligned(m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
        free_aligned(m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    }
    free_aligned(m_local_gate_input_ , config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    free_aligned(m_local_up_input_ , config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    free_aligned(m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    free_aligned(m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    free_aligned(m_local_intermediate_fp32_ , sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size);
    free_aligned(m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
    free_aligned(m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size);
    for (int i = 0; i < config_.group_max_len; i++) {
        free_aligned(m_output_fp32_[i], sizeof(float) * config_.hidden_size);
    } 
  
}

void MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend); 
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }

    int nth = config_.intermediate_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - gate_up_blocks_[nid].start_block * k;
  
        int expert_idx = x / gate_up_blocks_[nid].num_blocks; 
        int expert_id = expert_ids[expert_idx];
        int offset = x % gate_up_blocks_[nid].num_blocks; 
        int ith = gate_up_blocks_[nid].start_block + offset;
        assert(x == expert_idx * gate_up_blocks_[nid].num_blocks + offset);
        #ifdef USE_NUMA
            void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * gate_up_blocks_[nid].num_blocks + offset) * stride_gate_bytes_;
        #else
            void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif
        
        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        #ifdef USE_NUMA
            void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * gate_up_blocks_[nid].num_blocks + offset) * stride_up_bytes_;
        #else
            void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }  
    nth = config_.hidden_size / config_.stride;  
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int ith = task_id;
        int x = ith - down_blocks_[nid].start_block;  
        int offset = x % down_blocks_[nid].num_blocks;  
        
         for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];
            #ifdef USE_NUMA   
            void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * down_blocks_[nid].num_blocks + offset) * stride_down_bytes_;
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    Backend_NUMA::getInstance().do_work(qlen, nullptr, [&](int i){
        const void* gate_input_ptr;
        const void* up_input_ptr;        
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    //int stride = QK_K; 
    int stride = config_.stride;
    int nth = config_.intermediate_size / stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - m_gate_up_blocks_[nid].start_block * config_.expert_num;
        int expert_idx = x / m_gate_up_blocks_[nid].num_blocks; 
        int offset = x % m_gate_up_blocks_[nid].num_blocks; 
        int ith = m_gate_up_blocks_[nid].start_block + offset;   

        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_idx * m_gate_up_blocks_[nid].num_blocks + offset)* stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
            void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_idx * m_gate_up_blocks_[nid].num_blocks + offset)* stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
            void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
                float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
                void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
                from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            }
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < config_.expert_num; i++) {
            from_float(m_local_intermediate_fp32_ptr_[i], m_local_down_input_ptr_[i], config_.intermediate_size * m_local_num_[i], ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }  
    //stride = QK_K;
    stride = config_.stride;
    nth = config_.hidden_size / stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth,  nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int x = task_id - m_down_blocks_[nid].start_block * config_.expert_num;
        int expert_idx = x / m_down_blocks_[nid].num_blocks; 
        int offset = x % m_down_blocks_[nid].num_blocks; 
        int ith = m_down_blocks_[nid].start_block + offset;  

        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_idx * m_down_blocks_[nid].num_blocks + offset) * stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    Backend_NUMA::getInstance().do_work(qlen, nullptr,[&](int i){
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
         from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    //if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    //}
    //int forward_len = std::min(config_.group_max_len, qlen);
    //forward_many(forward_len, k, expert_ids, weights, input, output, backend);
    //forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}


