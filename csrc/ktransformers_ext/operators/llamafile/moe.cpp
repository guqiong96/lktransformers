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



MOE::MOE(MOEConfig config) { 
    
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    hidden_type_size = ggml_type_size(config_.hidden_type);
    hidden_blk_size = ggml_blck_size(config_.hidden_type);
    hidden_bytes = config_.hidden_size * hidden_type_size / hidden_blk_size;

    gate_vec_type = ggml_internal_get_type_traits(config_.gate_type).vec_dot_type;
    gate_type_size = ggml_type_size(gate_vec_type);
    gate_blk_size = ggml_blck_size(gate_vec_type);
    gate_bytes = config_.hidden_size * gate_type_size / gate_blk_size;

    up_vec_type = ggml_internal_get_type_traits(config_.up_type).vec_dot_type;
    up_type_size = ggml_type_size(up_vec_type);
    up_blk_size = ggml_blck_size(up_vec_type);
    up_bytes = config_.hidden_size * up_type_size / up_blk_size;

    down_vec_type = ggml_internal_get_type_traits(config_.down_type).vec_dot_type;
    down_type_size = ggml_type_size(down_vec_type);
    down_blk_size = ggml_blck_size(down_vec_type);
    down_bytes = config_.intermediate_size * down_type_size / down_blk_size;
    
    if(numa_nodes_ <= 8){
        config_.stride = QK_K;
        config_.group_min_len = 8;
    }else
        config_.group_min_len = numa_nodes_;
    config_.group_max_len = 1024;
     
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
    s_gate_input_ = (uint8_t*)allocate_aligned(gate_bytes);
    s_up_input_ = (uint8_t*)allocate_aligned(up_bytes);

    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_gate_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_up_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_intermediate_fp32_[i] = (float*)allocate_aligned(sizeof(float) * config_.intermediate_size);
        s_down_input_[i] = (uint8_t*)allocate_aligned(down_bytes);
        s_down_output_[i] = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    }  

    input_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);
    gate_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * gate_bytes);
    up_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * up_bytes);
    gate_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    up_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    down_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * down_bytes);
    down_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    output_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);  

    std::cout << "config_.stride : " << config_.stride << " down_blk_size :" << down_blk_size << " hidden_blk_size :" << hidden_blk_size << std::endl;
    std::cout << "MOE init success ." << std::endl;
}

MOE::~MOE() {
    for (int nid = 0; nid < numa_nodes_; nid++) {  
        free_aligned_numa(gate_numa_[nid], gate_numa_size_[nid]);
        free_aligned_numa(up_numa_[nid], up_numa_size_[nid]);
        free_aligned_numa(down_numa_[nid], down_numa_size_[nid]);
    }
    free_aligned(s_input_fp32_, sizeof(float) * config_.hidden_size);
    free_aligned(s_gate_input_, gate_bytes);
    free_aligned(s_up_input_, up_bytes);
    for (int i = 0; i < config_.routed_expert_num; i++) { 
        free_aligned(s_gate_output_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_up_output_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size);
        free_aligned(s_down_input_[i], down_bytes);
        free_aligned(s_down_output_[i], sizeof(float) * config_.hidden_size);
    } 

    for (int nid = 0; nid < numa_nodes_; nid++) {
        free_aligned(input_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size);
        free_aligned(gate_input_, config_.group_max_len * gate_bytes);
        free_aligned(up_input_, config_.group_max_len * up_bytes);
        free_aligned(gate_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
        free_aligned(up_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
        free_aligned(down_input_, config_.group_max_len * config_.routed_expert_num * down_bytes);
        free_aligned(down_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
        free_aligned(output_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size);  
    }    

}

void MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(hidden_bytes);
    std::vector<uint8_t> output(hidden_bytes);
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
    if (config_.hidden_type == gate_vec_type && config_.hidden_type == up_vec_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (gate_vec_type == up_vec_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != gate_vec_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != up_vec_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, up_vec_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }

    int nth = config_.intermediate_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        int x = task_id - start_block * k;
        int expert_idx = x / num_blocks; 
        int expert_id = expert_ids[expert_idx];
        int offset = x % num_blocks; 
        int ith = start_block + offset;
        assert(x == expert_idx * num_blocks + offset);
        #ifdef USE_NUMA
            void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * stride_gate_bytes_;
        #else
            void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif
        
        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, gate_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        #ifdef USE_NUMA
            void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * stride_up_bytes_;
        #else
            void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, up_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % down_blk_size == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * down_type_size / down_blk_size;
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, down_vec_type);
        }
    }, nullptr);
    if (config_.stride % down_blk_size != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, down_vec_type);
        }
    }  
    nth = config_.hidden_size / config_.stride;  
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;
        int ith = task_id;
        int x = ith - start_block;  
        int offset = x % num_blocks;  
        
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];
            #ifdef USE_NUMA   
            void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, down_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % hidden_blk_size == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * hidden_type_size / hidden_blk_size;
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);
    if (config_.stride % hidden_blk_size != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    
    if(qlen <= numa_nodes_){
        std::cerr << "qlen must ge than numa nodes "<< std::endl;
        std::abort();
    }
    int base = qlen / numa_nodes_;
    int remain = qlen % numa_nodes_;
    std::vector<NumaBlock> token_blocks(numa_nodes_);
    std::vector<int> token_numa_node(qlen);
    int current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = (base + (nid < remain));
        token_blocks[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        for(int i = current_block; i < current_block + n_blocks; i++){ 
            token_numa_node[i] = nid;
        }
        
        current_block += n_blocks; 

    }

    Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;  
        int token_id = task_id;  
        

        uint8_t* input_uint8_ptr = (uint8_t*)input + token_id * hidden_bytes;
        float* input_fp32_ptr = input_fp32_ + token_id * config_.hidden_size;
        uint8_t* gate_input_ptr = gate_input_ + token_id * gate_bytes;
        uint8_t* up_input_ptr = up_input_ + token_id * up_bytes;

        if (config_.hidden_type == gate_vec_type && config_.hidden_type == up_vec_type) {
            memcpy(gate_input_ptr, input_uint8_ptr, hidden_bytes);
            // up not need to copy
        } else {
            to_float(input_uint8_ptr, input_fp32_ptr, config_.hidden_size, config_.hidden_type);
            if (gate_vec_type == up_vec_type) {
                from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_type);
                // up not need to copy
            } else {
                if (config_.hidden_type != gate_vec_type) {
                    from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_type);
                } else {
                    memcpy(gate_input_ptr, input_uint8_ptr, hidden_bytes);
                }
                if (config_.hidden_type != up_vec_type) {
                    from_float(input_fp32_ptr, up_input_ptr, config_.hidden_size, up_vec_type); 
                } else {
                    memcpy(up_input_ptr, input_uint8_ptr, hidden_bytes);
                }
            }
        }
    }, nullptr); 
    
    int nth = config_.intermediate_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(qlen, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;
 
        int token_id = task_id % qlen; 

        int offset = 0;  
        int ith =  start_block + offset;
        size_t n_stride = num_blocks * config_.stride;
  

        uint8_t* gate_input_ptr = gate_input_ + token_id * gate_bytes;
        for(int i = 0; i < k; i++){
            int expert_idx = token_id * k + i;
            int expert_id = expert_ids[expert_idx];
            void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * stride_gate_bytes_;
            float* gate_output_ptr = gate_output_ + expert_idx * config_.intermediate_size + ith * config_.stride;
            
            llamafile_sgemm(n_stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, gate_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

            
            void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * stride_up_bytes_;
            uint8_t* up_input_ptr = (gate_vec_type == up_vec_type) 
                        ? gate_input_ptr   
                        : up_input_ + token_id * up_bytes; 
            float* up_output_ptr = up_output_ + expert_idx * config_.intermediate_size + ith * config_.stride;
            
            llamafile_sgemm(n_stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, up_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
            for (int i = 0; i < n_stride; i++) {
                up_output_ptr[i] = act_fn(gate_output_ptr[i]) * up_output_ptr[i]; 
            }

            if (config_.stride % ggml_blck_size(config_.down_type) == 0) { 
                uint8_t* down_input_ptr = down_input_ + (expert_idx * config_.intermediate_size + ith * config_.stride) * down_type_size / down_blk_size;
                from_float(up_output_ptr, down_input_ptr, n_stride, down_vec_type);
            }
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.down_type) != 0) {
        Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen, nullptr, [&](int task_id) {
            int nid = Backend_NUMA::numa_node_; 
            int start_block = gate_up_blocks_[nid].start_block;
            int num_blocks = gate_up_blocks_[nid].num_blocks;
    
            int token_id = task_id;  

            for(int i = 0; i < k; i++){
                int expert_idx = token_id * k + i;
                float* up_output_ptr_ = up_output_ + expert_idx * config_.intermediate_size;
                uint8_t* down_input_ptr = down_input_ + (expert_idx * config_.intermediate_size) * down_type_size / down_blk_size;
                from_float(up_output_ptr_, down_input_ptr, config_.intermediate_size, down_vec_type);
            }

        }, nullptr);
    }   
    nth = config_.hidden_size / config_.stride;  
    Backend_NUMA::getInstance().do_k_work_stealing_job(qlen, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;
 
        int token_id = task_id % qlen; 

        int offset = 0;  
        int ith =  start_block + offset;
        size_t n_stride = num_blocks * config_.stride;
        
        float* output_fp32_ptr = output_fp32_  + token_id * config_.hidden_size + ith * config_.stride;
        memset(output_fp32_ptr, 0, n_stride * sizeof(float)); 
        for(int i = 0; i < k; i++){
            int expert_idx = token_id * k + i;
            uint64_t expert_id = expert_ids[expert_idx]; 
            uint8_t* down_input_ptr = down_input_ + expert_idx * config_.intermediate_size * down_type_size / down_blk_size;
            uint8_t* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
            float* down_output_ptr = down_output_  + expert_idx * config_.hidden_size + ith * config_.stride;
            llamafile_sgemm(n_stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, down_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
             
            for (int i = 0; i < n_stride; i++) { 
                output_fp32_ptr[i] += down_output_ptr[i] * weights[expert_idx];
            }
        } 
        
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            void* output_ptr = (uint8_t*)output + (token_id * config_.hidden_size + ith * config_.stride) * hidden_type_size / hidden_blk_size;
            from_float(output_fp32_ptr, output_ptr, n_stride, config_.hidden_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        Backend_NUMA::getInstance().do_k_work_stealing_job(1 , qlen, nullptr, [&](int task_id) {
            int nid = Backend_NUMA::numa_node_; 
            int start_block = down_blocks_[nid].start_block;
            int num_blocks = down_blocks_[nid].num_blocks; 
    
            int token_id = task_id; 


            for(int i = 0; i < k; i++){
                int expert_idx = token_id * k + i;
                float* output_fp32_ptr = output_fp32_ + expert_idx * config_.hidden_size;
                void* output_ptr = (uint8_t*)output + (expert_idx * config_.hidden_size) * hidden_type_size / hidden_blk_size;
                from_float(output_fp32_ptr, output_ptr, config_.hidden_size, config_.hidden_type);
            }

        }, nullptr);
    } 
        
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* batch_size_tensor, Backend* backend) {

    qlen = batch_size_tensor[0];
    
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * hidden_bytes, (uint8_t*)output + i * hidden_bytes, backend);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend);

    batch_size_tensor[0] -= forward_len;
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * hidden_bytes, (uint8_t*)output + forward_len * hidden_bytes, batch_size_tensor, backend);
}
 

