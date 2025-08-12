/**
 * @Description  : 
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "backend_numa.h"
#include <fstream>
#include <unordered_set>


thread_local int Backend_NUMA::numa_node_ = -1;
thread_local int Backend_NUMA::thread_local_id_ = -1;

struct CpuInfo {
    int cpuid_id;
    int core_id;
    int physical_package_id;
    bool is_logic_core;
};

int read_topology(const std::string& cpu_path, const char* file) {
    std::ifstream ifs(cpu_path + "/" + file);
    int value;
    return ifs >> value ? value : -1;
}


std::vector<CpuInfo> get_cpu_info(int & num_cores_) { 
    std::vector<CpuInfo> result;
    const int total_cpus = numa_num_configured_cpus();
    std::unordered_set<int> seen_cores;
    
    int max_package_id = 0;
    for (int i = 0; i < total_cpus; ++i) {
        CpuInfo info{};
        
        const std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        
        int x = read_topology(cpu_dir, "topology/core_id");
        info.physical_package_id = read_topology(cpu_dir, "topology/physical_package_id");

        info.is_logic_core = !seen_cores.insert(x).second;

        if(!info.is_logic_core){
            num_cores_++;
            info.core_id = i;
            info.cpuid_id = i;
        }else{
            info.core_id = i % total_cpus; 
            info.cpuid_id = i;
        }

        result.push_back(info);

        if(info.physical_package_id > max_package_id){
            max_package_id = info.physical_package_id;
        }
    } 
    num_cores_ = num_cores_ * (max_package_id+1);
    return result;
}

Backend_NUMA::Backend_NUMA(int threads_per_node) {
    const char* env_threads = std::getenv("THREADS_PER_NODE");
    if (env_threads != nullptr) { 
        bool is_valid = true;
        for (const char* p = env_threads; *p != '\0'; ++p) {
            if (!std::isdigit(static_cast<unsigned char>(*p))) {
                is_valid = false;
                break;
            }
        }
        
        if (is_valid) {
            threads_per_node = std::atoi(env_threads);
            std::cout << "Using THREADS_PER_NODE from environment: " 
                      << threads_per_node << std::endl;
        }
    }
   
    num_cpus_ = numa_num_configured_cpus(); 
    num_cores_ = 0; 
    std::vector<CpuInfo> cpu_info = get_cpu_info(num_cores_);
    thread_per_node_ = std::min(threads_per_node, num_cpus_/numa_nodes_); 
    num_threads_ = numa_nodes_ * thread_per_node_;
     
    cpu_per_node_ = num_cpus_ / numa_nodes_;
    core_per_node_ = num_cores_ / numa_nodes_;
    
    thread_to_cpu_id.resize(num_threads_, -1);
    cpu_to_thread_id.resize(num_cpus_, -1);

    bool hyper_threading = num_cpus_ > num_cores_;
    for(int i=0; i< num_threads_; i++){
        int cpu_id = -1;
        int node_id = i / thread_per_node_;
        int offset = i % thread_per_node_;
        if(!hyper_threading){
            cpu_id = node_id * cpu_per_node_ + offset * cpu_per_node_ / thread_per_node_; 
        }else{ 
            cpu_id= node_id * core_per_node_ + offset % core_per_node_;
            if(offset >= core_per_node_){
                cpu_id = cpu_id + num_cores_;
            }
            
        }
        thread_to_cpu_id[i] = cpu_id;
        cpu_to_thread_id[cpu_id] = i; 
        std::cout << "Backend_NUMA init, thread_id: " << i << " cpu_id:" << cpu_id << " core_id:" << cpu_info[cpu_id].core_id << std::endl;
    }
    thread_state_.resize(num_cpus_);
    for (int i = 0; i < num_cpus_; i++) {
        A_ThreadState* state = new (std::align_val_t{64}) A_ThreadState(); 
        state->status.store(ThreadStatus::WAITING, std::memory_order_relaxed);
        state->curr.store(0, std::memory_order_relaxed);
        state->end = 0;
        thread_state_[i] = state;
    }
    
    workers_.resize(num_cpus_);
    for (int i = 0; i < num_cpus_; i++) {
        workers_[i] = std::thread(&Backend_NUMA::worker_thread, this, i);
    }
    std::cout << "Backend_NUMA init suscess, numa nodes: " << numa_nodes_ << " cors:" << num_cores_ << " cpus:" << num_cpus_ << " threads: " << num_threads_ << std::endl;
    if(hyper_threading){
        std::cout << "hyper_threading is using . "  << std::endl;
    }
}

Backend_NUMA::~Backend_NUMA() {
    for (int i = 0; i < num_cpus_; i++) {
        thread_state_[i]->status.store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 0; i < num_cpus_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }

    for (A_ThreadState* state : thread_state_) {
        state->~A_ThreadState();
        
        // 释放对齐内存
        #if __cpp_aligned_new >= 201606  // C++17 对齐 new
            operator delete(state, std::align_val_t{64});
        #else
            AlignedFree(state);  // 跨平台版本
        #endif
    }
    thread_state_.clear();
}

int Backend_NUMA::get_num_threads(){
    return num_threads_;
}

void Backend_NUMA::do_work(int m, std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
 
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func; 
 
    int base = m / thread_per_node_;
    int remain = m % thread_per_node_; 
    int begin = 0;
    int end = 0;     
    for (int i = 0; i < thread_per_node_; i++) {
        begin = end;
        end = begin + base + (i < remain);
        if(begin >= end) break;
        int cpu_id = thread_to_cpu_id[i];
        thread_state_[cpu_id]->end = end; 
        thread_state_[cpu_id]->curr.store(begin, std::memory_order_relaxed);
        thread_state_[cpu_id]->status.store(ThreadStatus::WORKING, std::memory_order_release);
    } 

    for (int i = 0; i < num_threads_; i++) {
        while (thread_state_[thread_to_cpu_id[i]]->status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
        }
    }
}


void Backend_NUMA::do_k_work_stealing_job(int k, int nth,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;

    int n_base = nth / numa_nodes_;
    int n_remain = nth % numa_nodes_;
    
    int begin = 0;
    int end = 0; 
    for(int i=0; i< numa_nodes_; i++){
        int local_nth = (n_base + (i < n_remain));   
        int num_node_tasks = local_nth *  k;
        if(num_node_tasks == 0) break; 
   
        int base = num_node_tasks / thread_per_node_;
        int remain = num_node_tasks % thread_per_node_;
        int node_base = i*thread_per_node_;
        for(int j=0; j< thread_per_node_; j++){
            begin = end;
            end = begin + base + (j < remain);
            if(begin >= end) break;
            int cpu_id = thread_to_cpu_id[node_base+j];
            thread_state_[cpu_id]->curr.store(begin, std::memory_order_relaxed);
            thread_state_[cpu_id]->end = end;
            thread_state_[cpu_id]->status.store(ThreadStatus::WORKING, std::memory_order_release);
        }
    }    

    for (int i = 0; i < num_threads_; i++) {
        while (thread_state_[thread_to_cpu_id[i]]->status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
        }
    }
}
  

void Backend_NUMA::process_tasks(int cpu_id) {

    
    int thread_id = cpu_to_thread_id[cpu_id];
    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    while (true) {
        int task_id = thread_state_[cpu_id]->curr.fetch_add(
            1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[cpu_id]->end) {
            break;
        }
        compute_func_(task_id);
    } 

    if(num_cpus_ > num_cores_){
        int begin = numa_node_ * core_per_node_ + num_cores_;
        int end =  begin + core_per_node_;
        for (int i = begin; i < end; i++) { 
            if (i == cpu_id) continue;
            if (thread_state_[cpu_id]->status.load(std::memory_order_acquire) !=
                ThreadStatus::WORKING) {
                continue;
            } 
            while (true) {
                int task_id = thread_state_[cpu_id]->curr.fetch_add(
                    1, std::memory_order_acq_rel);
                if (task_id >= thread_state_[cpu_id]->end) {
                    break;
                }
                compute_func_(task_id);
            } 
        } 
    }

    int begin = numa_node_ * core_per_node_;
    int end =  begin + core_per_node_;
    for (int i = begin; i < end; i++) { 
        if (i == cpu_id) continue;
        if (thread_state_[cpu_id]->status.load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            continue;
        } 
        while (true) {
            int task_id = thread_state_[cpu_id]->curr.fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[cpu_id]->end) {
                break;
            }
            compute_func_(task_id);
        } 
    }  
 

    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[cpu_id]->status.store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend_NUMA::worker_thread(int cpu_id) {
    auto start = std::chrono::steady_clock::now(); 
 
    if(cpu_id >= num_cores_) 
        numa_node_ = (cpu_id - num_cores_) / core_per_node_; 
    else
        numa_node_ = cpu_id / core_per_node_; 
  
    bind_to_cpu(cpu_id); 
    set_numa_mempolicy(numa_node_);
    
    while (true) {
        ThreadStatus status =
            thread_state_[cpu_id]->status.load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            process_tasks(cpu_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            auto now = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                      start)
                    .count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
} 


void bind_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity failed");
        exit(EXIT_FAILURE);
    }
}

void bind_to_numa_node(int node_id) { 
    struct bitmask *node_cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(node_id, node_cpus) != 0) {
        perror("Failed to get NUMA node CPUs");
        numa_free_cpumask(node_cpus);
        std::abort();
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
 
    for (unsigned int i = 0; i < node_cpus->size; ++i) {
        if (numa_bitmask_isbitset(node_cpus, i)) {
            CPU_SET(i, &cpuset);
        }
    }
    numa_free_cpumask(node_cpus);
 
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity failed"); 
    }
}
 
void set_numa_mempolicy(int node_id) {
    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, node_id);
 
    int policy = MPOL_BIND;

    if (set_mempolicy(policy, mask->maskp, mask->size) == -1) {
        std::cerr << "set_mempolicy failed for node " << node_id 
                  << ": " << errno << " (" << strerror(errno) << ")\n";
        std::abort();
    }
    numa_free_nodemask(mask);
}

void* allocate_aligned_numa(size_t size, int node, size_t* out_total_size = nullptr) {
    const size_t alignment = 64;
    size_t total_size = size + alignment - 1;
    void* raw_ptr = numa_alloc_onnode(total_size, node);
    if (!raw_ptr) return nullptr;
 
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
 
    uintptr_t* save_ptr = reinterpret_cast<uintptr_t*>(aligned_ptr) - 2;
    save_ptr[0] = reinterpret_cast<uintptr_t>(raw_ptr);  
    save_ptr[1] = total_size;                             

    if (out_total_size) *out_total_size = total_size;
    return aligned_ptr;
}
 
void free_aligned_numa(void* aligned_ptr) {
    if (!aligned_ptr) return;
 
    uintptr_t* save_ptr = reinterpret_cast<uintptr_t*>(aligned_ptr) - 2;
    void* raw_ptr = reinterpret_cast<void*>(save_ptr[0]);
    size_t total_size = save_ptr[1];

    numa_free(raw_ptr, total_size);
}

void* allocate_aligned(size_t size) {
    const size_t alignment = 64; 
    size_t total_size = size + alignment + sizeof(void*);
    void* raw_ptr = std::malloc(total_size);
    if (!raw_ptr) return nullptr;
 
    uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + sizeof(void*) + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
 
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    *prev_ptr = raw_ptr;

    return aligned_ptr;
}

void free_aligned(void* aligned_ptr, size_t size) {
    if (!aligned_ptr) return; 
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    void* raw_ptr = *prev_ptr;
    std::free(raw_ptr);
}

