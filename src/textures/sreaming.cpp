//
// Created by progamers on 8/5/25.
//
#include "textures/streaming.h"

namespace raw::textures {
    streaming_manager::streaming_manager(uint32_t gpu_cache_size, uint32_t cpu_cache_size)
        : gpu_cache(gpu_cache_size), cpu_cache(cpu_cache_size) {
        create(gpu_cache_size, cpu_cache_size);
    }

    void streaming_manager::create(uint32_t gpu_cache_size, uint32_t cpu_cache_size) {
        texture_pool.resize(gpu_cache_size);
        for (auto &slot: texture_pool) {
            slot.create();
        }
    }
} // namespace raw::texture