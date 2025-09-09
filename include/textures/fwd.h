//
// Created by progamers on 8/5/25.
//

#pragma once

#include <cuda_gl_interop.h>

#include <glm/glm.hpp>

namespace raw::textures {
struct texture_slot;
class streaming_manager;
class manager;
template<typename K, typename V>
class lru_cache;

using planet_id = std::size_t;

struct data {
	cudaSurfaceObject_t albedo_metallic;
	// stores normal as 2 values (x, y) z is determined in the shader via some simple math
	cudaSurfaceObject_t normal_roughness_ao;
};

namespace size {
static constexpr auto ULTRA_MAX = glm::uvec2(8192, 4096);
static constexpr auto MAX		= glm::uvec2(4096, 2048);
static constexpr auto MID		= glm::uvec2(2048, 1024);
static constexpr auto LOW		= glm::uvec2(1024, 512);
static constexpr auto ULTRA_LOW = (256, 128);
static constexpr auto MIN		= glm::uvec2(64, 32);
} // namespace size
} // namespace raw::textures
namespace raw::textures {
struct pbr_material;
struct planet_dna;
} // namespace raw::textures
