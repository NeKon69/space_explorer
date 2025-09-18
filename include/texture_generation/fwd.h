//
// Created by progamers on 9/12/25.
//

#pragma once
#include <raw/unique_ptr.h>
#include <surface_types.h>

#include <glm/vec2.hpp>
#include <queue>
#include <tuple>

#include "common/scoped_resource_handle.h"
#include "deleters/custom_deleters.h"
#include "device_types/device_ptr.h"

namespace raw::texture_generation {
using planet_id = uint32_t;

class i_planet_resource_manager;

struct texture_data {
	cudaSurfaceObject_t albedo_metallic;
	// stores normal as 2 values (x, y) z is determined in the shader via some simple math
	cudaSurfaceObject_t normal_roughness_ao;
};

using texture_generation_data	 = std::tuple<device_types::device_ptr<cudaSurfaceObject_t>,
											  device_types::device_ptr<cudaSurfaceObject_t>>;

struct gl_textures {
	unique_ptr<uint32_t, deleters::gl_texture> albedo_metallic =
		unique_ptr<uint32_t, deleters::gl_texture>(new UI(0));
	unique_ptr<uint32_t, deleters::gl_texture> normal_roughness_ao =
		unique_ptr<uint32_t, deleters::gl_texture>(new UI(0));
};

struct texture_pool {
	std::vector<gl_textures> textures;
	std::queue<size_t>		 free_indices;
};

class i_planet_source {
public:
	virtual void prepare() = 0;
	virtual void cleanup() = 0;

public:
	virtual ~i_planet_source() = default;
};

namespace LOD {
static inline auto ULTRA = glm::uvec2(8192, 4096);
static inline auto HIGH	 = glm::uvec2(4096, 2048);
static inline auto MID	 = glm::uvec2(2048, 1024);
static inline auto LOW	 = glm::uvec2(512, 256);
static inline auto TINY	 = glm::uvec2(64, 32);
} // namespace LOD

enum class LOD_LEVEL { Ultra, High, Mid, Low, Tiny, COUNT };
static inline constexpr int NUM_LOD_LEVELS = std::to_underlying(LOD_LEVEL::COUNT);

inline glm::uvec2 get_lod_size(const LOD_LEVEL level) {
	switch (level) {
		using enum LOD_LEVEL;
	case Ultra:
		return LOD::ULTRA;
	case High:
		return LOD::HIGH;
	case Mid:
		return LOD::MID;
	case Low:
		return LOD::LOW;
	case Tiny:
		return LOD::TINY;
	default:
		throw std::invalid_argument("Invalid LOD_LEVEL");
	}
}
using texture_generation_context = common::scoped_resource_handle<i_planet_source>;
} // namespace raw::texture_generation