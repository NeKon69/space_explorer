//
// Created by progamers on 8/5/25.
//

#pragma once
#include <glad/glad.h>
#include <raw_memory.h>

#include "../texture_generation/lru_cache.h"
#include "deleters/custom_deleters.h"
#include "device_types/cuda/from_gl/image.h"
#include "textures/data_type/compressed_cpu_texture.h"
#include "textures/manager.h"

namespace raw::textures {
struct texture_slot {
	// am is simplified from albedo_metallic
	device_types::cuda::from_gl::image albedo_metallic;
	device_types::cuda::from_gl::image normal_rough_ao;

	raw::unique_ptr<unsigned int, raw::deleters::gl_texture> am_id;
	raw::unique_ptr<unsigned int, raw::deleters::gl_texture> nrao_id;
	bool													 is_in_use			= false;
	planet_id												 assigned_planet_id = 0;

	texture_slot() : am_id(new UI(0)), nrao_id(new UI(0)) {};

	void create() {
		glGenTextures(1, am_id.get());
		glGenTextures(1, nrao_id.get());
		albedo_metallic.set_data(*am_id);
		normal_rough_ao.set_data(*nrao_id);
		is_in_use = true;
	}

	texture_slot(int alb_met, int norm_rough_ao, planet_id id);

	// : albedo_metallic(alb_met),
	// normal_rough_ao(norm_rough_ao),
	// am_id(new UI(0)),
	// nrao_id(new UI(0)),
	// is_in_use(true),
	// assigned_planet_id(id) {}
};

/**
 * @class streaming_manger - provides nice interface to handle all the hard things with texture
 * streaming
 * @brief everything besides recreating it should work asynchronously in the app to not stop the
 * main rendering thread
 */
class streaming_manager {
private:
	std::vector<texture_slot> texture_pool;
	// Technically storing this should be fine since we only change vector capacity once
	// And even if we want to recreate some textures we would need to either completely destroy
	// instance of an object Or create out own function member that can be called when we want to
	// recreate
	raw::textures::lru_cache<planet_id, typename decltype(texture_pool)::iterator> gpu_cache;
	raw::textures::lru_cache<planet_id, data_type::compressed_cpu_texture>		   cpu_cache;
	raw::textures::manager														   mgr;

	void create(uint32_t gpu_cache_size, uint32_t cpu_cache_size);

public:
	streaming_manager(uint32_t gpu_cache_size, uint32_t cpu_cache_size);
};
} // namespace raw::textures
