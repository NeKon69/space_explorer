//
// Created by progamers on 9/17/25.
//
// clang-format off
#include <glad/glad.h>
#include <GL/glext.h>
#include "texture_generation/cuda/planet_resource_manager.h"

// clang-format on

namespace raw::texture_generation::cuda {
std::array<texture_pool, NUM_LOD_LEVELS>& planet_resource_manager::get_pools() {
	return lod_pools;
}

auto detail::create_lod_caches(const std::array<size_t, NUM_LOD_LEVELS>& pool_sizes,
							   planet_resource_manager*					 manager)
	-> std::array<lru_cache<planet_id, planet_source>, NUM_LOD_LEVELS> {
	return [&] {
		std::array<lru_cache<planet_id, planet_source>, NUM_LOD_LEVELS> caches_temp;

		for (auto i = 0; i < NUM_LOD_LEVELS; ++i) {
			caches_temp[i] = lru_cache<planet_id, planet_source>(
				pool_sizes[i], [manager, i](const planet_id& id, const planet_source& binding) {
					manager->get_pools()[i].free_indices.push(binding.pool_index);
				});
		}
		return caches_temp;
	}();
}

void detail::setup_textures(uint32_t* texture, const glm::uvec2& texture_size, uint32_t quality,
							float max_anisotropy_level) {
	glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, quality, texture_size.x, texture_size.y, 0, GL_RGBA,
				 GL_UNSIGNED_BYTE, nullptr);
	glGenerateTextureMipmap(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_anisotropy_level);
}
auto detail::create_lod_pools(const std::array<size_t, NUM_LOD_LEVELS>& pool_sizes)
	-> std::array<texture_pool, NUM_LOD_LEVELS> {
	return [&]() {
		std::array<texture_pool, NUM_LOD_LEVELS> pools;
		for (int i = 0; i < NUM_LOD_LEVELS; i++) {
			pools[i].textures.resize(pool_sizes[i]);
			for (size_t j = 0; j < pool_sizes[i]; j++) {
				auto&	   current_textures = pools[i].textures[j];
				const auto resolution		= get_lod_size(static_cast<LOD_LEVEL>(i));
				GLfloat	   max_anisotropy;
				glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy);
				setup_textures(current_textures.albedo_metallic.get(), resolution, GL_SRGB8_ALPHA8,
							   max_anisotropy);
				setup_textures(current_textures.normal_roughness_ao.get(), resolution,
							   GL_RGBA16_SNORM, max_anisotropy);
				glBindTexture(GL_TEXTURE_2D, 0);
			}
		}
		return pools;
	}();
}
planet_resource_manager::planet_resource_manager(std::array<size_t, NUM_LOD_LEVELS> pool_sizes)
	: lod_caches(detail::create_lod_caches(pool_sizes, this)),
	  lod_pools(detail::create_lod_pools(pool_sizes)) {}

void planet_resource_manager::prepare(planet_id id, LOD_LEVEL lod_level) {
	lod_caches[std::to_underlying(lod_level)].put(id, planet_source());
}

texture_generation_context planet_resource_manager::create_context(planet_id id,
																   LOD_LEVEL lod_level) {
	return texture_generation_context(this, id, lod_level);
}

} // namespace raw::texture_generation::cuda