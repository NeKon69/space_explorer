//
// Created by progamers on 9/12/25.
//
#pragma once
#include <unordered_map>

#include "texture_generation/cuda/fwd.h"
#include "texture_generation/i_planet_resource_manager.h"
#include "texture_generation/lru_cache.h"

namespace raw::texture_generation::cuda {
namespace detail {
static std::array<lru_cache<planet_id, planet_source>, NUM_LOD_LEVELS> create_lod_caches(
	const std::array<size_t, NUM_LOD_LEVELS>& pool_sizes, planet_resource_manager* manager);
static std::array<texture_pool, NUM_LOD_LEVELS> create_lod_pools(
	const std::array<size_t, NUM_LOD_LEVELS>& pool_sizes);
void setup_textures(uint32_t* texture, const glm::uvec2& texture_size, uint32_t quality,
					float max_anisotropy_level);
} // namespace detail
class planet_resource_manager : public i_planet_resource_manager {
private:
	// Structure Overview:
	//
	//                  ARRAY INDEX (LOD_LEVEL)
	//   +-------------------------------------------------+
	//   | 0: Ultra       | 1: High       | ... | 4: Tiny  |
	//   +-------------------------------------------------+
	//
	// Each entry in the array contains its own LRU cache:
	//
	//          LRU Cache for specific LOD_LEVEL (e.g. Ultra)
	//
	//       +-----------------------------------------------+
	//       |             planet_id --> planet_source       |
	//       |    +-----------+   +-----------+   +--------+ |
	//       |    | planet 1  |   | planet 2  |   |  ...   | |
	//       |    |   source  |   |   source  |   |        | |
	//       |    +-----------+   +-----------+   +--------+ |
	//       +-----------------------------------------------+
	//
	//
	// Access layout:
	//
	//                   ARRAY INDEX (LOD_LEVEL)
	//   +-------------------------------------------------------+
	//   |              LRU Cache for each LOD_LEVEL             |
	//   |   +-----------------------------------------------+   |
	//   |   | planet_id 1 | planet_id 2 | planet_id 3 | ... |   |
	//   |   +-----------+ +-----------+ +-----------+ +-----+   |
	//   |    planet_source  planet_source  planet_source        |
	//   +-------------------------------------------------------+
	//
	// Summary:
	//
	// lod_caches[0]  --> LRU cache with {planet_id -> planet_source} for Ultra LOD
	// lod_caches[1]  --> LRU cache with {planet_id -> planet_source} for High LOD
	// ...
	// lod_caches[4]  --> LRU cache with {planet_id -> planet_source} for Tiny LOD

	std::array<lru_cache<planet_id, planet_source>, NUM_LOD_LEVELS> lod_caches;
	// This is allocated at the creation of this class and lasts until class is deleted
	std::array<texture_pool, NUM_LOD_LEVELS> lod_pools;

protected:
	void prepare(planet_id id, LOD_LEVEL lod_level) override;
	void cleanup() override;

public:
	std::array<texture_pool, NUM_LOD_LEVELS>& get_pools();
	planet_resource_manager(std::array<size_t, NUM_LOD_LEVELS> pool_size);
	texture_generation_data	   get_data(planet_id planet_id, LOD_LEVEL lod_level) override;
	texture_generation_context create_context(planet_id id, LOD_LEVEL lod_level) override;
};
} // namespace raw::texture_generation::cuda