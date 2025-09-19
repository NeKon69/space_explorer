//
// Created by progamers on 9/19/25.
//

#pragma once
#include "components/component.h"
#include "texture_generation/fwd.h"

namespace raw::texture_generation {
struct generation_recipe_component : components::component {
	float age;
	float tectonic_activity;
	float volcanic_activity;
	float crust_metallic_ratio;
	float crust_roughness_ratio;

	float axial_tilt;
	float base_temperature;
	float atmosphere_density;
	float atmosphere_opacity;
	float oxygen_level;
	float radiation_level;

	float hydrosphere_volume;
	float water_salinity;
	float polar_ice_caps_size;

	bool  has_vegetation;
	float vegetation_density;
	float vegetation_color_shift;

	bool  has_rings;
	bool  has_city_lights;
	float geo_luminescence;

	uint64_t seed;
};
} // namespace raw::texture_generation