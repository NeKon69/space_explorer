//
// Created by progamers on 8/5/25.
//

#pragma once

#include <glm/glm.hpp>

namespace raw::textures {
struct planet_dna {
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
};

struct pbr_material {
	glm::vec3 albedo;
	glm::vec3 normal;
	u_char	  roughness;
	u_char	  metallic;
	u_char	  ao;
	u_char	  emissive;
};
} // namespace raw::textures
