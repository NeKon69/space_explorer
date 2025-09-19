//
// Created by progamers on 9/19/25.
//

#pragma once
#include "components/component.h"
#include "rendering/fwd.h"

namespace raw::rendering {
struct visual_component : components::component {
	uint64_t albedo_metallic;
	uint64_t normal_roughness_ao;
	visual_component(visual_component&&)				 = default;
	visual_component& operator=(visual_component&&)		 = default;
	visual_component(const visual_component&)			 = default;
	visual_component& operator=(const visual_component&) = default;
};
} // namespace raw::rendering