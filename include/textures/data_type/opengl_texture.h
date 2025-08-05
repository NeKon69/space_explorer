//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_VRAM_TEXTURE_H
#define SPACE_EXPLORER_VRAM_TEXTURE_H
#include "textures/fwd.h"
namespace raw::texture {
struct opengl_texture {
	// We use bindless textures so we gotta store the address
	std::pair<int /*texture id*/, uint64_t /* vram address */> albedo_metallic	   = {0, 0};
	std::pair<int, uint64_t>								   normal_roughness_ao = {0, 0};
};
} // namespace raw::texture

#endif // SPACE_EXPLORER_VRAM_TEXTURE_H
