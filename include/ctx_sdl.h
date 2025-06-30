//
// Created by progamers on 6/26/25.
//

#ifndef SPACE_EXPLORER_CTX_SDL_H
#define SPACE_EXPLORER_CTX_SDL_H

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <glm/glm.hpp>
#include <string>
#include <utility>

namespace raw {

class ctx_sdl {
public:
	ctx_sdl();
	virtual ~ctx_sdl() noexcept;

    ctx_sdl(const ctx_sdl&) = delete;
    ctx_sdl& operator=(const ctx_sdl&) = delete;
    ctx_sdl(ctx_sdl&&) = delete;
    ctx_sdl& operator=(ctx_sdl&&) = delete;
};

} // namespace raw

#endif // SPACE_EXPLORER_CTX_SDL_H
