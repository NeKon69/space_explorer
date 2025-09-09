//
// Created by progamers on 6/26/25.
//

#pragma once

#include "window/fwd.h"

namespace raw::window {
class sdl_video {
public:
	sdl_video();

	~sdl_video() noexcept;

	sdl_video(const sdl_video &) = delete;

	sdl_video &operator=(const sdl_video &) = delete;
};
} // namespace raw::window

