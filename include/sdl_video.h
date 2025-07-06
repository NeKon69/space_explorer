//
// Created by progamers on 6/26/25.
//

#ifndef SPACE_EXPLORER_SDL_VIDEO_H
#define SPACE_EXPLORER_SDL_VIDEO_H

#include <SDL3/SDL.h>

#include <stdexcept>

namespace raw {

class sdl_video {
public:
	sdl_video();
	~sdl_video() noexcept;

	sdl_video(const sdl_video&)			   = delete;
	sdl_video& operator=(const sdl_video&) = delete;
};

} // namespace raw

#endif // SPACE_EXPLORER_SDL_VIDEO_H
