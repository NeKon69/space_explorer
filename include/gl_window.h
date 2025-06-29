//
// Created by progamers on 6/29/25.
//

#ifndef SPACE_EXPLORER_GL_WINDOW_H
#define SPACE_EXPLORER_GL_WINDOW_H

#include "window_manager.h"

namespace raw {
class gl_window final : public window_manager {
private:
	SDL_GLContextState* ctx = nullptr;
public:
	gl_window(std::string window_name);
	~gl_window() noexcept final;
};

} // namespace raw

#endif // SPACE_EXPLORER_GL_WINDOW_H
