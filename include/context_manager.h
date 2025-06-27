//
// Created by progamers on 6/26/25.
//

#ifndef SPACE_EXPLORER_CONTEXT_MANAGER_H
#define SPACE_EXPLORER_CONTEXT_MANAGER_H

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>

namespace raw {

 /** \brief
 * it would be smarter and more readable to create base class and derive 2 classes from it, BUT we have some hierarchy in which functions of destroying windows/contexts are called, so we'll stick with that approach.
 * doing that allows us to NEVER lose some not destroyed context, since the destructor calls will go like so -
 * window -> opengl context -> sdl_context.
 * it's not exactly fully correct order (which is EXACTLY why we need to have init/destroy functions), but it works, since we have in window destructor(the main class) correct calls - first opengl THEN window, and then SDL
 */

class ctx_sdl {
private:
    bool inited_sdl = false;
public:
    ctx_sdl() = default;
    void init();
    void destroy() noexcept;
    virtual ~ctx_sdl() noexcept;
};

class ctx_gl : public ctx_sdl {
private:
	SDL_GLContextState* ctx	   = nullptr;
	bool				inited_gl = false;

public:
    ctx_gl() = default;
	void init(SDL_Window* window);
    void destroy();
	virtual ~ctx_gl() noexcept;
};



} // namespace raw

#endif // SPACE_EXPLORER_CONTEXT_MANAGER_H
