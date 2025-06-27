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

class ctx_sdl {
private:
    bool inited_sdl = false;
public:
    ctx_sdl() = default;
    void init();
    void destroy() noexcept;
    virtual ~ctx_sdl() noexcept;
};

// it would be smarter and more readable to create base class and derive 2 classes from it, BUT we have some hierarchy in which functions of destroying windows are called, so we'll stick with that approach
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
