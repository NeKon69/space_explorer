//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_WINDOW_FWD_H
#define SPACE_EXPLORER_WINDOW_FWD_H

#include <SDL3/SDL.h>
#include <glad/glad.h>

namespace raw {
namespace gl {
    // add more if you need
    // nah bro, after 10 billion years i decided this sucks, so i am gonna remove it... ahm, cry about
    // it.
    inline static constexpr auto &ATTR = SDL_GL_SetAttribute;
    inline static constexpr auto &RULE = glEnable;
    inline static constexpr auto &DISABLE = glDisable;
    inline static constexpr auto &VIEW = glViewport;
    inline static constexpr auto &MOUSE_GRAB = SDL_SetWindowMouseGrab;
    inline static constexpr auto &RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
    inline static constexpr auto &CLEAR_COLOR = glClearColor;
    inline static constexpr auto &DEPTH_FUNC = glDepthFunc;
    inline static constexpr auto &STENCIL_OPERATION = glStencilOp;
    inline static constexpr auto &STENCIL_MASK = glStencilMask;
    inline static constexpr auto &STENCIL_FUNC = glStencilFunc;
} // namespace gl
class sdl_video;
class gl_window;
} // namespace raw

#endif // SPACE_EXPLORER_WINDOW_FWD_H