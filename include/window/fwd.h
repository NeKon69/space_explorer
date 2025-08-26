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
        inline PASSIVE_VALUE &ATTR = SDL_GL_SetAttribute;
        inline PASSIVE_VALUE &RULE = glEnable;
        inline PASSIVE_VALUE &DISABLE = glDisable;
        inline PASSIVE_VALUE &VIEW = glViewport;
        inline PASSIVE_VALUE &MOUSE_GRAB = SDL_SetWindowMouseGrab;
        inline PASSIVE_VALUE &RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
        inline PASSIVE_VALUE &CLEAR_COLOR = glClearColor;
        inline PASSIVE_VALUE &DEPTH_FUNC = glDepthFunc;
        inline PASSIVE_VALUE &STENCIL_OPERATION = glStencilOp;
        inline PASSIVE_VALUE &STENCIL_MASK = glStencilMask;
        inline PASSIVE_VALUE &STENCIL_FUNC = glStencilFunc;
    } // namespace gl
    class sdl_video;
    class gl_window;
} // namespace raw

#endif // SPACE_EXPLORER_WINDOW_FWD_H