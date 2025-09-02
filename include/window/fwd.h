//
// Created by progamers on 8/26/25.
//

#ifndef SPACE_EXPLORER_WINDOW_FWD_H
#define SPACE_EXPLORER_WINDOW_FWD_H

#include <SDL3/SDL.h>
#include <glad/glad.h>

/**
 * Convenience aliases for SDL and OpenGL function calls used by the windowing layer.
 *
 * Each identifier is a constexpr reference alias to the corresponding SDL/OpenGL function,
 * provided so call sites can use short, consistent names for GL/SDL window operations.
 * These aliases do not change semantics of the underlying functions.
 */

/**
 * Alias for SDL_GL_SetAttribute.
 */
 
/**
 * Alias for glEnable.
 */

/**
 * Alias for glDisable.
 */

/**
 * Alias for glViewport.
 */

/**
 * Alias for SDL_SetWindowMouseGrab.
 */

/**
 * Alias for SDL_SetWindowRelativeMouseMode.
 */

/**
 * Alias for glClearColor.
 */

/**
 * Alias for glDepthFunc.
 */

/**
 * Alias for glStencilOp.
 */

/**
 * Alias for glStencilMask.
 */

/**
 * Alias for glStencilFunc.
 */

/**
 * Forward declaration for graphics_data, which holds renderer/window-related resources.
 */
 
/**
 * Forward declaration for sdl_video, the platform video subsystem wrapper.
 */

/**
 * Forward declaration for gl_window, an OpenGL-backed window abstraction.
 */
namespace raw::window {
namespace gl {
// add more if you need
// nah bro, after 10 billion years i decided this sucks, so i am gonna remove it... ahm, cry about
// it.
inline static constexpr auto &ATTR				  = SDL_GL_SetAttribute;
inline static constexpr auto &RULE				  = glEnable;
inline static constexpr auto &DISABLE			  = glDisable;
inline static constexpr auto &VIEW				  = glViewport;
inline static constexpr auto &MOUSE_GRAB		  = SDL_SetWindowMouseGrab;
inline static constexpr auto &RELATIVE_MOUSE_MODE = SDL_SetWindowRelativeMouseMode;
inline static constexpr auto &CLEAR_COLOR		  = glClearColor;
inline static constexpr auto &DEPTH_FUNC		  = glDepthFunc;
inline static constexpr auto &STENCIL_OPERATION	  = glStencilOp;
inline static constexpr auto &STENCIL_MASK		  = glStencilMask;
inline static constexpr auto &STENCIL_FUNC		  = glStencilFunc;
} // namespace gl

struct graphics_data;
class sdl_video;
class gl_window;
} // namespace raw::window

#endif // SPACE_EXPLORER_WINDOW_FWD_H