//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_RENDERER_H
#define SPACE_EXPLORER_RENDERER_H

#include "core/fwd.h"
#include "rendering/fwd.h"
#include "window/gl_window.h"

namespace raw::rendering {
// I am not entirely sure what to put here, so I guess I just put something and then expend if
// needed
class renderer {
private:
	raw::window::gl_window window;
	graphics::gl_context_lock<graphics::context_type::MAIN> gl_context;

public:
	explicit renderer(const std::string &window_name = "Mike Hawk");

	[[nodiscard]] bool window_running() const noexcept;

	raw::window::gl_window *operator->();

	void render(raw::rendering::queue &command_queue, const core::camera::camera &camera) ;
};
} // namespace raw::rendering

#endif // SPACE_EXPLORER_RENDERER_H