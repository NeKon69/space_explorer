//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_RENDERER_H
#define SPACE_EXPLORER_RENDERER_H

#include <raw_memory.h>

#include "camera.h"
#include "gl_window.h"
#include "render_command.h"
namespace raw::rendering {
// I am not entirely sure what to put here, so I guess I just put something and then expend if
// needed
class renderer {
private:
	raw::gl_window window;

public:
	explicit renderer(const std::string& window_name = "Mike Hawk");
	[[nodiscard]] bool window_running() const noexcept;
	raw::gl_window*	   operator->();
	void render(raw::rendering::queue& command_queue, const raw::camera& camera) const;
};

} // namespace raw::rendering

#endif // SPACE_EXPLORER_RENDERER_H
