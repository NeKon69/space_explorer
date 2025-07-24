//
// Created by progamers on 7/4/25.
//
#include "scene.h"
namespace raw {
scene::scene(const std::string& window_name) : event_handler(), renderer(window_name), camera() {
	manager.init();
	auto resolution = renderer.window.get_window_size();
	camera.set_window_resolution(resolution.x, resolution.y);
	event_handler.setup(this);
}

raw::rendering::queue scene::build_rendering_queue() const {
	raw::rendering::queue queue;
	for (raw::UI i = 0; i < 1; ++i) {
		raw::rendering::command cmd;
		cmd.shader	  = object_shader;
		cmd.mesh	  = sphere_mesh;
		cmd.inst_data = raw::rendering::instance_data(system.get_vbo(), system.amount());
	}
	return queue;
}

void scene::run() {
	while (renderer.window_running()) {
		manager.handle_events();
		auto commands = manager.get_commands();
		for (const auto& command : commands) {
			command->execute(*this);
		}
		system.update_sim();

		auto queue = build_rendering_queue();

		renderer.render(queue, camera);
	}
}

} // namespace raw