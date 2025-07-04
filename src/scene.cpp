//
// Created by progamers on 7/4/25.
//
#include "scene.h"
namespace raw {
scene::scene(const std::string& window_name) : event_handler(), renderer(window_name), camera() {
    auto resolution = renderer.window.get_window_size();
	camera.set_window_resolution(resolution.x, resolution.y);
    event_handler.setup(this);
}
void scene::run() {
    while (renderer.window.running) {
        SDL_Event event;
        while (renderer.window.poll_event(&event)) {
            event_handler.handle(event);
        }
        event_handler._update();
        renderer.render();
    }
}

} // namespace raw