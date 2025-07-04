//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_SCENE_H
#define SPACE_EXPLORER_SCENE_H

#include <glm/glm.hpp>

#include "camera.h"
#include "renderer.h"
#include "event_handler.h"

namespace raw {
class scene {
private:
	raw::event_handler	event_handler;
    raw::renderer		renderer;
	raw::camera camera;
	glm::mat4	view_matrix;
	glm::mat4	projection_matrix;
    friend class event_handler;
    friend class renderer;
public:
    explicit scene(const std::string& window_name = "Mike Hawk");
    void run();
};

} // namespace raw
#endif // SPACE_EXPLORER_SCENE_H
