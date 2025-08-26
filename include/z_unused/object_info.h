//
// Created by progamers on 7/6/25.
//

#ifndef SPACE_EXPLORER_OBJECT_INFO_H
#define SPACE_EXPLORER_OBJECT_INFO_H

#include "object.h"

namespace raw {
enum class param_type { POS, SCALE, ROTATION };
class object_info {
private:
	raw::shared_ptr<raw::object> object;
	glm::mat4					 transform = glm::mat4(1.0);
public:
	object_info() = default;
	object_info(raw::shared_ptr<raw::object> obj);
	object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position);
	object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position, glm::vec3 scale);
	object_info(raw::shared_ptr<raw::object> obj, glm::vec3 position, glm::vec3 scale,
				glm::vec3 rotation, float degree);
	object_info(raw::shared_ptr<raw::object> obj, glm::mat4 transformation);
	[[nodiscard]] inline raw::shared_ptr<raw::object> get_object() const {
		return object;
	}
	[[nodiscard]] inline glm::mat4 get_transform() const {
		return transform;
	}
};
} // namespace raw

#endif // SPACE_EXPLORER_OBJECT_INFO_H
