//
// Created by progamers on 7/6/25.
//

#ifndef SPACE_EXPLORER_OBJECT_INFO_H
#define SPACE_EXPLORER_OBJECT_INFO_H

#include "object.h"

namespace raw {
class object_info {
private:
    raw::shared_ptr<raw::object> object;
    glm::vec3					  position;
    // I don't like storing info about angles/scale, but we'll see will I use it or nah.
    glm::vec3					  rotation;
    glm::vec3					  scale;
public:
};
} // namespace raw

#endif // SPACE_EXPLORER_OBJECT_INFO_H
