//
// Created by progamers on 6/27/25.
//

#ifndef SPACE_EXPLORER_CAMERA_H
#define SPACE_EXPLORER_CAMERA_H

#include <glm/glm.hpp>

#include "../../cuda_types/error.h"

namespace raw::core {
    namespace predef {
        static constexpr auto CAMERA_POS = glm::vec3(0.0f, 0.0f, 5.0f);
        static constexpr auto CAMERA_FRONT = glm::vec3(0.0f, 0.0f, -1.0f);
        static constexpr auto CAMERA_UP = glm::vec3(0.0f, 1.0f, 0.0f);
        static constexpr auto SENSITIVITY = 0.1f;
        static constexpr auto CAMERA_SPEED = 0.05f;
        static constexpr auto FOV = 45.0f;
        static constexpr auto NEAR_PLANE = 0.1f;
        static constexpr auto FAR_PLANE = 1000.0f;
        static constexpr auto WINDOW_WIDTH = 2560.f;
        static constexpr auto WINDOW_HEIGHT = 1440.f;
        static constexpr auto ASPECT_RATIO = WINDOW_WIDTH / WINDOW_HEIGHT;
        static constexpr auto UPDATE_MOVE_TIME = 1000.f / 8.f;
    } // namespace predef

    namespace camera_move {
        void UP(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        void DOWN(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        void LEFT(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        void RIGHT(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        void FORWARD(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        void BACKWARD(glm::vec3 &pos, glm::vec3 front, glm::vec3 up);

        // Not defined, acts only as a type for function pointer
        void MOVE_FUNCTION(glm::vec3 &, glm::vec3, glm::vec3);
    } // namespace camera_move

    class camera {
    private:
        glm::vec3 camera_pos;
        glm::vec3 camera_front;
        glm::vec3 camera_up;

        float yaw = -90.f, pitch = 0.f;
        float fov = 0.f;
        // this member need to be manually set (I have the setter for it), since I don't really want to
        // somehow make window and camera depend on each other
        float window_aspect_ratio = predef::ASPECT_RATIO;

    public:
        // that constructor seems to be useless, since who in the right mind would create vec3's and
        // then pass it here, I think more logical would be to let the class handle all of it
        camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up);

        camera(glm::vec3 _pos, glm::vec3 _front, glm::vec3 _up, float _fov);

        camera();

        // basically same things just under different names
        [[nodiscard]] glm::mat4 view_projection() const;

        [[nodiscard]] glm::mat4 projection() const;

        [[nodiscard]] glm::mat4 view() const;

        /**
	 * @brief
	 * set new rotation for camera
	 * @param yaw absolute angle in degrees on y axis
	 * @param pitch absolute angle in degrees on x axis
	 */
        template<typename... Func>
        void set_rotation(float xoffset, float yoffset, Func &&... update_shader_uniforms) {
            yaw += xoffset;
            pitch -= yoffset;
            if (pitch > 89.0f) {
                pitch = 89.0f;
            }
            if (pitch < -89.0f) {
                pitch = -89.0f;
            }
            glm::vec3 front;
            front.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
            front.y = sinf(glm::radians(pitch));
            front.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
            camera_front = glm::normalize(front);
        }

        // again I come up with some stupid and at the same time genius ideas
        /**
	 * @brief
	 * move camera by the function parameter(you can pass any function specified in raw::camera_move
	 * namespace, or add one yourself)
	 * @param func function from raw::camera_move namespace
	 */
        template<typename... Func>
        void move(decltype(camera_move::MOVE_FUNCTION) func, Func &&... update_shader_uniforms) {
            func(camera_pos, camera_front, camera_up);
        }

        void move(const glm::vec3 &offset);

        template<typename... Func>
        void adjust_fov(float delta, Func &&... update_shader_uniforms) {
            fov += delta;
            if (fov < 1.0f) {
                fov = 1.0f;
            }
            if (fov > 180.0f) {
                fov = 180.0f;
            }
            std::cout << "[Debug] New FOV value: " << fov << "\n";
        }

        [[nodiscard]] inline glm::vec3 pos() const {
            return camera_pos;
        }

        [[nodiscard]] inline glm::vec3 front() const {
            return camera_front;
        }

        [[nodiscard]] inline glm::vec3 up() const {
            return camera_up;
        }

        [[nodiscard]] inline glm::vec3 right() const {
            return glm::cross(camera_front, camera_up);
        }

        void inline set_window_resolution(int x, int y) {
            window_aspect_ratio = static_cast<float>(x) / static_cast<float>(y);
        }
    };
} // namespace raw::core

#endif // SPACE_EXPLORER_CAMERA_H