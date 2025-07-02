//
// Created by progamers on 6/2/25.
//
// Yes, I know, I am crazy, but I am still learning, so be less brutal at me pls.
#define STB_IMAGE_IMPLEMENTATION
#include <SDL3/SDL.h>
#include <ft2build.h>
#include <glad/glad.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <print>
#include <span>
#include <string>
#include FT_FREETYPE_H

#include "button.h"
#include "camera.h"
#include "clock.h"
#include "gl_window.h"
#include "objects/cube.h"
#include "shader.h"

PASSIVE_VALUE AM_POINT_LIGHTS = 4;

#define base_types(type, amount) std::vector<type>(amount, type(0.05))
namespace raw {
float calculate_phong_diffuse(const glm::vec3& light_dir, const glm::vec3& normal) {
	return fmax(0.0f, glm::dot(light_dir, glm::normalize(normal)));
}
// i remember days when i was learning linear algebra and i was like "what the crap matrices and
// vectors are"
glm::vec3 calculate_normal(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
	glm::vec3 edge1	 = v2 - v1;
	glm::vec3 edge2	 = v3 - v1;
	glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
	return normal;
}

glm::mat4 translate(const glm::vec3& offset) {
	glm::mat4 base = glm::mat4(1.0f);
	base[3][0]	   = offset.x;
	base[3][1]	   = offset.y;
	base[3][2]	   = offset.z;
	return base;
}
glm::mat4 rotate_y(float angle) {
	glm::mat4 base = glm::mat4(1.0f);
	base[0][0]	   = cosf(angle);
	base[2][0]	   = -sinf(angle);
	base[0][2]	   = sinf(angle);
	base[2][2]	   = cosf(angle);
	return base;
}
glm::mat4 scale(const glm::vec3& factors) {
	glm::mat4 base = glm::mat4(1.0f);
	for (int i = 0; i < 3; ++i) {
		base[i][i] = factors[i];
	}
	return base;
}

glm::mat4 look_at(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
	glm::mat4 base	 = glm::mat4(1.0f);
	glm::vec3 z_axis = glm::normalize(eye - center);
	glm::vec3 x_axis = glm::normalize(glm::cross(up, z_axis));
	glm::vec3 y_axis = glm::normalize(glm::cross(z_axis, x_axis));

	base[0][0] = x_axis.x;
	base[0][1] = x_axis.y;
	base[0][2] = x_axis.z;

	base[1][0] = y_axis.x;
	base[1][1] = y_axis.y;
	base[1][2] = y_axis.z;

	base[2][0] = z_axis.x;
	base[2][1] = z_axis.y;
	base[2][2] = z_axis.z;

	base[3][0] = eye.x;
	base[3][1] = eye.y;
	base[3][2] = eye.z;

	return glm::inverse(base);
}

glm::mat4 perspective(float fov_rad, float aspect, float near, float far) {
	float	  fov_scale = 1 / tanf(fov_rad / 2);
	glm::mat4 base(1.0f);

	base[0][0] = fov_scale / aspect;
	base[1][1] = fov_scale;
	base[2][2] = -(far + near) / (far - near);
	base[2][3] = -1;
	base[3][2] = -(2 * far * near) / (far - near);
	base[3][3] = 0;
	return base;
}

} // namespace raw
template<typename mat>
// I guess no more bit shifting string to kout (cout)
void print_matrix(mat matrix) {
	auto size = matrix.length();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			std::print("{}\t", matrix[i][j]);
		}
		std::print("\n");
	}
}

int main(int argc, char* argv[]) {
	bool										  dir_light = true;
	raw::clock									  clock;
	raw::clock									  clock_callback;
	std::unordered_map<SDL_Scancode, raw::button> buttons;

	float light_pos[] = {// positions in x y z coordinate system
						 2.5, 2.5, 5, -5, -5, 10, 0, -5, -5, -5, 5, 5};

	glm::vec3 cube_positions[] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 5.0f, -15.0f)};

	raw::gl_window window("Mike Hawk");

    raw::gl::RULE(GL_DEPTH_TEST);
	raw::gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	raw::gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 4);
	raw::gl::RULE(GL_MULTISAMPLE);

	auto resolution = window.get_window_size();
	raw::gl::VIEW(0, 0, resolution.x, resolution.y);

	bool	  running = true;
	SDL_Event event;

	raw::gl::CLEAR_COLOR(0.0f, 0.0f, 0.0f, 0.0f);

	raw::shared_ptr<raw::shader> light_shader(
		new raw::shader("shaders/light/vertex_shader.glsl", "shaders/light/color_shader.frag"));

	raw::camera camera;

	glm::mat4 model_transform	= glm::mat4(1.0f);
	glm::mat4 view_matrix		= camera.value();
	glm::mat4 projection_matrix = glm::mat4(1.0f);
	projection_matrix			= raw::perspective(
		  glm::radians(45.0f), static_cast<float>(resolution.x) / static_cast<float>(resolution.y),
		  0.1f, 100.0f);

	light_shader->use();
	light_shader->set_mat4("view", glm::value_ptr(view_matrix));
	light_shader->set_mat4("projection", glm::value_ptr(projection_matrix));
	light_shader->set_vec3("lightColor", 1, 1, 1);

	raw::shared_ptr<raw::shader> object_shader(
		new raw::shader("shaders/objects/vertex_shader.glsl", "shaders/objects/color_shader.frag"));
	object_shader->use();
	object_shader->set_float("obj_mat.shininess", 32.0f);
	object_shader->set_vec3("dir_light.direction", 0.0f, -1.0f, 0.0f);
	object_shader->set_vec3("dir_light.ambient", 0.1f, 0.1f, 0.1f);
	object_shader->set_vec3("dir_light.diffuse", 0.8f, 0.8f, 0.8f);
	object_shader->set_vec3("dir_light.specular", 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < AM_POINT_LIGHTS; ++i) {
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].position", light_pos[i * 3],
							   light_pos[i * 3 + 1], light_pos[i * 3 + 2]);
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].ambient", 0.05f, 0.05f,
							   0.05f);
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].diffuse", 0.8f, 0.8f, 0.8f);
		object_shader->set_vec3("point_lights[" + std::to_string(i) + "].specular", 1.0f, 1.0f,
							   1.0f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].constant", 1.0f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].linear", 0.09f);
		object_shader->set_float("point_lights[" + std::to_string(i) + "].quadratic", 0.032f);
	}
	object_shader->set_vec3("sp_light.ambient", 0.0f, 0.0f, 0.0f);
	object_shader->set_vec3("sp_light.diffuse", 1.0f, 1.0f, 1.0f);
	object_shader->set_vec3("sp_light.specular", 1.0f, 1.0f, 1.0f);
	object_shader->set_float("sp_light.cut_off", glm::cos(glm::radians(12.5f)));
	object_shader->set_float("sp_light.outer_cut_off", glm::cos(glm::radians(17.5f)));
	object_shader->set_float("sp_light.constant", 1.0f);
	object_shader->set_float("sp_light.linear", 0.09f);
	object_shader->set_float("sp_light.quadratic", 0.032f);

	object_shader->set_mat4("view", glm::value_ptr(view_matrix));
	object_shader->set_mat4("projection", glm::value_ptr(projection_matrix));

	object_shader->set_vec3("obj_mat.ambient", 0.1f, 0.1f, 0.1f);
	object_shader->set_vec3("obj_mat.diffuse", 1.0f, 0.5f, 0.31f);
	object_shader->set_vec3("obj_mat.specular", 0.5f, 0.5f, 0.5f);

	raw::gl::MOUSE_GRAB(window.get(), true);
	raw::gl::RELATIVE_MOUSE_MODE(window.get(), true);

	raw::gl::RULE(GL_DEPTH_TEST);
	raw::gl::ATTR(SDL_GL_MULTISAMPLEBUFFERS, 1);
	raw::gl::ATTR(SDL_GL_MULTISAMPLESAMPLES, 4);
	raw::gl::RULE(GL_MULTISAMPLE);
	raw::gl::CLEAR_COLOR(0.1f, 0.1f, 0.1f, 1.0f);

	raw::cube cube(object_shader);
	raw::cube light_cube(light_shader);

	float yaw	= -90.0f;
	float pitch = 0.0f;

	constexpr long updateMoveTime = 144;
	auto		   start		  = std::chrono::high_resolution_clock::now();
	auto		   end			  = std::chrono::high_resolution_clock::now();

	glm::quat object_quat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	float	  delta_angle = 1.0f;
	{
		static const auto TAB_CALLBACK	 = std::function([&camera, &clock_callback]() {
			  auto time_since_epoch = clock_callback.get_elapsed_time();
			  time_since_epoch.to_milli();
			  if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				  camera.move(raw::camera_move::DOWN);
				  clock_callback.restart();
			  }
		  });
		static const auto SPACE_CALLBACK = std::function([&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::UP);
				clock_callback.restart();
			}
		});
		static const auto LEFT_CALLBACK	 = std::function([&camera, &clock_callback]() {
			 auto time_since_epoch = clock_callback.get_elapsed_time();
			 time_since_epoch.to_milli();
			 if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				 camera.move(raw::camera_move::LEFT);
				 clock_callback.restart();
			 }
		 });
		static auto		  RIGHT_CALLBACK = [&camera, &clock_callback]() {
			  auto time_since_epoch = clock_callback.get_elapsed_time();
			  time_since_epoch.to_milli();
			  if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				  camera.move(raw::camera_move::RIGHT);
				  clock_callback.restart();
			  }
		};

		static auto UP_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::FORWARD);
				clock_callback.restart();
			}
		};

		static auto DOWN_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::BACKWARD);
				clock_callback.restart();
			}
		};

		static auto W_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::FORWARD);
				clock_callback.restart();
			}
		};

		static auto A_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::LEFT);
				clock_callback.restart();
			}
		};

		static auto S_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::BACKWARD);
				clock_callback.restart();
			}
		};

		static auto D_CALLBACK = [&camera, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				camera.move(raw::camera_move::RIGHT);
				clock_callback.restart();
			}
		};

		static auto I_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto K_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto J_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto L_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto U_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto O_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = delta * object_quat;
				clock_callback.restart();
			}
		};

		static auto T_CALLBACK = [&object_shader, &dir_light]() {
			object_shader->use();
			dir_light = true;
			object_shader->set_bool("need_dir_light", dir_light);
		};

		static auto T_RELEASE_CALLBACK = [&object_shader, &dir_light]() {
			object_shader->use();
			dir_light = false;
			object_shader->set_bool("need_dir_light", dir_light);
		};

		static auto ESCAPE_CALLBACK = [&running]() {
			std::cout << "Don't close me mf!" << std::endl;
			running = false;
		};

		buttons[SDL_SCANCODE_TAB]	 = raw::button(TAB_CALLBACK);
		buttons[SDL_SCANCODE_SPACE]	 = raw::button(SPACE_CALLBACK);
		buttons[SDL_SCANCODE_LEFT]	 = raw::button(LEFT_CALLBACK);
		buttons[SDL_SCANCODE_RIGHT]	 = raw::button(RIGHT_CALLBACK);
		buttons[SDL_SCANCODE_UP]	 = raw::button(UP_CALLBACK);
		buttons[SDL_SCANCODE_DOWN]	 = raw::button(DOWN_CALLBACK);
		buttons[SDL_SCANCODE_W]		 = raw::button(W_CALLBACK);
		buttons[SDL_SCANCODE_A]		 = raw::button(A_CALLBACK);
		buttons[SDL_SCANCODE_S]		 = raw::button(S_CALLBACK);
		buttons[SDL_SCANCODE_D]		 = raw::button(D_CALLBACK);
		buttons[SDL_SCANCODE_I]		 = raw::button(I_CALLBACK);
		buttons[SDL_SCANCODE_K]		 = raw::button(K_CALLBACK);
		buttons[SDL_SCANCODE_J]		 = raw::button(J_CALLBACK);
		buttons[SDL_SCANCODE_L]		 = raw::button(L_CALLBACK);
		buttons[SDL_SCANCODE_U]		 = raw::button(U_CALLBACK);
		buttons[SDL_SCANCODE_O]		 = raw::button(O_CALLBACK);
		buttons[SDL_SCANCODE_T]		 = raw::button(T_CALLBACK, T_RELEASE_CALLBACK);
		buttons[SDL_SCANCODE_ESCAPE] = raw::button(ESCAPE_CALLBACK);
	}
	float fov = 45.0f;
	while (running) {
		while (window.poll_event(&event)) {
			if (event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			} else if (event.type == SDL_EVENT_KEY_DOWN) {
				if (buttons.contains(event.key.scancode)) {
					buttons[event.key.scancode].press();
				}
			} else if (event.type == SDL_EVENT_MOUSE_MOTION) {
				float xoffset = event.motion.xrel;
				float yoffset = event.motion.yrel;

				xoffset *= raw::predef::SENSITIVITY;
				yoffset *= raw::predef::SENSITIVITY;

				yaw += xoffset;
				pitch -= yoffset;

				camera.rotate(yaw, pitch);
			} else if (event.type == SDL_EVENT_KEY_UP) {
				if (buttons.contains(event.key.scancode)) {
					buttons[event.key.scancode].release();
				}
			} else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
				event.wheel.y < 0 ? fov += 1.0f : fov -= 1.0f;
				if (fov < 1)
					fov = 1.0f;
				if (fov > 180.0f)
					fov = 180.0f;
				projection_matrix = glm::perspective(
					glm::radians(fov), resolution.x / float(resolution.y), 0.1f, 100.0f);
				light_shader->use();
				light_shader->set_mat4("projection", glm::value_ptr(projection_matrix));
				object_shader->use();
				object_shader->set_mat4("projection", glm::value_ptr(projection_matrix));
			}
		}
		for (auto& [key, button] : buttons) {
			button.update();
		}
		view_matrix = camera.value();

		window.clear();

		object_shader->use();
		object_shader->set_vec3("sp_light.position", camera.pos());
		object_shader->set_vec3("sp_light.direction", camera.front());
		object_shader->set_vec3("viewPos", camera.pos());
		object_shader->set_bool("need_dir_light", dir_light);
		object_shader->set_mat4("view", glm::value_ptr(view_matrix));

		light_shader->use();
		light_shader->set_mat4("view", glm::value_ptr(view_matrix));

		// render cubes
		for (unsigned int i = 0; i < 2; i++) {
			cube.move(cube_positions[i]);
			cube.draw();
		}

		// render platform
		cube.move(glm::vec3(0.0f, -2.0f, 0.0f));
		cube.scale(glm::vec3(15.0f, 0.2f, 15.0f));
		cube.draw();

		for (int i = 0; i < std::size(light_pos) / 3; ++i) {
			light_cube.move(
				glm::vec3(light_pos[i * 3], light_pos[i * 3 + 1], light_pos[i * 3 + 2]));
			light_cube.scale(glm::vec3(0.2f));
			light_cube.draw();
		}
		SDL_GL_SwapWindow(window.get());
	}
	return 0;
}