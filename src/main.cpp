//
// Created by progamers on 6/2/25.
//
// Yes, I know, I am crazy, but I am still learning, so be less brutal at me pls.
#define STB_IMAGE_IMPLEMENTATION
#include <SDL3/SDL.h>
#include <ft2build.h>
#include <glad/glad.h>

#include <chrono>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <print>
#include <string>
#include <filesystem>
#include FT_FREETYPE_H

#include "button.h"
#include "camera.h"
#include "clock.h"
#include "gl_window.h"
#include "model.h"
#include "shader.h"
#include "stb_image.h"

PASSIVE_VALUE AM_POINT_LIGHTS = 4;

#define base_types(type, amount) std::vector<type>(amount, type(0.05))
namespace raw {
float calculate_phong_diffuse(const glm::vec3& light_dir, const glm::vec3& normal) {
	return fmax(0.0f, glm::dot(light_dir, glm::normalize(normal)));
}
// i remember days when i was learning linear algebra and i was like "what the crap matrices and vectors are"
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
	stbi_set_flip_vertically_on_load(true);

	// Данные для куба-источника света
	float light_cube_vertices[] = {-0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,	0.5f,  -0.5f,
								   0.5f,  0.5f,	 -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,

								   -0.5f, -0.5f, 0.5f,	0.5f,  -0.5f, 0.5f,	 0.5f,	0.5f,  0.5f,
								   0.5f,  0.5f,	 0.5f,	-0.5f, 0.5f,  0.5f,	 -0.5f, -0.5f, 0.5f,

								   -0.5f, 0.5f,	 0.5f,	-0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,
								   -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,	 -0.5f, 0.5f,  0.5f,

								   0.5f,  0.5f,	 0.5f,	0.5f,  0.5f,  -0.5f, 0.5f,	-0.5f, -0.5f,
								   0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,	 0.5f,	0.5f,  0.5f,

								   -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,	-0.5f, 0.5f,
								   0.5f,  -0.5f, 0.5f,	-0.5f, -0.5f, 0.5f,	 -0.5f, -0.5f, -0.5f,

								   -0.5f, 0.5f,	 -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,	0.5f,  0.5f,
								   0.5f,  0.5f,	 0.5f,	-0.5f, 0.5f,  0.5f,	 -0.5f, 0.5f,  -0.5f};

	float light_pos[] = {// positions in x y z coordinate system
						 2.5, 2.5, 5, -5, -5, 10, 0, -5, -5, -5, 5, 5};

	raw::gl_window window_mgr("Mike Hawk");
	raw::model backpack("assets/models/Eyeball/eyeball.obj");

	// I can't wait to just demolish all those things with my new progamers models
	unsigned int light_vao = 0;
	glGenVertexArrays(1, &light_vao);
	glBindVertexArray(light_vao);

	unsigned int light_vbo = 0;
	glGenBuffers(1, &light_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, light_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(light_cube_vertices), light_cube_vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	auto resolution = window_mgr.get_window_size();
	window_mgr.set_state(raw::gl::VIEW, 0, 0, resolution.x, resolution.y);

	bool	  running = true;
	SDL_Event event;

	window_mgr.set_state(raw::gl::CLEAR_COLOR, 0.0f, 0.0f, 0.0f, 0.0f);

	raw::shader light_shader("shaders/light/vertex_shader.glsl", "shaders/light/color_shader.frag");

	raw::camera camera;

	glm::mat4 model_transform	= glm::mat4(1.0f);
	glm::mat4 view_matrix		= camera.value();
	glm::mat4 projection_matrix = glm::mat4(1.0f);
	projection_matrix			= raw::perspective(
		  glm::radians(45.0f), static_cast<float>(resolution.x) / static_cast<float>(resolution.y),
		  0.1f, 100.0f);

	light_shader.use();
	light_shader.set_mat4("view", glm::value_ptr(view_matrix));
	light_shader.set_mat4("projection", glm::value_ptr(projection_matrix));
	light_shader.set_vec3("lightColor", 1, 1, 1);

	raw::shader model_shader("shaders/models/vertex_shader.glsl",
							 "shaders/models/color_shader.frag");
	model_shader.use();
	model_shader.set_int("obj_mat.diffuse_map", 0);
	model_shader.set_int("obj_mat.specular_map", 1);
	model_shader.set_float("obj_mat.shininess", 32.0f);
	model_shader.set_vec3("dir_light.direction", 0.0f, -1.0f, 0.0f);
	model_shader.set_vec3("dir_light.ambient", 0.1f, 0.1f, 0.1f);
	model_shader.set_vec3("dir_light.diffuse", 0.8f, 0.8f, 0.8f);
	model_shader.set_vec3("dir_light.specular", 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < AM_POINT_LIGHTS; ++i) {
		model_shader.set_vec3("point_lights[" + std::to_string(i) + "].position", light_pos[i * 3],
							  light_pos[i * 3 + 1], light_pos[i * 3 + 2]);
		model_shader.set_vec3("point_lights[" + std::to_string(i) + "].ambient", 0.05f, 0.05f,
							  0.05f);
		model_shader.set_vec3("point_lights[" + std::to_string(i) + "].diffuse", 0.8f, 0.8f, 0.8f);
		model_shader.set_vec3("point_lights[" + std::to_string(i) + "].specular", 1.0f, 1.0f, 1.0f);
		model_shader.set_float("point_lights[" + std::to_string(i) + "].constant", 1.0f);
		model_shader.set_float("point_lights[" + std::to_string(i) + "].linear", 0.09f);
		model_shader.set_float("point_lights[" + std::to_string(i) + "].quadratic", 0.032f);
	}
	model_shader.set_vec3("sp_light.ambient", 0.0f, 0.0f, 0.0f);
	model_shader.set_vec3("sp_light.diffuse", 1.0f, 1.0f, 1.0f);
	model_shader.set_vec3("sp_light.specular", 1.0f, 1.0f, 1.0f);
	model_shader.set_float("sp_light.cut_off", glm::cos(glm::radians(12.5f)));
	model_shader.set_float("sp_light.outer_cut_off", glm::cos(glm::radians(17.5f)));
	model_shader.set_float("sp_light.constant", 1.0f);
	model_shader.set_float("sp_light.linear", 0.09f);
	model_shader.set_float("sp_light.quadratic", 0.032f);

	model_shader.set_mat4("model", glm::value_ptr(model_transform));
	model_shader.set_mat4("view", glm::value_ptr(view_matrix));
	model_shader.set_mat4("projection", glm::value_ptr(projection_matrix));

	window_mgr.set_state(raw::gl::MOUSE_GRAB, window_mgr.get(), true);
	window_mgr.set_state(raw::gl::RELATIVE_MOUSE_MODE, window_mgr.get(), true);

	window_mgr.set_state(raw::gl::RULE, GL_DEPTH_TEST);
	window_mgr.set_state(raw::gl::ATTR, SDL_GL_MULTISAMPLEBUFFERS, 1);
	window_mgr.set_state(raw::gl::ATTR, SDL_GL_MULTISAMPLESAMPLES, 4);
	window_mgr.set_state(raw::gl::RULE, GL_MULTISAMPLE);
	window_mgr.set_state(raw::gl::CLEAR_COLOR, 1.0f, 0.0f, 0.0f, 0.0f);

	float yaw = -90.0f, pitch = 0.0f;

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
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto K_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto J_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto L_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto U_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto O_CALLBACK = [&object_quat, &delta_angle, &clock_callback]() {
			auto time_since_epoch = clock_callback.get_elapsed_time();
			time_since_epoch.to_milli();
			if (!(time_since_epoch < raw::time(1000 / updateMoveTime))) {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = object_quat * delta;
				clock_callback.restart();
			}
		};

		static auto T_CALLBACK = [&model_shader, &dir_light]() {
			model_shader.use();
			dir_light = true;
			model_shader.set_bool("need_dir_light", dir_light);
		};

		static auto T_RELEASE_CALLBACK = [&model_shader, &dir_light]() {
			model_shader.use();
			dir_light = false;
			model_shader.set_bool("need_dir_light", dir_light);
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
		while (window_mgr.poll_event(&event)) {
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
				light_shader.use();
				light_shader.set_mat4("projection", glm::value_ptr(projection_matrix));
				model_shader.use();
				model_shader.set_mat4("projection", glm::value_ptr(projection_matrix));
			}
		}
		for (auto& [key, button] : buttons) {
			button.update();
		}
		view_matrix = camera.value();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		light_shader.use();
		light_shader.set_mat4("view", glm::value_ptr(view_matrix));

		model_shader.use();
		model_shader.set_vec3("sp_light.position", camera.pos());
		model_shader.set_vec3("sp_light.direction", camera.front());
		model_shader.set_vec3("viewPos", camera.pos());
		model_shader.set_bool("need_dir_light", dir_light);
		model_shader.set_mat4("view", glm::value_ptr(view_matrix));

		glBindVertexArray(light_vao);
		light_shader.use();
		for (int i = 0; i < sizeof(light_pos) / sizeof(float) / 3; ++i) {
			glm::mat4 current_light_model_matrix = glm::mat4(1.0f);
			current_light_model_matrix			 = glm::translate(
				  current_light_model_matrix,
				  glm::vec3(light_pos[i * 3], light_pos[i * 3 + 1], light_pos[i * 3 + 2]));
			current_light_model_matrix = glm::scale(current_light_model_matrix, glm::vec3(0.2f));
			light_shader.set_mat4("model", glm::value_ptr(current_light_model_matrix));
			glDrawArrays(GL_TRIANGLES, 0, 36);
		}
		glBindVertexArray(0);

		glm::mat4 bugatti_model_transform = glm::mat4(1.0f);
		bugatti_model_transform =
			glm::translate(bugatti_model_transform, glm::vec3(0.0f, -1.0f, 0.0f));
		bugatti_model_transform = glm::scale(bugatti_model_transform, glm::vec3(0.02f));
		bugatti_model_transform = bugatti_model_transform * glm::mat4_cast(object_quat);

		model_shader.use();
		model_shader.set_mat4("model", glm::value_ptr(bugatti_model_transform));

		backpack.draw(model_shader);

		SDL_GL_SwapWindow(window_mgr.get());
	}

	glDeleteVertexArrays(1, &light_vao);
	glDeleteBuffers(1, &light_vbo);

	return 0;
}