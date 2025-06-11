//
// Created by progamers on 6/2/25.
//
// Yes, I know, I am crazy, but I am still learning, so be less brutal at me pls.
#define STB_IMAGE_IMPLEMENTATION
#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
#include <iostream>
#include <memory>

#include "shader.h"
#include "stb_image.h"

namespace raw {
float calculate_phong_diffuse(const glm::vec3& light_dir, const glm::vec3& normal) {
	return fmax(0.0f, glm::dot(light_dir, glm::normalize(normal)));
}

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

namespace raw {
enum class button { TAB, SPACE, LEFT, RIGHT, UP, DOWN, W, A, S, D, I, K, J, L, U, O, NONE };
}

int main(int argc, char* argv[]) {
	stbi_set_flip_vertically_on_load(true);

	float cube_pos[] = {
		-0.5f, -0.5f, -0.5f, 0.0f,	0.0f,  -1.0f, 1.0f,	 0.0f,	0.0f,  0.5f,  -0.5f, -0.5f,
		0.0f,  0.0f,  -1.0f, 0.0f,	1.0f,  0.0f,  0.5f,	 0.5f,	-0.5f, 0.0f,  0.0f,	 -1.0f,
		0.0f,  0.0f,  1.0f,	 -0.5f, 0.5f,  -0.5f, 0.0f,	 0.0f,	-1.0f, 1.0f,  1.0f,	 0.0f,

		-0.5f, -0.5f, 0.5f,	 0.0f,	0.0f,  1.0f,  1.0f,	 0.0f,	1.0f,  0.5f,  -0.5f, 0.5f,
		0.0f,  0.0f,  1.0f,	 0.0f,	1.0f,  1.0f,  0.5f,	 0.5f,	0.5f,  0.0f,  0.0f,	 1.0f,
		1.0f,  1.0f,  1.0f,	 -0.5f, 0.5f,  0.5f,  0.0f,	 0.0f,	1.0f,  0.0f,  0.0f,	 0.0f,

		-0.5f, 0.5f,  0.5f,	 -1.0f, 0.0f,  0.0f,  0.0f,	 0.0f,	0.0f,  -0.5f, 0.5f,	 -0.5f,
		-1.0f, 0.0f,  0.0f,	 1.0f,	1.0f,  0.0f,  -0.5f, -0.5f, -0.5f, -1.0f, 0.0f,	 0.0f,
		1.0f,  0.0f,  0.0f,	 -0.5f, -0.5f, 0.5f,  -1.0f, 0.0f,	0.0f,  1.0f,  0.0f,	 1.0f,

		0.5f,  0.5f,  0.5f,	 1.0f,	0.0f,  0.0f,  1.0f,	 1.0f,	1.0f,  0.5f,  0.5f,	 -0.5f,
		1.0f,  0.0f,  0.0f,	 0.0f,	0.0f,  1.0f,  0.5f,	 -0.5f, -0.5f, 1.0f,  0.0f,	 0.0f,
		0.0f,  1.0f,  0.0f,	 0.5f,	-0.5f, 0.5f,  1.0f,	 0.0f,	0.0f,  0.0f,  1.0f,	 1.0f,

		-0.5f, -0.5f, -0.5f, 0.0f,	-1.0f, 0.0f,  1.0f,	 0.0f,	0.0f,  0.5f,  -0.5f, -0.5f,
		0.0f,  -1.0f, 0.0f,	 0.0f,	1.0f,  0.0f,  0.5f,	 -0.5f, 0.5f,  0.0f,  -1.0f, 0.0f,
		0.0f,  1.0f,  1.0f,	 -0.5f, -0.5f, 0.5f,  0.0f,	 -1.0f, 0.0f,  1.0f,  0.0f,	 1.0f,

		-0.5f, 0.5f,  -0.5f, 0.0f,	1.0f,  0.0f,  1.0f,	 1.0f,	0.0f,  0.5f,  0.5f,	 -0.5f,
		0.0f,  1.0f,  0.0f,	 0.0f,	0.0f,  1.0f,  0.5f,	 0.5f,	0.5f,  0.0f,  1.0f,	 0.0f,
		1.0f,  1.0f,  1.0f,	 -0.5f, 0.5f,  0.5f,  0.0f,	 1.0f,	0.0f,  0.0f,  0.0f,	 0.0f,
	};

	unsigned int indices[] = {0,  1,  2,  2,  3,  0,

							  4,  5,  6,  6,  7,  4,

							  8,  9,  10, 10, 11, 8,

							  12, 13, 14, 14, 15, 12,

							  16, 17, 18, 18, 19, 16,

							  20, 21, 22, 22, 23, 20};

	glm::vec3 cubes[] {glm::vec3(-2.0f, 0.0f, 0.0f),   glm::vec3(2.0f, 0.0f, 0.0f),
					   glm::vec3(-1.5f, -2.2f, -2.5f), glm::vec3(-3.8f, -2.0f, -12.3f),
					   glm::vec3(2.4f, -0.4f, -3.5f),  glm::vec3(-1.7f, 3.0f, -7.5f),
					   glm::vec3(1.3f, -2.0f, -2.5f),  glm::vec3(1.5f, 2.0f, -2.5f),
					   glm::vec3(1.5f, 0.2f, -1.5f),   glm::vec3(-1.3f, 1.0f, -1.5f),
					   glm::vec3(0.0f, 0.0f, -1.5f),   glm::vec3(5.0f, 0.f, 0.f),
					   glm::vec3(10.0f, 0.f, 0.f)};

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	std::cout << "SDL initialized successfully!" << std::endl;
	std::cout << "First step is complete." << std::endl;

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window* window =
		SDL_CreateWindow("SDL OpenGL Window - QUATERNIONS (World Space)", 2560, 1440,
						 (SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_FULLSCREEN));

	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}
	SDL_GLContext gl_context = SDL_GL_CreateContext(window);

	if (!gl_context) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError()
				  << std::endl;
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	std::cout << "OpenGL context created successfully!" << std::endl;

	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		SDL_GL_DestroyContext(gl_context);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	std::cout << "GLAD initialized successfully!" << std::endl;

	unsigned int vao_1 = 0;
	glGenVertexArrays(1, &vao_1);
	glBindVertexArray(vao_1);

	unsigned int vbo_1 = 0;
	glGenBuffers(1, &vbo_1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);

	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_pos), cube_pos, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	unsigned int ebo_1 = 0;

	glGenBuffers(1, &ebo_1);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	unsigned int light_vao = 0;
	glGenVertexArrays(1, &light_vao);
	glBindVertexArray(light_vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	float normal_lines_data[24 * 2 * 3];
	float normal_length = 0.2f;

	unsigned int j = 0;
	for (size_t i = 0; i < sizeof(cube_pos) / sizeof(cube_pos[0]); i += 9) {
		glm::vec3 position	 = glm::vec3(cube_pos[i], cube_pos[i + 1], cube_pos[i + 2]);
		glm::vec3 normal_vec = glm::vec3(cube_pos[i + 3], cube_pos[i + 4], cube_pos[i + 5]);

		normal_lines_data[j++] = position.x;
		normal_lines_data[j++] = position.y;
		normal_lines_data[j++] = position.z;

		glm::vec3 end_point	   = position + normal_vec * normal_length;
		normal_lines_data[j++] = end_point.x;
		normal_lines_data[j++] = end_point.y;
		normal_lines_data[j++] = end_point.z;
	}

	unsigned int line_vao = 0;
	glGenVertexArrays(1, &line_vao);
	glBindVertexArray(line_vao);

	unsigned int vbo_lines = 0;
	glGenBuffers(1, &vbo_lines);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_lines);
	glBufferData(GL_ARRAY_BUFFER, sizeof(normal_lines_data), normal_lines_data, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	int width, height;
	SDL_GetWindowSize(window, &width, &height);
	glViewport(0, 0, width, height);

	bool	  running = true;
	SDL_Event event;

	glClearColor(0.5f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	raw::shader shader_program("shaders/objects/vertex_shader.glsl",
							   "shaders/objects/color_shader.frag");
	raw::shader light_shader("shaders/light/vertex_shader.glsl", "shaders/light/color_shader.frag");
	light_shader.use();
	light_shader.set_vec3("lightColor", 1.0f, 1.0f, 1.0f);
	shader_program.use();

	glm::vec3 cameraPos	  = glm::vec3(0.0f, 0.0f, 5.0f);
	glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraUp	  = glm::vec3(0.0f, 1.0f, 0.0f);

	glm::mat4 model		 = glm::mat4(1.0f);
	glm::mat4 view		 = raw::look_at(cameraPos, cameraFront + cameraPos, cameraUp);
	glm::mat4 projection = glm::mat4(1.0f);
	shader_program.set_mat4("model", glm::value_ptr(model));
	shader_program.set_mat4("view", glm::value_ptr(view));
	shader_program.set_mat4("projection", glm::value_ptr(projection));

	glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
	model = glm::translate(model, lightPos);
	model = glm::scale(model, glm::vec3(0.2f));
	light_shader.use();
	light_shader.set_mat4("model", glm::value_ptr(model));
	light_shader.set_mat4("view", glm::value_ptr(view));
	light_shader.set_mat4("projection", glm::value_ptr(projection));

	raw::shader lines_shader("shaders/lines/vertex_shader.glsl", "shaders/lines/color_shader.frag");
	lines_shader.use();
	model = glm::mat4(1.0f);
	lines_shader.set_mat4("model", glm::value_ptr(model));
	lines_shader.set_mat4("view", glm::value_ptr(view));
	lines_shader.set_mat4("projection", glm::value_ptr(projection));

	SDL_SetWindowMouseGrab(window, true);
	SDL_SetWindowRelativeMouseMode(window, true);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 100);

	glEnable(GL_MULTISAMPLE);

	float yaw = -90.0f, pitch = 0.0f;
	float oldx = 0.0f, oldy = 0.0f;

	float		   cameraSpeed	  = 0.05f;
	constexpr long updateMoveTime = 360;
	auto		   start		  = std::chrono::high_resolution_clock::now();
	auto		   end			  = std::chrono::high_resolution_clock::now();

	raw::button pressedButton = raw::button::NONE;

	float fov = 45.0f;

	glm::quat object_quat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	float	  delta_angle = 1.0f;

	while (running) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			} else if (event.type == SDL_EVENT_KEY_DOWN) {
				if (event.key.scancode == SDL_SCANCODE_SPACE) {
					pressedButton = raw::button::SPACE;
				} else if (event.key.scancode == SDL_SCANCODE_ESCAPE) {
					std::cout << "Don't close me mf!" << std::endl;
					running = false;
				} else if (event.key.scancode == SDL_SCANCODE_LEFT) {
					pressedButton = raw::button::LEFT;
				} else if (event.key.scancode == SDL_SCANCODE_RIGHT) {
					pressedButton = raw::button::RIGHT;
				} else if (event.key.scancode == SDL_SCANCODE_UP) {
					pressedButton = raw::button::UP;
				} else if (event.key.scancode == SDL_SCANCODE_DOWN) {
					pressedButton = raw::button::DOWN;
				} else if (event.key.scancode == SDL_SCANCODE_TAB) {
					pressedButton = raw::button::TAB;
				} else if (event.key.scancode == SDL_SCANCODE_A) {
					pressedButton = raw::button::A;
				} else if (event.key.scancode == SDL_SCANCODE_W) {
					pressedButton = raw::button::W;
				} else if (event.key.scancode == SDL_SCANCODE_S) {
					pressedButton = raw::button::S;
				} else if (event.key.scancode == SDL_SCANCODE_D) {
					pressedButton = raw::button::D;
				} else if (event.key.scancode == SDL_SCANCODE_I) {
					pressedButton = raw::button::I;
				} else if (event.key.scancode == SDL_SCANCODE_K) {
					pressedButton = raw::button::K;
				} else if (event.key.scancode == SDL_SCANCODE_J) {
					pressedButton = raw::button::J;
				} else if (event.key.scancode == SDL_SCANCODE_L) {
					pressedButton = raw::button::L;
				} else if (event.key.scancode == SDL_SCANCODE_U) {
					pressedButton = raw::button::U;
				} else if (event.key.scancode == SDL_SCANCODE_O) {
					pressedButton = raw::button::O;
				}
			} else if (event.type == SDL_EVENT_MOUSE_MOTION) {
				float xoffset = event.motion.xrel;
				float yoffset = event.motion.yrel;

				float sensitivity = 0.1f;
				xoffset *= sensitivity;
				yoffset *= sensitivity;

				yaw += xoffset;
				pitch -= yoffset;

				if (pitch > 89.0f) {
					pitch = 89.0f;
				}
				if (pitch < -89.0f) {
					pitch = -89.0f;
				}

				glm::vec3 front;
				front.x		= cos(glm::radians(yaw)) * cos(glm::radians(pitch));
				front.y		= sin(glm::radians(pitch));
				front.z		= sin(glm::radians(yaw)) * cos(glm::radians(pitch));
				cameraFront = glm::normalize(front);
			} else if (event.type == SDL_EVENT_KEY_UP) {
				pressedButton = raw::button::NONE;
			} else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
				event.wheel.y < 0 ? fov += 1.0f : fov -= 1.0f;
				if (fov < 0.0000001f)
					fov = 1.0f;
				if (fov > 180.0f)
					fov = 180.0f;
				projection =
					raw::perspective(glm::radians(fov), width / float(height), 0.1f, 100.0f);
				shader_program.use();
				shader_program.set_mat4("projection", glm::value_ptr(projection));
				light_shader.use();
				light_shader.set_mat4("projection", glm::value_ptr(projection));
			}
		}
		if ((end - start > std::chrono::milliseconds(1000 / updateMoveTime)) &&
			pressedButton != raw::button::NONE) {
			switch (pressedButton) {
			case raw::button::TAB:
				cameraPos.y -= cameraSpeed;
				break;
			case raw::button::SPACE:
				cameraPos.y += cameraSpeed;
				break;
			case raw::button::LEFT:
				cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
				break;
			case raw::button::RIGHT:
				cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
				break;
			case raw::button::UP:
				cameraPos += cameraFront * cameraSpeed;
				break;
			case raw::button::DOWN:
				cameraPos -= cameraFront * cameraSpeed;
				break;
			case raw::button::A:
				cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
				break;
			case raw::button::D:
				cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
				break;
			case raw::button::W:
				cameraPos += cameraFront * cameraSpeed;
				break;
			case raw::button::S:
				cameraPos -= cameraFront * cameraSpeed;
				break;
			case raw::button::I: {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = object_quat * delta;
				break;
			}
			case raw::button::K: {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(1.0f, 0.0f, 0.0f));
				object_quat = object_quat * delta;
				break;
			}
			case raw::button::J: {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = object_quat * delta;
				break;
			}
			case raw::button::L: {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 1.0f, 0.0f));
				object_quat = object_quat * delta;
				break;
			}
			case raw::button::U: {
				glm::quat delta =
					glm::angleAxis(glm::radians(-delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = object_quat * delta;
				break;
			}
			case raw::button::O: {
				glm::quat delta =
					glm::angleAxis(glm::radians(delta_angle), glm::vec3(0.0f, 0.0f, 1.0f));
				object_quat = object_quat * delta;
				break;
			}
			}
			start = std::chrono::high_resolution_clock::now();
		}
		end = std::chrono::high_resolution_clock::now();

		view = raw::look_at(cameraPos, cameraPos + cameraFront, cameraUp);

		shader_program.use();
		shader_program.set_vec3("lightPos", lightPos.x, lightPos.y, lightPos.z);
		shader_program.set_vec3("viewPos", cameraPos.x, cameraPos.y, cameraPos.z);
		shader_program.set_vec3("lightColor", 1.0f, 1.0f, 1.0f);

		shader_program.set_float("ambientStrength", 0.2f);
		shader_program.set_float("specularStrength", 0.5f);
		shader_program.set_float("shininess", 32.0f);
		shader_program.set_mat4("view", glm::value_ptr(view));
		light_shader.use();
		light_shader.set_mat4("view", glm::value_ptr(view));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float rotation_angle = (float)SDL_GetTicks() / 1000.0f * glm::radians(50.0f);
		for (unsigned int i = 0; i < sizeof(cubes) / sizeof(glm::vec3); i++) {
			glm::mat4 current_cube_model(1.0f);

			if (i == 1) {
				glm::mat4 model_quat_rotation = glm::mat4_cast(object_quat);
				current_cube_model =
					glm::translate(glm::mat4(1.0f), cubes[i]) * model_quat_rotation;
			} else {
				if (i == 10) {
					current_cube_model = glm::translate(
						glm::scale(model, glm::vec3(-0.5f, -0.5f, -0.5f)), cubes[10]);

				} else if (i < 10) {
					current_cube_model = glm::translate(current_cube_model, cubes[i]);
					float angle		   = glm::radians(20.0f * i) + rotation_angle;
					current_cube_model =
						glm::rotate(current_cube_model, angle, glm::vec3(0.5f, 1.0f, 0.0f));
				} else if (i == 11) {
					current_cube_model = glm::translate(current_cube_model, cubes[i]);
				} else {
					glm::vec3 rotationCenter   = cubes[11];
					glm::vec3 relativePosition = cubes[i] - rotationCenter;

					current_cube_model = glm::translate(current_cube_model, rotationCenter);
					current_cube_model = glm::rotate(current_cube_model, rotation_angle,
													 glm::vec3(0.0f, 1.0f, 0.0f));
					current_cube_model = glm::translate(current_cube_model, -rotationCenter);

					current_cube_model = glm::translate(current_cube_model, cubes[i]);
					float angle		   = glm::radians(20.0f * i) + rotation_angle;
					current_cube_model =
						glm::rotate(current_cube_model, angle, glm::vec3(0.5f, 1.0f, 0.0f));
					current_cube_model =
						glm::scale(current_cube_model, glm::vec3(0.5f, 0.5f, 0.5f));
				}
			}

			lines_shader.use();
			lines_shader.set_mat4("model", glm::value_ptr(current_cube_model));
			lines_shader.set_mat4("view", glm::value_ptr(view));
			lines_shader.set_mat4("projection", glm::value_ptr(projection));

			glBindVertexArray(line_vao);
			glDrawArrays(GL_LINES, 0, 48);
			glBindVertexArray(vao_1);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
			shader_program.use();
			shader_program.set_mat4("model", glm::value_ptr(current_cube_model));
			shader_program.set_mat4("view", glm::value_ptr(view));
			shader_program.set_mat4("projection", glm::value_ptr(projection));
			glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
		}

		glBindVertexArray(light_vao);
		light_shader.use();
		glm::mat4 lightModel = glm::mat4(1.0f);
		lightModel			 = glm::translate(lightModel, lightPos);
		lightModel			 = glm::scale(lightModel, glm::vec3(0.2f));
		light_shader.set_mat4("model", glm::value_ptr(lightModel));
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
		glBindVertexArray(0);
		SDL_GL_SwapWindow(window);
	}

    glDeleteVertexArrays(1, &vao_1);
    glDeleteVertexArrays(1, &light_vao);
    glDeleteVertexArrays(1, &line_vao);
    glDeleteBuffers(1, &vbo_1);
    glDeleteBuffers(1, &vbo_lines);
    glDeleteBuffers(1, &ebo_1);
    SDL_GL_DestroyContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}