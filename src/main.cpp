//
// Created by progamers on 6/2/25.
//
#define STB_IMAGE_IMPLEMENTATION
#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
#include <iostream>
#include <memory>

#include "shader.h"
#include "stb_image.h"

namespace raw {
enum class button { TAB, SPACE, LEFT, RIGHT, UP, DOWN, NONE };
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

	glm::vec3 cubes[] {glm::vec3(0.0f, 0.0f, 0.0f),	   glm::vec3(2.0f, 5.0f, -15.0f),
					   glm::vec3(-1.5f, -2.2f, -2.5f), glm::vec3(-3.8f, -2.0f, -12.3f),
					   glm::vec3(2.4f, -0.4f, -3.5f),  glm::vec3(-1.7f, 3.0f, -7.5f),
					   glm::vec3(1.3f, -2.0f, -2.5f),  glm::vec3(1.5f, 2.0f, -2.5f),
					   glm::vec3(1.5f, 0.2f, -1.5f),   glm::vec3(-1.3f, 1.0f, -1.5f),
					   glm::vec3(0.0f, 0.0f, -1.5f)};

	// Initialize SDL
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	std::cout << "SDL initialized successfully!" << std::endl;
	std::cout << "First step is complete." << std::endl;

	// Set SDL attributes, version, core, whether we should use double buffer or not (used for
	// rendering), and 3d depth
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	// Create the window
	SDL_Window* window =
		SDL_CreateWindow("SDL OpenGL Window", 2560, 1440,
						 (SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_FULLSCREEN));

	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}
	// Initialize opengl context
	SDL_GLContext gl_context = SDL_GL_CreateContext(window);

	if (!gl_context) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError()
				  << std::endl;
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	std::cout << "OpenGL context created successfully!" << std::endl;

	// Initialize glad context
	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		SDL_GL_DestroyContext(gl_context);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	std::cout << "GLAD initialized successfully!" << std::endl;

	// Create vertex array object and bind it for the first triangle
	unsigned int vao_1 = 0;
	glGenVertexArrays(1, &vao_1);
	glBindVertexArray(vao_1);

	// Create vertex buffer object and bind it
	unsigned int vbo_1 = 0;
	glGenBuffers(1, &vbo_1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);

	// Bind the vertex data to the buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_pos), cube_pos,
				 GL_STATIC_DRAW); // Here we are passing the vertex data to the GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	unsigned int ebo_1 = 0;

	// Create element buffer object and bind it
	glGenBuffers(1, &ebo_1);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
	// Bind the index data to the buffer
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
				 GL_STATIC_DRAW); // Here we are passing the index data to the GPU
	// Unbind the vertex array object and the buffer

	// Unbind all the buffers and vertex array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind the vertex buffer object

	unsigned int light_vao = 0;
	glGenVertexArrays(1, &light_vao);
	glBindVertexArray(light_vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	// Set the viewport to the window size
	int width, height;
	SDL_GetWindowSize(window, &width, &height);
	glViewport(0, 0, width, height); // Set the viewport to the window size

	bool	  running = true;
	SDL_Event event;

	glClearColor(0.5f, 0.0f, 0.0f, 1.0f); // Set the clear color to a dark blue
	glEnable(GL_DEPTH_TEST);

	raw::shader shader_program("shaders/objects/vertex_shader.glsl",
							   "shaders/objects/color_shader.frag");
	raw::shader light_shader("shaders/light/vertex_shader.glsl", "shaders/light/color_shader.frag");
	light_shader.use();
	light_shader.set_vec3("lightColor", 1.0f, 1.0f, 1.0f);
	shader_program.use();

	glm::vec3 cameraPos	  = glm::vec3(0.0f, 0.0f, 3.0f);
	glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraUp	  = glm::vec3(0.0f, 1.0f, 0.0f);

	glm::mat4 model		 = glm::mat4(1.0f);
	glm::mat4 view		 = glm::lookAt(cameraPos, cameraFront + cameraPos, cameraUp);
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
	light_shader.set_mat4("projection", glm::value_ptr(glm::mat4(1.0f)));

	SDL_SetWindowMouseGrab(window, true);
	SDL_SetWindowRelativeMouseMode(window, true);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);

	float yaw = -90.0f, pitch = 0.0f;
	float oldx = 0.0f, oldy = 0.0f;

	float		   cameraSpeed	  = 0.05f;
	constexpr long updateMoveTime = 360;
	auto		   start		  = std::chrono::high_resolution_clock::now();
	auto		   end			  = std::chrono::high_resolution_clock::now();

	raw::button pressedButton = raw::button::NONE;

	float fov = 45.0f;

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
					glm::perspective(glm::radians(fov), width / float(height), 0.1f, 100.0f);
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
			}
			start = std::chrono::high_resolution_clock::now();
		}
		end = std::chrono::high_resolution_clock::now();

		view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

		shader_program.use();
		shader_program.set_vec3("lightPos", lightPos.x, lightPos.y, lightPos.z);
		shader_program.set_vec3("viewPos", cameraPos.x, cameraPos.y, cameraPos.z);
		shader_program.set_vec3("lightColor", 1.0f, 1.0f, 1.0f);

		shader_program.set_float("ambientStrength", 0.1f);
		shader_program.set_float("specularStrength", 0.5f);
		shader_program.set_float("shininess", 32.0f);
		shader_program.set_mat4("view", glm::value_ptr(view));
		light_shader.use();
		light_shader.set_mat4("view", glm::value_ptr(view));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the screen and depth buffer

		glBindVertexArray(vao_1); // Bind the vertex array object
		float rotation_angle = (float)SDL_GetTicks() / 1000.0f * glm::radians(50.0f);
		for (unsigned int i = 0; i <= 10; i++) {
			glm::mat4 model = glm::mat4(1.0f);
			if (i == 10) {
				model = glm::translate(glm::scale(model, glm::vec3(-0.5f, -0.5f, -0.5f)),
									   glm::vec3(4.0f, -4.0f, 0.0f));
			} else {
				model		= glm::translate(model, cubes[i]);
				float angle = glm::radians(20.0f * i) + rotation_angle;
				model		= glm::rotate(model, angle, glm::vec3(0.5f, 1.0f, 0.0f));
			}
			shader_program.use();
			shader_program.set_mat4("model", glm::value_ptr(model));
			glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
		}

		glBindVertexArray(light_vao);
		light_shader.use();
		glm::mat4 lightModel = glm::mat4(1.0f);
		lightModel			 = glm::translate(lightModel, lightPos);
		lightModel			 = glm::scale(lightModel, glm::vec3(0.2f));
		light_shader.set_mat4("model", glm::value_ptr(lightModel));
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);
        SDL_GL_SwapWindow(window);
	}

	int nrAttributes;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
	std::cout << "Maximum nr of vertex attributes supported: " << nrAttributes << std::endl;

	SDL_GL_DestroyContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}