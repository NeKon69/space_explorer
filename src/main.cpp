//
// Created by progamers on 6/2/25.
//
#define STB_IMAGE_IMPLEMENTATION
#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>

#include "shader.h"
#include "stb_image.h"

int main(int argc, char* argv[]) {
	stbi_set_flip_vertically_on_load(true);

	float cube_pos[] = {
		-0.5f, -0.5f, -0.5f,   1.0f, 0.0f, 0.0f,
		0.5f, -0.5f, -0.5f,   0.0f, 1.0f, 0.0f,
		0.5f,  0.5f, -0.5f,   0.0f, 0.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,   1.0f, 1.0f, 0.0f,

		-0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 1.0f,
		0.5f, -0.5f,  0.5f,   0.0f, 1.0f, 1.0f,
		0.5f,  0.5f,  0.5f,   1.0f, 1.0f, 1.0f,
		-0.5f,  0.5f,  0.5f,   0.0f, 0.0f, 0.0f
	};
	
	unsigned int indices[] = {
		0, 1, 2,
		2, 3, 0,

		4, 5, 6,
		6, 7, 4,

		7, 6, 2,
		2, 3, 7,

		0, 1, 5,
		5, 4, 0,

		1, 5, 6,
		6, 2, 1,

		0, 4, 7,
		7, 3, 0
	};

	// Initialize SDL
	if(!SDL_Init(SDL_INIT_VIDEO)) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	std::cout << "SDL initialized successfully!" << std::endl;
	std::cout << "First step is complete." << std::endl;

	// Set SDL attributes, version, core, whether we should use double buffer or not (used for rendering), and 3d depth
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	// Create the window
	SDL_Window* window = SDL_CreateWindow("SDL OpenGL Window", 800, 600, (SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE));

	if(!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}
	// Initialize opengl context
	SDL_GLContext gl_context = SDL_GL_CreateContext(window);

	if(!gl_context) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_pos), cube_pos, GL_STATIC_DRAW); // Here we are passing the vertex data to the GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr); // 0 is the index of the vertex attribute, 3 is the number of components per vertex (x, y, z), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (nullptr means start at the beginning)
	glEnableVertexAttribArray(0); // Enable the vertex attribute at index 0

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // 1 is the index of the color attribute, 3 is the number of components per color (r, g, b), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (3 * sizeof(float) means start after the position data)
	glEnableVertexAttribArray(1); // Enable the vertex attribute at index 1

	unsigned int ebo_1 = 0;

	// Create element buffer object and bind it
	glGenBuffers(1, &ebo_1);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
	// Bind the index data to the buffer
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW); // Here we are passing the index data to the GPU
	// Unbind the vertex array object and the buffer

	// Unbind all the buffers and vertex array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind the vertex buffer object

	// Set the viewport to the window size
	int width, height;
	SDL_GetWindowSize(window, &width, &height);
	glViewport(0, 0, width, height); // Set the viewport to the window size

	bool running = true;
	SDL_Event event;

	glClearColor(0.5f, 0.0f, 0.0f, 1.0f); // Set the clear color to a dark blue
	glEnable(GL_DEPTH_TEST);

	raw::shader shader_program("shaders/vertex_shader.glsl", "shaders/color_shader.frag");

	shader_program.use();

	shader_program.set_int("our_texture", 0);
	shader_program.set_int("our_texture_2", 1);

	glm::mat4 model = glm::mat4(1.0f);
	GLuint modelLoc = glGetUniformLocation(shader_program.id, "model");
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			}
			if(event.type == SDL_EVENT_KEY_DOWN) {
				if(event.key.scancode == SDL_SCANCODE_SPACE){
					model = glm::rotate(model, glm::radians(1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
				}
				else if(event.key.scancode == SDL_SCANCODE_ESCAPE) {
					std::cout << "Exiting..." << std::endl;
					running = false;
				}
				else if(event.key.scancode == SDL_SCANCODE_LEFT) {
					model = glm::translate(model, glm::vec3(-0.1f, 0.0f, 0.0f));
				}
				else if(event.key.scancode == SDL_SCANCODE_RIGHT) {
					model = glm::translate(model, glm::vec3(0.1f, 0.0f, 0.0f));
				}
				else if(event.key.scancode == SDL_SCANCODE_UP) {
					model = glm::translate(model, glm::vec3(0.0f, 0.1f, 0.0f));
				}
				else if(event.key.scancode == SDL_SCANCODE_DOWN) {
					model = glm::translate(model, glm::vec3(0.0f, -0.1f, 0.0f));
				}
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
			}
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the screen and depth buffer

		glBindVertexArray(vao_1); // Bind the vertex array object
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
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