//
// Created by progamers on 6/2/25.
//
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include "shader.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
	stbi_set_flip_vertically_on_load(true);

	float triangle_pos_1[] = {
		// positions         // colors         // texture coords
		0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
		0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
		-0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
		-0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left
	};
	
	// What openGL should draw for me, in this example if library follows the given array drawing steps, we will get a square made out of 2 triangles
	unsigned int indices[] = {
		0, 1, 2,
		2, 3, 0
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

	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	std::cout << "GLAD initialized successfully!" << std::endl;

	int width_image = 0, height_image = 0, nrChannels = 0;
	unsigned char *data = stbi_load("images/neofetch5.png", &width_image, &height_image, &nrChannels, 0);

	std::unique_ptr<unsigned char, void (*)(void*)> textureData(data, stbi_image_free);

	if (textureData) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_image, height_image, 0, GL_RGB,
					 GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else {
		std::cerr << "Failed to load texture " << stbi_failure_reason() << std::endl;
		SDL_GL_DestroyContext(gl_context);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	// Create vertex array object and bind it for the first triangle
	unsigned int vao_1 = 0;
	glGenVertexArrays(1, &vao_1);
	glBindVertexArray(vao_1);

	// Create vertex buffer object and bind it
	unsigned int vbo_1 = 0;
	glGenBuffers(1, &vbo_1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);

	// Bind the vertex data to the buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_pos_1), triangle_pos_1, GL_STATIC_DRAW); // Here we are passing the vertex data to the GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), nullptr); // 0 is the index of the vertex attribute, 3 is the number of components per vertex (x, y, z), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (nullptr means start at the beginning)
	glEnableVertexAttribArray(0); // Enable the vertex attribute at index 0

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // 1 is the index of the color attribute, 3 is the number of components per color (r, g, b), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (3 * sizeof(float) means start after the position data)
	glEnableVertexAttribArray(1); // Enable the vertex attribute at index 1

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); // 2 is the index of the texture coordinate attribute, 2 is the number of components per texture coordinate (u, v), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (6 * sizeof(float) means start after the color data)
	glEnableVertexAttribArray(2); // Enable the vertex attribute at index 2

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

	raw::shader shader_program("shaders/vertex_shader.glsl", "shaders/color_shader.frag");

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			}
		}

		glClear(GL_COLOR_BUFFER_BIT); // Clear the screen and depth buffer
		shader_program.use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		shader_program.set_int("our_texture", 0);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindVertexArray(vao_1); // Bind the vertex array object
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_1);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
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