//
// Created by progamers on 6/2/25.
//
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <iostream>

int main(int argc, char* argv[]) {
	if(!SDL_Init(SDL_INIT_VIDEO)) {
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

	SDL_Window* window = SDL_CreateWindow("SDL OpenGL Window", 800, 600, (SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE));

	if(!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	SDL_GLContext gl_context = SDL_GL_CreateContext(window);

	if(!gl_context) {
		std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
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



	bool running = true;
	SDL_Event event;

	glClearColor(0.65f, 0.65f, 0.65f, 1.f);

	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			}
		}
		glClear(GL_COLOR_BUFFER_BIT);
		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DestroyContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}