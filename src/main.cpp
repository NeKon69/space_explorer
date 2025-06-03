//
// Created by progamers on 6/2/25.
//
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
	// Define vertices position on the screen (openGL uses normalized values for some reason)
	float triangle_pos_1[] = {
		-0.5f, -0.5f, 0.0f,
		 0.5f, -0.5f, 0.0f,
		 0.5f,  0.5f, 0.0f
	};
	float triangle_pos_2[] = {
		0.5f,  0.5f, 0.0f,
		-0.5f,  0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};
	
	// What openGL should draw for me, in this example if library follows the given array drawing steps, we will get a square made out of 2 triangles
	unsigned int indices[] = {
		0, 1, 2,
		2, 3, 0
	};
	// Try open the header file
	std::ifstream header_file("shaders/vertex_shader.glsl");
	if (!header_file.is_open()) {
		std::cerr << "Failed to open shader file." << std::endl;
		return 1;
	}
	// Add it's content to the string stream
	std::stringstream buffer;
	buffer << header_file.rdbuf();
	// Then pass it into const char* via some machinations
	const char* glsl_content;
	std::string glsl_content_str = buffer.str();
	glsl_content = glsl_content_str.c_str();

	std::ifstream header_file_2("shaders/color_shader.frag");
	if (!header_file_2.is_open()) {
		std::cerr << "Failed to open fragment shader file." << std::endl;
		return 1;
	}
	std::stringstream buffer_2;
	buffer_2 << header_file_2.rdbuf();
	const char* frag_content;
	std::string frag_content_str = buffer_2.str();
	frag_content = frag_content_str.c_str();


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
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_pos_1), triangle_pos_1, GL_STATIC_DRAW); // Here we are passing the vertex data to the GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr); // 0 is the index of the vertex attribute, 3 is the number of components per vertex (x, y, z), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (nullptr means start at the beginning)
	glEnableVertexAttribArray(0); // Enable the vertex attribute at index 0

	// Unbind all the buffers and vertex array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Initialize the second vertex array object and bind it for the second triangle
	unsigned int vao_2 = 0;
	glGenVertexArrays(1, &vao_2);
	glBindVertexArray(vao_2);

	// Create vertex buffer object and bind it
	unsigned int vbo_2 = 0;
	glGenBuffers(1, &vbo_2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_2);

	// Bind the vertex data to the buffer for the second triangle
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_pos_2), triangle_pos_2, GL_STATIC_DRAW); // Here we are passing the vertex data to the GPU
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr); // 0 is the index of the vertex attribute, 3 is the number of components per vertex (x, y, z), GL_FLOAT is the type of each component, GL_FALSE means we don't normalize the data, and the last parameter is the offset in bytes (nullptr means start at the beginning)
	glEnableVertexAttribArray(0); // Enable the vertex attribute at index 0

	// Unbind all the buffers and vertex array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// Members to check for shader compilation errors
	int  success;
	char infoLog[512];

	// Shaders and the program
	unsigned int shader_program = glCreateProgram(); // Create a shader program
	unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &glsl_content, nullptr); // Load the fragment shader source code vertex_shader is the shader ID, 1
	glCompileShader(vertex_shader); // Compile the vertex shader
	// Then check for the compilation errors
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
		std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
		exit(1);
	}

	glShaderSource(fragment_shader, 1, &frag_content, nullptr); // Load the fragment shader source code
	glCompileShader(fragment_shader); // Compile the fragment shader
	// Then check for the compilation errors
	// It's good to mention that we pretend to go to next functions only if previous steps were successful, that means infoLog will be empty
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
		std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
		exit(1);
	}

	glAttachShader(shader_program, vertex_shader); // Attach the vertex shader to the program
	glAttachShader(shader_program, fragment_shader); // Attach the fragment shader to the program

	glLinkProgram(shader_program); // Link the program
	glGetProgramiv(shader_program, GL_LINK_STATUS, &success); // Check for linking errors
	if(!success) {
		glGetProgramInfoLog(shader_program, 512, nullptr, infoLog);
		std::cerr << "Shader program linking failed: " << infoLog << std::endl;
		exit(1);
	}

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	// Set the viewport to the window size
	int width, height;
	SDL_GetWindowSize(window, &width, &height);
	glViewport(0, 0, width, height); // Set the viewport to the window size


	bool running = true;
	SDL_Event event;

	glUseProgram(shader_program); // Use the shader program we created

	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_EVENT_QUIT) {
				std::cout << "Don't close me mf!" << std::endl;
				running = false;
			}
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glBindVertexArray(vao_1); // Bind the vertex array object
		glDrawArrays(GL_TRIANGLES, 0, 3); // Draw the vertices using the vertex array object, GL_TRIANGLES means we will draw triangles, 0 is the starting index, and 6 is the number of vertices to draw
		glBindVertexArray(vao_2); // Unbind the vertex array object after drawing
		glDrawArrays(GL_TRIANGLES, 0, 3); // Draw the second triangle
		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DestroyContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}