//
// Created by progamers on 6/4/25.
//
#include "shader.h"

namespace raw {
shader::shader(const char* vertex_path, const char* fragment_path) {
	set_shaders(vertex_path, fragment_path);
}

shader::~shader() {
	glDeleteProgram(id);
}

bool shader::set_shaders(const char* vertex_path, const char* fragment_path) {
	std::ifstream header_file(vertex_path);
	if (!header_file.is_open()) {
		std::cerr << "Failed to open shader file." << std::endl;
		return false;
	}
	// Add it's content to the string stream
	std::stringstream buffer;
	buffer << header_file.rdbuf();
	// Then pass it into const char* via some machinations
	const char* glsl_content;
	std::string glsl_content_str = buffer.str();
	glsl_content				 = glsl_content_str.c_str();

	std::ifstream header_file_2(fragment_path);
	if (!header_file_2.is_open()) {
		std::cerr << "Failed to open fragment shader file." << std::endl;
		return 1;
	}
	std::stringstream buffer_2;
	buffer_2 << header_file_2.rdbuf();
	const char* frag_content;
	std::string frag_content_str = buffer_2.str();
	frag_content				 = frag_content_str.c_str();

	// Members to check for shader compilation errors
	int	 success;
	char infoLog[512];

	// Shaders and the program
	unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	unsigned int vertex_shader	 = glCreateShader(GL_VERTEX_SHADER);
	id							 = glCreateProgram(); // Create a shader program

	glShaderSource(
		vertex_shader, 1, &glsl_content,
		nullptr); // Load the fragment shader source code vertex_shader is the shader ID, 1
	glCompileShader(vertex_shader); // Compile the vertex shader
	// Then check for the compilation errors
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
		std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
		return false;
	}

	glShaderSource(fragment_shader, 1, &frag_content,
				   nullptr);		  // Load the fragment shader source code
	glCompileShader(fragment_shader); // Compile the fragment shader
	// Then check for the compilation errors
	// It's good to mention that we pretend to go to next functions only if previous steps were
	// successful, that means infoLog will be empty
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
		std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
		return false;
	}

	glAttachShader(id, vertex_shader);	 // Attach the vertex shader to the program
	glAttachShader(id, fragment_shader); // Attach the fragment shader to the program

	glLinkProgram(id);							  // Link the program
	glGetProgramiv(id, GL_LINK_STATUS, &success); // Check for linking errors
	if (!success) {
		glGetProgramInfoLog(id, 512, nullptr, infoLog);
		std::cerr << "Shader program linking failed: " << infoLog << std::endl;
		return false;
	}

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	glUseProgram(id); // Use the shader program

	return true;
}

void shader::use() const {
	glUseProgram(id);
}

bool shader::set_bool(const std::string& name, bool value) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform1i(location, (int)value);
	return true;
}

bool shader::set_int(const std::string& name, int value) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform1i(location, value);
	return true;
}

bool shader::set_float(const std::string& name, float value) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform1f(location, value);
	return true;
}

bool shader::set_vec2(const std::string& name, float x, float y) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform2f(location, x, y);
	return true;
}

bool shader::set_vec3(const std::string& name, float x, float y, float z) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform3f(location, x, y, z);
	return true;
}

bool shader::set_vec4(const std::string& name, float x, float y, float z, float w) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniform4f(location, x, y, z, w);
	return true;
}

bool shader::set_mat4(const std::string& name, const float* value) const {
	unsigned int location = glGetUniformLocation(id, name.c_str());
	if (location == (unsigned int)(-1))
		return false;
	glUniformMatrix4fv(location, 1, GL_FALSE, value);
	return true;
}

} // namespace raw