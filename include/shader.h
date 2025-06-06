//
// Created by progamers on 6/4/25.
//

#ifndef SPACE_EXPLORER_SHADER_H
#define SPACE_EXPLORER_SHADER_H

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace raw {

class shader {
public:
	unsigned int id;
	shader() = delete;
	shader(const char* vertex_path, const char* fragment_path);
	~shader();

	bool set_shaders(const char* vertex_path, const char* fragment_path);
	void use() const;
	bool set_bool(const std::string& name, bool value) const;
	bool set_int(const std::string& name, int value) const;
	bool set_float(const std::string& name, float value) const;
	bool set_vec2(const std::string& name, float x, float y) const;
	bool set_vec3(const std::string& name, float x, float y, float z) const;
	bool set_vec4(const std::string& name, float x, float y, float z, float w) const;
	bool set_mat4(const std::string& name, const float* value) const;
};

} // namespace raw



#endif // SPACE_EXPLORER_SHADER_H
