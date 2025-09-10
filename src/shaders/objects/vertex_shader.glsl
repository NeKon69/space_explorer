#version 410 core
#extension GL_NV_gpu_shader5 : enable
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_tex_coord;
layout(location = 3) in vec3 a_tangent;
layout(location = 4) in vec3 a_bitangent;
// Instancing (one for each instance)
layout(location = 5) in mat4 a_model;
//layout(location = 9) in uint64_t texture_location1;
//layout(location = 10) in uint64_t texture_location2;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;

void main() {
	// THIS IS ONLY FOR TESTING I USUALLY HAVE just mat4 mod = aModel;
	mat4 mod = a_model;
	//    if (aModel != mat4(0.0)) {
	//        mod = model;
	//    }

	gl_Position = projection * view * mod * vec4(a_pos, 1.0);
	FragPos        = vec3(mod * vec4(a_pos, 1.0));
	Normal        = mat3(transpose(inverse(mod))) * a_normal;
}