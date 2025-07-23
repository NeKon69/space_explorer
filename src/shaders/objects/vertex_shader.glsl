#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in mat4 aModel;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    mat4 mod = model;
    if(aModel != mat4(1.0f)) {
        mod = aModel;
    }
    gl_Position = projection * view * mod * vec4(aPos, 1.0);
    FragPos = vec3(mod * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(mod))) * aNormal;
}