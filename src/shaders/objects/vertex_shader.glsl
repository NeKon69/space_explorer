#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aModel1;
layout (location = 3) in vec4 aModel2;
layout (location = 4) in vec4 aModel3;
layout (location = 5) in vec4 aModel4;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;

void main() {
    // THIS IS ONLY FOR TESTING I USUALLY HAVE just mat4 mod = aModel;
    mat4 mod = mat4(aModel1, aModel2, aModel3, aModel4);
    //    if (aModel != mat4(0.0)) {
    //        mod = model;
    //    }

    gl_Position = projection * view * mod * vec4(aPos, 1.0);
    FragPos = vec3(mod * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(mod))) * aNormal;
}