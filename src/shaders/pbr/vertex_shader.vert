#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_tex_coord;
layout (location = 3) in vec3 a_tangent;
layout (location = 4) in vec3 a_bitangent;
// Instancing
layout (location = 5) in mat4 a_model;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_tex_coord;
// Tangent Bitanent Normal
out mat3 v_tbn;
// Hopefully one day I'll replace that shit with just a VP matrix since we don't use those one by one
uniform mat4 view;
uniform mat4 projection;

void main() {

    v_world_pos = vec3(a_model * vec4(a_pos, 1.0));
    v_tex_coord = a_tex_coord;

    // We all know what trick to preserve bugs if we do uneven scaling
    mat4 normal_matrix = transpose(inverse(mat4(a_model)));

    vec3 T = normalize(normal_matrix * a_tangent);
    vec3 B = normalize(normal_matrix * a_bitangent);
    vec3 N = normalize(normal_matrix * a_normal);

    v_tbn = mat4(T, B, N);
    v_normal = N;

    gl_Position = projection * view * vec4(v_world_pos);
}