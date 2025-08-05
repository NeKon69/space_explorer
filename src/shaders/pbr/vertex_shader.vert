#version 330 core
struct pbr_data {
    // I will pack the values into 4 floats as following -> albedo + metallic, normal + roughness, clouds + ao (for texture generation)
    //    uniform vec3 albedo;
    //    uniform float metallic;

    //    uniform vec3 normal;
    //    uniform float roughness;

    //    uniform vec3 clouds;
    //    uniform float ao;

    sampler2D albedo_metallic;
    sampler2D normal_roughness;
    sampler2D clouds_ao;
};
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_tex_coord;
layout (location = 3) in vec3 a_tangent;
layout (location = 4) in vec3 a_bitangent;
// Instancing (one for each instance)
layout (location = 5) in mat4 a_model;
// mat4 takes 4 vec4 slots, so new location would be 9
layout (location = 5 + 4) in pbr_data data;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_tex_coord;
// Tangent Bitanent Normal
out mat3 v_tbn;
out pbr_data v_textures;
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
    v_textures = data;

    gl_Position = projection * view * vec4(v_world_pos);
}