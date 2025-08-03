#version 330
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_tex_coord;
in mat4 v_tbn;

// Bla bla bla you should use textures
// Shut your ass i am on the base of implementing this shit
uniform float albedo;
uniform float normal;
uniform float metallic;
uniform float roughness;
uniform float ao;

struct light {
    vec3 position;
    vec3 color;
};
// I need to make it non fixable, maybe put it in some instanced shit dk
#define NR_LIGHTS 4
uniform light lights[NR_LIGHTS];
uniform vec3 camera_pos;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a2 = pow(roughness, 4);
    // Prese
    float n_dot_h = mat(dot(N, H), 0.0);
}
void main() {

}