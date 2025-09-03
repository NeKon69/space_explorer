#version 460 core
in vec3     v_world_pos;
in vec3     v_normal;
in vec2     v_tex_coord;
in mat3     v_tbn;
out vec4 FragColor;

// uniform samplerCube irradiance;
// uniform samplerCube prefilter;
// uniform sampler2d brdf_lut;

struct light {
	vec3 position;
	vec3 color;
};
// I need to make it non fixable, maybe put it in some instanced shit dk
#define NR_LIGHTS 4
uniform light lights[NR_LIGHTS];
uniform vec3  camera_pos;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
	float a2 = pow(roughness, 4);
	// Prevent from negative values
	float n_dot_h  = max(dot(N, H), 0.0);
	float n_dot_h2 = pow(n_dot_h, 2);

	float nom    = a2;
	float denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
	denom        = PI * pow(denom, 2);

	return nom / denom;
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
	float r        = roughness + 1.0f;
	float k        = pow(r, 2) / 8.0f;
	float nom    = n_dot_v;
	float denom = n_dot_v * (1.0 - k) + k;
	return nom / denom;
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
	// Again, preventing negative values
	float n_dot_v = max(dot(N, V), 0.0f);
	float n_dot_l = max(dot(N, L), 0.0f);
	float ggx_2      = geometry_schlick_ggx(n_dot_v, roughness);
	float ggx_1      = geometry_schlick_ggx(n_dot_l, roughness);

	return ggx_1 * ggx_2;
}

vec3 fresnel_schlick(float cos_theta, vec3 f_0) {
	return f_0 + (1.0f - f_0) * pow(clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

void main() {
	// Later will replace with textures;
	vec4 alb_met    = vec4(0.5, 0.5, 0.75, 1);
	vec4 norm_rough = vec4(v_normal, 1);
	vec4 clouds_ao    = vec4(1.0, 1.0, 1.0, 0.25);
	vec3 albedo_val = alb_met.rgb;

	vec3 normal_tangent_space = vec3(norm_rough.rgb * 2.0f - 1.0f);
	vec3 N                      = normalize(mat3(v_tbn) * normal_tangent_space);

	float metallic_val    = alb_met.a;
	float roughness_val = norm_rough.a;
	float ao_val        = clouds_ao.a;

	vec3 V = normalize(camera_pos - v_world_pos);

	vec3 f_0 = vec3(0.04f);
	f_0         = mix(f_0, albedo_val, metallic_val);

	vec3 lo = vec3(0.0f);
	for (int i = 0; i < NR_LIGHTS; ++i) {
		vec3 L = normalize(lights[i].position - v_world_pos);
		vec3 H = normalize(V + L);

		float distance      = length(lights[i].position - v_world_pos);
		float attenuation = 1.0f / pow(distance, 2);
		vec3  radiance      = lights[i].color * attenuation;

		float NDF = distribution_ggx(N, H, roughness_val);
		float G      = geometry_smith(N, V, L, roughness_val);
		vec3  F      = fresnel_schlick(max(dot(H, V), 0.0f), f_0);

		vec3  numerator      = NDF * G * F;
		float denominator = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f) + 0.0001f;
		vec3  specular      = numerator / denominator;

		vec3 ks = F;
		vec3 kd = vec3(1.0f) - ks;
		kd += 1.0f - metallic_val;

		float n_dot_l = max(dot(N, L), 0.0f);
		lo += (kd * albedo_val / PI + specular) * radiance * n_dot_l;
	}

	// I'll add BG later I plan to just stick some hdr/mp4 there with galaxy
	//    vec3 f_ibl = fresnel_shlick(max(dot(N, V), 0.0f), f_0);
	//    vec3 ks_ibl = f_ibl;
	//    vec3 kd_ibl = f_ibl;
	//    kd_ibl += 1.0f - metallic_val;
	//
	//    vec3 irradiance = texture(irradiance, N).rgb;
	//    vec3 diffuse_ibl = irradiance * albedo_val;
	//
	//    vec3 R = reflect(-V, N);
	//    const float MAX_REFLECTION_LOD = 4.0f;
	//    vec3 prefiltered_color = textureLod(prefilter, R, roughness_val * MAX_REFLECTION_LOD).rgb;
	//    vec2 brdf = texture(brdf_lut, vec2(max(dot(N, V), 0.0f), 0.0f, roughness_val)).rg;
	//    vec3 specular_ibl = prefiltered_color * (f_ibl * brdf.x + brdf.y);
	//
	//    vec3 ambient = (kd_ibl * diffuse_ibl + specular_ibl) * ao_val;
	//    vec3 color = ambient + lo;

	vec3 color = lo;
	color       = color / (color + vec3(1.0f));
	color       = pow(color, vec3(1.0f / 2.2f));

	FragColor = vec4(color, 1.0f);
}