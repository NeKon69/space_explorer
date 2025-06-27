#version 410 core
struct directional_light {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct material {
    float shininess;
};

struct point_light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct spot_light {
    vec3 position;
    vec3 direction;

    float cut_off;
    float outer_cut_off;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

#define AM_POINT_LIGHTS 4

uniform point_light point_lights[AM_POINT_LIGHTS];
uniform material obj_mat;
uniform directional_light dir_light;
uniform spot_light sp_light;
uniform bool need_dir_light;
uniform vec3 viewPos;

out vec4 FragColor;

vec3 calc_dir_light(directional_light light, vec3 normal, vec3 viewDir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(norm, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), obj_mat.shininess);
    // combine results
    vec3 ambient = light.ambient * Color;
    vec3 diffuse = light.diffuse * diff * Color;
    vec3 specular = light.specular * spec;
    return (ambient + diffuse + specular);
}

vec3 calc_point_light(point_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light.position - frag_pos);

    // calculate auttentation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // then ambient
    vec3 ambient = light.ambient * Color;
    float standardDiff = max(dot(norm, lightDir), 0.0);

    vec3 diffuse = light.diffuse * Color;

    // and finally specular
    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);
    vec3 specular = light.specular * spec;

    vec3 result = ambient * attenuation + diffuse * attenuation + specular * attenuation;
    return result;
}

vec3 calc_spot_light(spot_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 lightDir = normalize(light.position - frag_pos);
    float theta = dot(lightDir, normalize(-light.direction));
    float intensity = smoothstep(light.outer_cut_off, light.cut_off, theta);
    vec3 norm = normalize(normal);

    // calculate auttentation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // then ambient
    vec3 ambient = light.ambient * Color;

    // then calculate some fancy spread coeffecient
    float diff = max(dot(normal, lightDir), 0.0);

    // then diffuse based on spread
    vec3 diffuse = light.diffuse * diff * Color;

    // and finally specular
    vec3 reflectDir = normalize(reflect(-lightDir, normal));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);

    vec3 specular = light.specular * spec;

    return ((diffuse + specular) * intensity * attenuation + ambient);
}


void main() {
    vec3 result = vec3(0.0);
    if(need_dir_light) {
        result = calc_dir_light(dir_light, Normal, normalize(viewPos - FragPos));
    }

    for(int i = 0; i < AM_POINT_LIGHTS; ++i) {
        result += calc_point_light(point_lights[i], Normal, FragPos, normalize(viewPos - FragPos));
    }

    result += calc_spot_light(sp_light, Normal, FragPos, normalize(viewPos - FragPos));
    FragColor = vec4(result, 1.0);
}