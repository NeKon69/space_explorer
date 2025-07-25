#version 410 core
struct directional_light {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct point_light {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct spot_light {
    vec3 position;
    vec3 direction;

    float cut_off;
    float outer_cut_off;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;


#define AM_POINT_LIGHTS 5

uniform point_light point_lights[AM_POINT_LIGHTS];
uniform material obj_mat;
uniform directional_light dir_light;
uniform spot_light sp_light;
uniform bool need_dir_light;

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
    vec3 ambient = light.ambient * obj_mat.ambient;
    vec3 diffuse = light.diffuse * diff * obj_mat.diffuse;
    vec3 specular = light.specular * spec * obj_mat.specular;
    return (ambient + diffuse + specular);
}

vec3 calc_point_light(point_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light.position - frag_pos);

    // calculate attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    // then ambient
    vec3 ambient = light.ambient * obj_mat.ambient;

    // then diffuse
    float standardDiff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * standardDiff * obj_mat.diffuse;

    // and finally specular
    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);
    vec3 specular = light.specular * spec * obj_mat.specular;

    vec3 result = (ambient + diffuse + specular) * attenuation;
    return result;
}

vec3 calc_spot_light(spot_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 lightDir = normalize(light.position - frag_pos);
    float theta = dot(lightDir, normalize(-light.direction));
    float intensity = smoothstep(light.outer_cut_off, light.cut_off, theta);
    vec3 norm = normalize(normal);

    // calculate attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    // then ambient
    vec3 ambient = light.ambient * obj_mat.ambient;

    // then diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * obj_mat.diffuse;

    // and finally specular
    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);
    vec3 specular = light.specular * spec * obj_mat.specular;

    return ((ambient * attenuation) + (diffuse + specular) * intensity * attenuation);
}


void main() {
    vec3 result = vec3(0.0);
    vec3 viewDir = normalize(sp_light.position - FragPos);

    if (need_dir_light) {
        result += calc_dir_light(dir_light, Normal, viewDir);
    }

    for (int i = 0; i < AM_POINT_LIGHTS; ++i) {
        result += calc_point_light(point_lights[i], Normal, FragPos, viewDir);
    }

    result += calc_spot_light(sp_light, Normal, FragPos, viewDir);
    FragColor = vec4(Normal, 1.0);
}