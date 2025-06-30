#version 410 core

struct directional_light {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct material {
    sampler2D diffuse_map;
    sampler2D specular_map;
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
in vec2 TexCoord;

#define MAX_POINT_LIGHTS 4
uniform point_light point_lights[MAX_POINT_LIGHTS];
uniform material obj_mat;
uniform directional_light dir_light;
uniform spot_light sp_light;
uniform bool need_dir_light;
uniform vec3 viewPos;

out vec4 FragColor;

vec3 calc_dir_light(directional_light light, vec3 normal, vec3 viewDir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-light.direction);

    vec3 diffuse_color = texture(obj_mat.diffuse_map, TexCoord).rgb;
    vec3 specular_color = texture(obj_mat.specular_map, TexCoord).rgb;

    float diff = max(dot(norm, lightDir), 0.0);

    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), obj_mat.shininess);

    vec3 ambient  = light.ambient * diffuse_color;
    vec3 diffuse  = light.diffuse * diff * diffuse_color;
    vec3 specular = light.specular * spec * specular_color;

    return (ambient + diffuse + specular);
}

vec3 calc_point_light(point_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light.position - frag_pos);

    vec3 diffuse_color = texture(obj_mat.diffuse_map, TexCoord).rgb;
    vec3 specular_color = texture(obj_mat.specular_map, TexCoord).rgb;

    float distance    = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    vec3 ambient = light.ambient * diffuse_color;

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * diffuse_color;

    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);
    vec3 specular = light.specular * spec * specular_color;

    return (ambient + diffuse + specular) * attenuation;
}

vec3 calc_spot_light(spot_light light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light.position - frag_pos);

    vec3 diffuse_color = texture(obj_mat.diffuse_map, TexCoord).rgb;
    vec3 specular_color = texture(obj_mat.specular_map, TexCoord).rgb;

    float distance    = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    vec3 ambient = light.ambient * diffuse_color;

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * diffuse_color;

    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(view_dir, reflectDir), 0.0), obj_mat.shininess);
    vec3 specular = light.specular * spec * specular_color;

    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cut_off - light.outer_cut_off;
    float intensity = clamp((theta - light.outer_cut_off) / epsilon, 0.0, 1.0);

    return ((diffuse + specular) * intensity * attenuation + ambient);
}

void main() {
    vec3 result = vec3(0.0);
    vec3 viewDir = normalize(viewPos - FragPos);

    if(need_dir_light) {
        result += calc_dir_light(dir_light, Normal, viewDir);
    }

    for(int i = 0; i < MAX_POINT_LIGHTS; ++i) {
        result += calc_point_light(point_lights[i], Normal, FragPos, viewDir);
    }

    result += calc_spot_light(sp_light, Normal, FragPos, viewDir);

    FragColor = vec4(result, 1.0);
}