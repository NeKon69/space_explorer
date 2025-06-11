#version 410 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform vec3 lightColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform float shininess;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);

    float distance = length(lightPos - FragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    vec3 attenuatedLightColor = lightColor * attenuation;
    vec3 ambient = ambientStrength * attenuatedLightColor * Color;
    float standardDiff = max(dot(norm, lightDir), 0.0);

    float spreadExponent = 0.7;
    float spreadContribution = 0.6;
    float spreadDiff = pow(standardDiff, spreadExponent);

    float finalDiff = mix(standardDiff, spreadDiff, spreadContribution);
    finalDiff = clamp(finalDiff, 0.0, 1.0);

    vec3 diffuse = finalDiff * attenuatedLightColor * Color;
    vec3 reflectDir = normalize(reflect(-lightDir, norm));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * attenuatedLightColor;

    vec3 result = ambient + diffuse + specular;

    FragColor = vec4(result, 1.0);
}