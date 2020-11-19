#version 460 core
#extension GL_ARB_separate_shader_objects: enable

layout(binding = 2) uniform sampler2D textureSampler;

layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputTexture;

layout(location = 0) out vec4 outputColor;

void main()
{
	vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	vec4 lightPosition = vec4(-8.0f, 8.0f, 0.0f, 1.0f);
	vec4 lightDirection = normalize(lightPosition - vec4(inputPosition, 1.0f));
	float intensity = max(dot(vec4(inputNormal, 0.0f), lightDirection), 0.0f);
	vec4 pointLight = intensity * lightColor;
	vec4 ambientLight = vec4(0.04f, 0.04f, 0.04f, 1.0f);

	outputColor = (pointLight + ambientLight) * texture(textureSampler, inputTexture);
}