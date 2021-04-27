#version 460 core

layout(binding = 0) uniform Transform {
	mat4 transform;
};

layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputTexture;

layout(location = 0) out vec3 outputPosition;
layout(location = 1) out vec3 outputNormal;
layout(location = 2) out vec2 outputTexture;

void main()
{
	outputPosition = inputPosition;
	outputNormal = inputNormal;
	outputTexture = inputTexture;

	gl_Position = transform * vec4(outputPosition, 1.0f);
}
