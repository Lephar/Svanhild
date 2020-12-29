#version 460 core
#extension GL_ARB_separate_shader_objects: enable

layout(binding = 1) uniform sampler2D textureSampler;

layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec3 inputNormal;
layout(location = 2) in vec2 inputTexture;

layout(location = 0) out vec4 outputColor;

void main()
{
	outputColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);

	gl_FragDepth = 1.0f;
}