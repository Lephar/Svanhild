#pragma once

#define GLFW_INCLUDE_VULKAN
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <array>
#include <limits>
#include <vector>
#include <chrono>
#include <memory>
#include <fstream>
#include <iostream>

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <tinygltf/tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace svh {
	struct Details {
		uint32_t imageCount;
		vk::Extent2D swapchainExtent;
		vk::SurfaceTransformFlagBitsKHR swapchainTransform;
		vk::Format depthStencilFormat;
		vk::SurfaceFormatKHR surfaceFormat;
		vk::PresentModeKHR presentMode;
		vk::Format imageFormat;
		uint32_t mipLevels;
		vk::SampleCountFlagBits sampleCount;
	};

	struct Vertex {
		glm::vec4 pos;
		glm::vec4 nor;
		glm::vec2 tex;
	};

	struct Buffer {
		vk::Buffer data;
		vk::DeviceMemory memory;
	};

	struct Image {
		vk::ImageView view;
		vk::Image data;
		vk::DeviceMemory memory;
	};

	struct Mesh {
		uint32_t indexOffset;
		uint32_t indexLength;
		uint32_t vertexOffset;
		uint32_t vertexLength;
		uint32_t imageIndex;
	};

	struct Model {
		uint32_t meshIndex;
		uint32_t meshCount;
	};

	struct Prop {
		uint32_t modelIndex;
		uint32_t uniformIndex;
		std::vector<vk::DescriptorSet> descriptors;
	};
}