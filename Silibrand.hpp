#pragma once

#define GLFW_INCLUDE_VULKAN
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

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
		vk::SurfaceFormatKHR surfaceFormat;
		vk::PresentModeKHR presentMode;
		uint32_t mipLevels;
		vk::SampleCountFlagBits sampleCount;
	};
}