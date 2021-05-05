#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <array>
#include <limits>
#include <vector>
#include <chrono>
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tinygltf/tiny_gltf.h>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

namespace svh {
	constexpr auto epsilon = 0.0009765625f;

	enum class Type {
		Mesh,
		Portal,
		Player,
		Observer
	};

	struct Controls {
		uint8_t observer;
		double_t mouseX;
		double_t mouseY;
		float_t deltaX;
		float_t deltaY;
		uint8_t keyW;
		uint8_t keyA;
		uint8_t keyS;
		uint8_t keyD;
		uint8_t keyQ;
		uint8_t keyE;
		uint8_t keyR;
		uint8_t keyF;
	};

	struct State {
		uint32_t currentImage;
		uint16_t frameCount;
		float_t timeDelta;
		float_t checkPoint;
		std::chrono::time_point<std::chrono::high_resolution_clock> previousTime;
		std::chrono::time_point<std::chrono::high_resolution_clock> currentTime;
	};

	struct Details {
		uint32_t imageCount;
		uint32_t portalCount;
		uint32_t meshCount;
		uint32_t uniformAlignment;
		uint32_t uniformStride;
		uint32_t uniformSize;
		vk::Extent2D swapchainExtent;
		vk::SurfaceTransformFlagBitsKHR swapchainTransform;
		vk::Format depthStencilFormat;
		vk::SurfaceFormatKHR surfaceFormat;
		vk::PresentModeKHR presentMode;
		vk::Format imageFormat;
		vk::SampleCountFlagBits sampleCount;
		uint32_t mipLevels;
		float_t maxAnisotropy;
	};

	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 texture;
	};

	struct Camera {
		glm::vec3 position;
		glm::vec3 direction;
		glm::vec3 up;
	};

	struct Buffer {
		vk::Buffer buffer;
		vk::DeviceMemory memory;
	};

	struct Image {
		vk::Image image;
		vk::ImageView view;
		vk::DeviceMemory memory;
		vk::DescriptorSet descriptor;
	};

	struct Mesh {
		uint32_t indexOffset;
		uint32_t indexLength;
		uint32_t vertexOffset;
		uint32_t vertexLength;
		uint32_t textureIndex;

		glm::vec3 origin;
		glm::vec3 normal;
		glm::vec3 minBorders;
		glm::vec3 maxBorders;

		uint8_t sourceRoom;
		glm::mat4 sourceTransform;
	};

	struct Portal {
		Mesh mesh;
		uint8_t pair;
		uint8_t targetRoom;

		glm::mat4 targetTransform;
		glm::mat4 cameraTransform;

		vk::Pipeline stencilPipeline;
		vk::Pipeline renderPipeline;
	};
}
