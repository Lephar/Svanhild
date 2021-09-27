#pragma once

#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_IMPLEMENTATION

#define STBI_ONLY_JPEG
#define STBI_NO_FAILURE_STRINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <array>
#include <string>
#include <limits>
#include <optional>
#include <vector>
#include <queue>
#include <chrono>
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <semaphore>
#include <shared_mutex>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tinygltf/tiny_gltf.h>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <librealsense2/rs.hpp>

namespace svh {
	constexpr auto epsilon = 0.0009765625f;

	enum class Type {
		Mesh,
		Portal,
		Camera
	};
	
	enum class Status {
		NotRecorded,
		Recording,
		Invalidated,
		Ready,
		Used,
		InUse
	};

	struct Controls {
		bool keyW;
		bool keyA;
		bool keyS;
		bool keyD;

		double_t mouseX;
		double_t mouseY;
		double_t deltaX;
		double_t deltaY;
	};

	struct Details {
		uint32_t imageCount;
		uint32_t minImageCount;
		uint32_t maxImageCount;
		uint32_t concurrentImageCount;
		uint32_t commandBufferPerImage;
		uint32_t queueCount;
		uint32_t portalCount;
		uint32_t meshCount;
		uint32_t uniformAlignment;
		uint32_t uniformQueueStride;
		uint32_t uniformFrameStride;
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

	struct State {
		uint32_t currentImage;
		uint32_t totalFrameCount;
		std::atomic<uint32_t> frameCount;
		std::atomic<uint32_t> recordingCount;
		std::atomic<bool> threadsActive;
		double_t timeDelta;
		double_t checkPoint;
		std::chrono::time_point<std::chrono::high_resolution_clock> previousTime;
		std::chrono::time_point<std::chrono::high_resolution_clock> currentTime;
	};

	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 texture;
	};

	struct Camera {
		uint32_t room;

		glm::vec3 position;
		glm::vec3 direction;
		glm::vec3 up;

		glm::vec3 previous;

		glm::mat4 transform;
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
		std::string textureName;

		glm::vec3 origin;
		glm::vec3 minBorders;
		glm::vec3 maxBorders;

		uint8_t sourceRoom;
		glm::mat4 sourceTransform;
	};

	struct Portal {
		Mesh mesh;
		uint8_t pairIndex;
		uint8_t targetRoom;

		glm::vec3 direction;
		glm::mat4 targetTransform;
		glm::mat4 cameraTransform;

		glm::mat4 transform;
	};
}
