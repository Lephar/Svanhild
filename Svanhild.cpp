#include "Svanhild.hpp"

GLFWwindow* window;
tinygltf::TinyGLTF objectLoader;
shaderc::Compiler shaderCompiler;
shaderc::CompileOptions shaderOptions;

svh::Controls controls;
svh::State state;
svh::Camera observerCamera, playerCamera;
glm::vec3 previousPosition;
std::vector<uint16_t> indices;
std::vector<svh::Vertex> vertices;
std::vector<svh::Mesh> meshes;
std::vector<svh::Portal> portals;

vk::Instance instance;
vk::SurfaceKHR surface;
vk::PhysicalDevice physicalDevice;
vk::Device device;
vk::Queue queue;
vk::CommandPool commandPool;
svh::Details details;
vk::SwapchainKHR swapchain;
std::vector<vk::Image> swapchainImages;
std::vector<vk::ImageView> swapchainViews;
vk::RenderPass renderPass;
vk::ShaderModule vertexShader, fragmentShader, stencilShader;
vk::Sampler sampler;
vk::DescriptorSetLayout descriptorSetLayout;
vk::PipelineLayout pipelineLayout;
vk::Pipeline graphicsPipeline;
svh::Image colorImage, depthImage;
std::vector<vk::Framebuffer> framebuffers;
svh::Buffer indexBuffer, vertexBuffer, uniformBuffer;
vk::DescriptorPool descriptorPool;
std::vector<vk::CommandBuffer> commandBuffers;
std::vector<vk::Fence> frameFences, orderFences;
std::vector<vk::Semaphore> availableSemaphores, finishedSemaphores;

#ifndef NDEBUG

vk::DebugUtilsMessengerEXT messenger;
vk::DispatchLoaderDynamic functionLoader;

VKAPI_ATTR VkBool32 VKAPI_CALL messageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
	VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	static_cast<void>(type);
	static_cast<void>(pUserData);

	if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		std::cout << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

#endif

void mouseCallback(GLFWwindow* handle, double x, double y) {
	static_cast<void>(handle);

	controls.deltaX = controls.mouseX - x;
	controls.deltaY = y - controls.mouseY;
	controls.mouseX = x;
	controls.mouseY = y;
}

void keyboardCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) {
	static_cast<void>(scancode);
	static_cast<void>(mods);

	if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_W)
			controls.keyW = 0;
		else if (key == GLFW_KEY_S)
			controls.keyS = 0;
		else if (key == GLFW_KEY_A)
			controls.keyA = 0;
		else if (key == GLFW_KEY_D)
			controls.keyD = 0;
		else if (key == GLFW_KEY_Q)
			controls.keyQ = 0;
		else if (key == GLFW_KEY_E)
			controls.keyE = 0;
		else if (key == GLFW_KEY_R)
			controls.keyR = 0;
		else if (key == GLFW_KEY_F)
			controls.keyF = 0;
	}
	else if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_W)
			controls.keyW = 1;
		else if (key == GLFW_KEY_S)
			controls.keyS = 1;
		else if (key == GLFW_KEY_A)
			controls.keyA = 1;
		else if (key == GLFW_KEY_D)
			controls.keyD = 1;
		else if (key == GLFW_KEY_Q)
			controls.keyQ = 1;
		else if (key == GLFW_KEY_E)
			controls.keyE = 1;
		else if (key == GLFW_KEY_R)
			controls.keyR = 1;
		else if (key == GLFW_KEY_F)
			controls.keyF = 1;
		else if (key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(handle, 1);
		else if (key == GLFW_KEY_TAB) {
			if (controls.observer)
				glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			controls.observer = !controls.observer;
		}
	}
}

void resizeEvent(GLFWwindow* handle, int width, int height) {
	static_cast<void>(width);
	static_cast<void>(height);

	glfwSetWindowSize(handle, details.swapchainExtent.width, details.swapchainExtent.width);
}

void initializeControls() {
	controls.observer = 0u;
	controls.deltaX = 0.0f;
	controls.deltaY = 0.0f;
	controls.keyW = 0u;
	controls.keyA = 0u;
	controls.keyS = 0u;
	controls.keyD = 0u;
	controls.keyQ = 0u;
	controls.keyE = 0u;
	controls.keyR = 0u;
	controls.keyF = 0u;
}

void initializeCore() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

#ifndef NDEBUG
	window = glfwCreateWindow(1280, 720, "", nullptr, nullptr);
#else
	glfwWindowHint(GLFW_DECORATED, NULL);
	window = glfwCreateWindow(1920, 1080, "", glfwGetPrimaryMonitor(), nullptr);
#endif

	initializeControls();
	glfwSetKeyCallback(window, keyboardCallback);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetFramebufferSizeCallback(window, resizeEvent);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwGetCursorPos(window, &controls.mouseX, &controls.mouseY);

	auto extensionCount = 0u;
	auto extensionNames = glfwGetRequiredInstanceExtensions(&extensionCount);
	std::vector<const char*> layers{}, extensions{ extensionNames, extensionNames + extensionCount };

#ifndef NDEBUG
	layers.push_back("VK_LAYER_KHRONOS_validation");
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	vk::ApplicationInfo applicationInfo{};
	applicationInfo.pApplicationName = "Svanhild";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
	applicationInfo.pEngineName = "Svanhild Engine";
	applicationInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
	applicationInfo.apiVersion = VK_API_VERSION_1_2;

	vk::InstanceCreateInfo instanceInfo{};
	instanceInfo.pApplicationInfo = &applicationInfo;
	instanceInfo.enabledLayerCount = layers.size();
	instanceInfo.ppEnabledLayerNames = layers.data();
	instanceInfo.enabledExtensionCount = extensions.size();
	instanceInfo.ppEnabledExtensionNames = extensions.data();

#ifndef NDEBUG
	vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{};
	messengerInfo.flags = vk::DebugUtilsMessengerCreateFlagsEXT{};
	messengerInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	messengerInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
	messengerInfo.pfnUserCallback = messageCallback;

	instanceInfo.pNext = &messengerInfo;
#endif

	instance = vk::createInstance(instanceInfo);

#ifndef NDEBUG
	functionLoader = vk::DispatchLoaderDynamic{ instance, vkGetInstanceProcAddr };
	messenger = instance.createDebugUtilsMessengerEXT(messengerInfo, nullptr, functionLoader);
#endif

	VkSurfaceKHR surfaceHandle;
	glfwCreateWindowSurface(instance, window, nullptr, &surfaceHandle);
	surface = vk::SurfaceKHR{ surfaceHandle };
}

//TODO: Implement a better device selection
vk::PhysicalDevice pickPhysicalDevice() {
	auto physicalDevices = instance.enumeratePhysicalDevices();

	for (auto& temporaryDevice : physicalDevices) {
		auto properties = temporaryDevice.getProperties();

		if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
			return temporaryDevice;
	}

	return physicalDevices.front();
}

//TODO: Add exclusive queue family support
uint32_t selectQueueFamily() {
	auto queueFamilies = physicalDevice.getQueueFamilyProperties();

	for (auto index = 0u; index < queueFamilies.size(); index++) {
		auto graphicsSupport = queueFamilies.at(index).queueFlags & vk::QueueFlagBits::eGraphics;
		auto presentSupport = physicalDevice.getSurfaceSupportKHR(index, surface);

		if (graphicsSupport && presentSupport)
			return index;
	}

	return std::numeric_limits<uint32_t>::max();
}

//TODO: Check format availability and generate mipmaps
svh::Details generateDetails() {
	svh::Details temporaryDetails;

	temporaryDetails.meshCount = 0;
	temporaryDetails.portalCount = 0;

	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	glfwGetFramebufferSize(window, reinterpret_cast<int32_t*>(&surfaceCapabilities.currentExtent.width),
		reinterpret_cast<int32_t*>(&surfaceCapabilities.currentExtent.height));
	surfaceCapabilities.currentExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
		std::min(surfaceCapabilities.maxImageExtent.width, surfaceCapabilities.currentExtent.width));
	surfaceCapabilities.currentExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
		std::min(surfaceCapabilities.maxImageExtent.height, surfaceCapabilities.currentExtent.height));

	temporaryDetails.imageCount = std::min(surfaceCapabilities.minImageCount + 1, surfaceCapabilities.maxImageCount);
	temporaryDetails.swapchainExtent = surfaceCapabilities.currentExtent;
	temporaryDetails.swapchainTransform = surfaceCapabilities.currentTransform;

	auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	temporaryDetails.surfaceFormat = surfaceFormats.front();

	for (auto& surfaceFormat : surfaceFormats)
		if (surfaceFormat.format == vk::Format::eB8G8R8A8Srgb && surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			temporaryDetails.surfaceFormat = surfaceFormat;

	auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
	auto immediateSupport = false, mailboxSupport = false;

	for (auto& presentMode : presentModes) {
		if (presentMode == vk::PresentModeKHR::eMailbox)
			mailboxSupport = true;
		else if (presentMode == vk::PresentModeKHR::eImmediate)
			immediateSupport = true;
	}

	if (mailboxSupport)
		temporaryDetails.presentMode = vk::PresentModeKHR::eMailbox;
	else if (immediateSupport)
		temporaryDetails.presentMode = vk::PresentModeKHR::eImmediate;
	else
		temporaryDetails.presentMode = vk::PresentModeKHR::eFifo;

	temporaryDetails.imageFormat = vk::Format::eR8G8B8A8Srgb;
	temporaryDetails.depthStencilFormat = vk::Format::eD32SfloatS8Uint;

	temporaryDetails.mipLevels = 1;
	temporaryDetails.sampleCount = vk::SampleCountFlagBits::e2;

	auto deviceProperties = physicalDevice.getProperties();
	temporaryDetails.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;
	temporaryDetails.uniformAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;

	return temporaryDetails;
}

void createDevice() {
	physicalDevice = pickPhysicalDevice();
	details = generateDetails();
	auto familyIndex = selectQueueFamily();

	auto queuePriority = 1.0f;
	vk::PhysicalDeviceFeatures deviceFeatures{};
	std::vector<const char*> extensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	vk::DeviceQueueCreateInfo queueInfo{};
	queueInfo.queueFamilyIndex = familyIndex;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &queuePriority;

	vk::DeviceCreateInfo deviceInfo{};
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	deviceInfo.pEnabledFeatures = &deviceFeatures;
	deviceInfo.enabledExtensionCount = extensions.size();
	deviceInfo.ppEnabledExtensionNames = extensions.data();

	vk::CommandPoolCreateInfo poolInfo{};
	poolInfo.queueFamilyIndex = familyIndex;

	device = physicalDevice.createDevice(deviceInfo);
	queue = device.getQueue(familyIndex, 0);
	commandPool = device.createCommandPool(poolInfo);
}

VkImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags flags, uint32_t mipLevels) {
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.viewType = vk::ImageViewType::e2D;
	viewInfo.image = image;
	viewInfo.format = format;
	viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
	viewInfo.subresourceRange.aspectMask = flags;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.layerCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;

	return device.createImageView(viewInfo);
}

uint32_t chooseMemoryType(uint32_t filter, vk::MemoryPropertyFlags flags) {
	auto memoryProperties = physicalDevice.getMemoryProperties();

	for (auto index = 0u; index < memoryProperties.memoryTypeCount; index++)
		if ((filter & (1 << index)) && (memoryProperties.memoryTypes[index].propertyFlags & flags) == flags)
			return index;

	return std::numeric_limits<uint32_t>::max();
}

void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
	vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
	vk::BufferCreateInfo bufferInfo{};
	bufferInfo.sharingMode = vk::SharingMode::eExclusive;
	bufferInfo.usage = usage;
	bufferInfo.size = size;

	buffer = device.createBuffer(bufferInfo);
	auto memoryRequirements = device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo allocateInfo{};
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.memoryTypeIndex = chooseMemoryType(memoryRequirements.memoryTypeBits, properties);

	bufferMemory = device.allocateMemory(allocateInfo);
	static_cast<void>(device.bindBufferMemory(buffer, bufferMemory, 0));
}

void createImage(uint32_t imageWidth, uint32_t imageHeight, uint32_t mipLevels, vk::SampleCountFlagBits samples,
	vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
	vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
	vk::ImageCreateInfo imageInfo{};
	imageInfo.imageType = vk::ImageType::e2D;
	imageInfo.extent.width = imageWidth;
	imageInfo.extent.height = imageHeight;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = vk::ImageLayout::eUndefined;
	imageInfo.usage = usage;
	imageInfo.samples = samples;
	imageInfo.sharingMode = vk::SharingMode::eExclusive;

	image = device.createImage(imageInfo);
	auto memoryRequirements = device.getImageMemoryRequirements(image);

	vk::MemoryAllocateInfo allocateInfo{};
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.memoryTypeIndex = chooseMemoryType(memoryRequirements.memoryTypeBits, properties);

	imageMemory = device.allocateMemory(allocateInfo);
	static_cast<void>(device.bindImageMemory(image, imageMemory, 0));
}

vk::CommandBuffer beginSingleTimeCommand() {
	vk::CommandBufferAllocateInfo allocateInfo{};
	allocateInfo.level = vk::CommandBufferLevel::ePrimary;
	allocateInfo.commandPool = commandPool;
	allocateInfo.commandBufferCount = 1;

	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto commandBuffer = device.allocateCommandBuffers(allocateInfo).at(0);
	static_cast<void>(commandBuffer.begin(beginInfo));
	return commandBuffer;
}

void endSingleTimeCommand(vk::CommandBuffer commandBuffer) {
	static_cast<void>(commandBuffer.end());

	vk::SubmitInfo submitInfo{};
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	static_cast<void>(queue.submit(1, &submitInfo, nullptr));
	static_cast<void>(queue.waitIdle());
	device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
	vk::BufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = size;

	auto commandBuffer = beginSingleTimeCommand();
	commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);
	endSingleTimeCommand(commandBuffer);
}

void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t imageWidth, uint32_t imageHeight) {
	vk::Offset3D offset{};
	offset.x = 0;
	offset.y = 0;
	offset.z = 0;

	vk::Extent3D extent{};
	extent.width = imageWidth;
	extent.height = imageHeight;
	extent.depth = 1;

	vk::BufferImageCopy region{};
	region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageOffset = offset;
	region.imageExtent = extent;

	auto commandBuffer = beginSingleTimeCommand();
	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
	endSingleTimeCommand(commandBuffer);
}

void transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
	vk::ImageMemoryBarrier barrier{};
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = mipLevels;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	vk::PipelineStageFlags sourceStage, destinationStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
		barrier.srcAccessMask = vk::AccessFlags{};
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage = vk::PipelineStageFlagBits::eTransfer;
	}
	else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
		newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		sourceStage = vk::PipelineStageFlagBits::eTransfer;
		destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
	}

	auto commandBuffer = beginSingleTimeCommand();
	commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags{},
		0, nullptr, 0, nullptr, 1, &barrier);
	endSingleTimeCommand(commandBuffer);
}

//TODO: Use continuous memory for images
svh::Image loadTexture(uint32_t width, uint32_t height, uint32_t levels, vk::Format format, std::vector<uint8_t>& data) {
	svh::Image image{};
	svh::Buffer buffer{};

	createBuffer(data.size(), vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer.buffer, buffer.memory);

	auto imageData = device.mapMemory(buffer.memory, 0, data.size());
	std::memcpy(imageData, data.data(), data.size());
	device.unmapMemory(buffer.memory);

	createImage(width, height, levels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal, image.image, image.memory);
	transitionImageLayout(image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, levels);
	copyBufferToImage(buffer.buffer, image.image, width, height);
	transitionImageLayout(image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, levels);
	image.view = createImageView(image.image, format, vk::ImageAspectFlagBits::eColor, levels);

	device.destroyBuffer(buffer.buffer);
	device.freeMemory(buffer.memory);

	return image;
}

void loadMesh(tinygltf::Model& modelData, tinygltf::Mesh& meshData, glm::mat4 transform, svh::Type type) {
	for (auto& primitive : meshData.primitives) {
		auto& indexReference = modelData.bufferViews.at(primitive.indices);
		auto& indexData = modelData.buffers.at(indexReference.buffer);

		svh::Mesh mesh{};

		mesh.indexOffset = indices.size();
		mesh.indexLength = indexReference.byteLength / sizeof(uint16_t);

		indices.resize(mesh.indexOffset + mesh.indexLength);
		std::memcpy(indices.data() + mesh.indexOffset, indexData.data.data() + indexReference.byteOffset, indexReference.byteLength);

		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> texcoords;

		for (auto& attribute : primitive.attributes) {
			auto& accessor = modelData.accessors.at(attribute.second);
			auto& primitiveView = modelData.bufferViews.at(accessor.bufferView);
			auto& primitiveBuffer = modelData.buffers.at(primitiveView.buffer);

			if (attribute.first.compare("POSITION") == 0) {
				positions.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(positions.data(), primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
			else if (attribute.first.compare("NORMAL") == 0) {
				normals.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(normals.data(), primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
			else if (attribute.first.compare("TEXCOORD_0") == 0) {
				texcoords.resize(primitiveView.byteLength / sizeof(glm::vec2));
				std::memcpy(texcoords.data(), primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
		}

		mesh.vertexOffset = vertices.size();
		mesh.vertexLength = texcoords.size();

		for (auto index = 0u; index < mesh.vertexLength; index++) {
			svh::Vertex vertex{};
			vertex.position = transform * glm::vec4{ positions.at(index), 1.0f };
			vertex.normal = transform * glm::vec4{ normals.at(index), 0.0f };
			vertex.texture = texcoords.at(index);
			vertices.push_back(vertex);
		}

		auto& material = modelData.materials.at(primitive.material);
		for (auto& value : material.values) {
			if (!value.first.compare("baseColorTexture")) {
				auto& image = modelData.images.at(value.second.TextureIndex());
				mesh.texture = loadTexture(image.width, image.height, details.mipLevels, details.imageFormat, image.image);
				break;
			}
		}

		if (type == svh::Type::Mesh) {
			details.meshCount++;
			meshes.push_back(mesh);
		}
		else {
			auto origin = glm::vec3{ 0.0f }, normal = glm::vec3{ 0.0f };
			auto min = glm::vec3{ std::numeric_limits<float_t>::max() }, max = glm::vec3{ std::numeric_limits<float_t>::min() };

			for (auto index = 0u; index < mesh.vertexLength; index++) {
				auto& vertex = vertices.at(mesh.vertexOffset + index);

				origin += vertex.position;
				normal += vertex.normal;

				min.x = std::min(min.x, vertex.position.x);
				min.y = std::min(min.y, vertex.position.y);
				min.z = std::min(min.z, vertex.position.z);

				max.x = std::max(max.x, vertex.position.x);
				max.y = std::max(max.y, vertex.position.y);
				max.z = std::max(max.z, vertex.position.z);
			}

			svh::Portal portal{};
			portal.mesh = mesh;
			portal.origin = origin / static_cast<float_t>(mesh.vertexLength);
			portal.normal = glm::normalize(normal);
			portal.minBorders = min;
			portal.maxBorders = max;
			portal.matrix = transform;

			details.portalCount++;
			portals.push_back(portal);
		}


	}
}

void loadNode(tinygltf::Model& modelData, tinygltf::Node& nodeData, glm::mat4 transform, svh::Type type) {
	glm::mat4 scale{ 1.0f }, rotation{ 1.0f }, translation{ 1.0f };

	if (!nodeData.rotation.empty())
		rotation = glm::toMat4(glm::qua{ nodeData.rotation.at(3), nodeData.rotation.at(0), nodeData.rotation.at(1), nodeData.rotation.at(2) });
	for (auto index = 0u; index < nodeData.scale.size(); index++)
		scale[index][index] = nodeData.scale.at(index);
	for (auto index = 0u; index < nodeData.translation.size(); index++)
		translation[3][index] = nodeData.translation.at(index);

	transform = transform * translation * rotation * scale;

	if (type == svh::Type::Player || type == svh::Type::Observer) {
		auto& camera = type == svh::Type::Observer ? observerCamera : playerCamera;

		camera.position = transform * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
		camera.direction = transform * glm::vec4{ 0.0f, -1.0f, 0.0f, 0.0f };
		camera.up = transform * glm::vec4{ 0.0f, 0.0f, 1.0f, 0.0f };

		if (type == svh::Type::Player)
			previousPosition = playerCamera.position;
	}

	else {
		if (nodeData.mesh >= 0 && nodeData.mesh < modelData.meshes.size())
			loadMesh(modelData, modelData.meshes.at(nodeData.mesh), transform, type);

		for (auto& childIndex : nodeData.children)
			loadNode(modelData, modelData.nodes.at(childIndex), transform, type);
	}
}

void loadModel(std::string filename, svh::Type type) {
	std::string error, warning;
	tinygltf::Model modelData;

	auto result = objectLoader.LoadBinaryFromFile(&modelData, &error, &warning, "models/" + filename);

#ifndef NDEBUG
	if (!warning.empty())
		std::cout << "GLTF Warning: " << warning << std::endl;
	if (!error.empty())
		std::cout << "GLTF Error: " << error << std::endl;
	if (!result)
		return;
#endif

	auto& scene = modelData.scenes.at(modelData.defaultScene);
	for (auto& nodeIndex : scene.nodes)
		loadNode(modelData, modelData.nodes.at(nodeIndex), glm::mat4{ 1.0f }, type);

	if (type == svh::Type::Portal) {
		auto& blue = portals.at(details.portalCount - 2), & orange = portals.at(details.portalCount - 1);

		blue.transform = blue.matrix * glm::rotate(glm::radians(180.0f), glm::vec3{ 0.0f, 0.0f, 1.0f }) * glm::inverse(orange.matrix);
		orange.transform = orange.matrix * glm::rotate(glm::radians(180.0f), glm::vec3{ 0.0f, 0.0f, 1.0f }) * glm::inverse(blue.matrix);
	}
}

void createScene() {
	loadModel("Observer.glb", svh::Type::Observer);
	loadModel("Player.glb", svh::Type::Player);
	loadModel("Portal.glb", svh::Type::Portal);
	loadModel("Room1.glb", svh::Type::Mesh);
	loadModel("Room2.glb", svh::Type::Mesh);
}

//TODO: Implement swapchain recreation on window resize
void createSwapchain() {
	vk::SwapchainCreateInfoKHR swapchainInfo{};
	swapchainInfo.flags = vk::SwapchainCreateFlagsKHR{};
	swapchainInfo.surface = surface;
	swapchainInfo.minImageCount = details.imageCount;
	swapchainInfo.imageFormat = details.surfaceFormat.format;
	swapchainInfo.imageColorSpace = details.surfaceFormat.colorSpace;
	swapchainInfo.imageExtent = details.swapchainExtent;
	swapchainInfo.imageArrayLayers = 1;
	swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
	swapchainInfo.queueFamilyIndexCount = 0;
	swapchainInfo.pQueueFamilyIndices = nullptr;
	swapchainInfo.preTransform = details.swapchainTransform;
	swapchainInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
	swapchainInfo.presentMode = vk::PresentModeKHR::eImmediate;
	swapchainInfo.clipped = true;
	swapchainInfo.oldSwapchain = nullptr;

	swapchain = device.createSwapchainKHR(swapchainInfo);
	swapchainImages = device.getSwapchainImagesKHR(swapchain);
	details.imageCount = swapchainImages.size();

	for (auto& swapchainImage : swapchainImages)
		swapchainViews.push_back(createImageView(swapchainImage, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
}

void createRenderPass() {
	vk::AttachmentDescription colorAttachment{};
	colorAttachment.format = details.surfaceFormat.format;
	colorAttachment.samples = details.sampleCount;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
	colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentDescription depthAttachment{};
	depthAttachment.format = details.depthStencilFormat;
	depthAttachment.samples = details.sampleCount;
	depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
	depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
	depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::AttachmentDescription resolveAttachment{};
	resolveAttachment.format = details.surfaceFormat.format;
	resolveAttachment.samples = vk::SampleCountFlagBits::e1;
	resolveAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
	resolveAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	resolveAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	resolveAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	resolveAttachment.initialLayout = vk::ImageLayout::eUndefined;
	resolveAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	std::array<vk::AttachmentDescription, 3> attachments{ colorAttachment, depthAttachment, resolveAttachment };

	vk::AttachmentReference colorReference{};
	colorReference.attachment = 0;
	colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentReference depthReference{};
	depthReference.attachment = 1;
	depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::AttachmentReference resolveReference{};
	resolveReference.attachment = 2;
	resolveReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpass{};
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;
	subpass.pDepthStencilAttachment = &depthReference;
	subpass.pResolveAttachments = &resolveReference;

	vk::SubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
	dependency.srcAccessMask = vk::AccessFlags{};
	dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
	dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.attachmentCount = attachments.size();
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	renderPass = device.createRenderPass(renderPassInfo, nullptr);
}

vk::ShaderModule loadShader(std::string name, shaderc_shader_kind kind) {
	auto path = std::filesystem::current_path() / "shaders" / name;
	auto size = std::filesystem::file_size(path);

	std::ifstream file(path);
	std::vector<char> code(size);
	file.read(code.data(), size);

	auto shaderModule = shaderCompiler.CompileGlslToSpv(code.data(), code.size(), kind, name.c_str(), "main", shaderOptions);

#ifndef NDEBUG
	if (shaderModule.GetCompilationStatus() != shaderc_compilation_status_success) {
		if(!shaderModule.GetErrorMessage().empty())
			std::cout << "SHADERC Error: " << shaderModule.GetErrorMessage() << std::endl;
		return nullptr;
	}
#endif

	std::vector<uint32_t> data(shaderModule.cbegin(), shaderModule.cend());

	vk::ShaderModuleCreateInfo shaderInfo{};
	shaderInfo.flags = vk::ShaderModuleCreateFlags{};
	shaderInfo.codeSize = data.size() * sizeof(uint32_t);
	shaderInfo.pCode = data.data();
	
	return device.createShaderModule(shaderInfo);
}

void createPipelineLayout() {
	shaderOptions.SetOptimizationLevel(shaderc_optimization_level_performance);
	vertexShader = loadShader("vertex.vert", shaderc_glsl_vertex_shader);
	fragmentShader = loadShader("fragment.frag", shaderc_glsl_fragment_shader);

	vk::SamplerCreateInfo samplerInfo{};
	samplerInfo.magFilter = vk::Filter::eLinear;
	samplerInfo.minFilter = vk::Filter::eLinear;
	samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
	samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
	samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
	samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = vk::CompareOp::eAlways;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxAnisotropy = details.maxAnisotropy;
	samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 1.0f;

	sampler = device.createSampler(samplerInfo);

	vk::DescriptorSetLayoutBinding uniformLayoutBinding{};
	uniformLayoutBinding.binding = 0;
	uniformLayoutBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
	uniformLayoutBinding.descriptorCount = 1;
	uniformLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
	uniformLayoutBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
	samplerLayoutBinding.pImmutableSamplers = nullptr;

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings{ uniformLayoutBinding, samplerLayoutBinding };

	vk::DescriptorSetLayoutCreateInfo descriptorInfo{};
	descriptorInfo.bindingCount = bindings.size();
	descriptorInfo.pBindings = bindings.data();

	descriptorSetLayout = device.createDescriptorSetLayout(descriptorInfo);

	vk::PipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &descriptorSetLayout;
	layoutInfo.pushConstantRangeCount = 0;
	layoutInfo.pPushConstantRanges = nullptr;

	pipelineLayout = device.createPipelineLayout(layoutInfo);
}

void createPipelines() {
	vk::PipelineShaderStageCreateInfo vertexShaderInfo{};
	vertexShaderInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertexShaderInfo.module = vertexShader;
	vertexShaderInfo.pName = "main";
	vertexShaderInfo.pSpecializationInfo = nullptr;

	vk::PipelineShaderStageCreateInfo fragmentShaderInfo{};
	fragmentShaderInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragmentShaderInfo.module = fragmentShader;
	fragmentShaderInfo.pName = "main";
	fragmentShaderInfo.pSpecializationInfo = nullptr;

	std::array<vk::PipelineShaderStageCreateInfo, 2> renderShaderStages{ vertexShaderInfo, fragmentShaderInfo };

	vk::VertexInputBindingDescription bindingDescription{};
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(svh::Vertex);
	bindingDescription.inputRate = vk::VertexInputRate::eVertex;

	vk::VertexInputAttributeDescription positionDescription{};
	positionDescription.binding = 0;
	positionDescription.location = 0;
	positionDescription.format = vk::Format::eR32G32B32Sfloat;
	positionDescription.offset = offsetof(svh::Vertex, position);

	vk::VertexInputAttributeDescription normalDescription{};
	normalDescription.binding = 0;
	normalDescription.location = 1;
	normalDescription.format = vk::Format::eR32G32B32Sfloat;
	normalDescription.offset = offsetof(svh::Vertex, normal);

	vk::VertexInputAttributeDescription textureDescription{};
	textureDescription.binding = 0;
	textureDescription.location = 2;
	textureDescription.format = vk::Format::eR32G32Sfloat;
	textureDescription.offset = offsetof(svh::Vertex, texture);

	std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{ positionDescription, normalDescription, textureDescription };

	vk::PipelineVertexInputStateCreateInfo inputInfo{};
	inputInfo.vertexBindingDescriptionCount = 1;
	inputInfo.pVertexBindingDescriptions = &bindingDescription;
	inputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
	inputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	vk::PipelineInputAssemblyStateCreateInfo assemblyInfo{};
	assemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;
	assemblyInfo.primitiveRestartEnable = false;

	vk::PipelineRasterizationStateCreateInfo rasterizerInfo{};
	rasterizerInfo.depthClampEnable = false;
	rasterizerInfo.rasterizerDiscardEnable = false;
	rasterizerInfo.polygonMode = vk::PolygonMode::eFill;
	rasterizerInfo.lineWidth = 1.0f;
	rasterizerInfo.cullMode = vk::CullModeFlagBits::eBack;
	rasterizerInfo.frontFace = vk::FrontFace::eCounterClockwise;
	rasterizerInfo.depthBiasEnable = false;
	rasterizerInfo.depthBiasConstantFactor = 0.0f;
	rasterizerInfo.depthBiasClamp = 0.0f;
	rasterizerInfo.depthBiasSlopeFactor = 0.0f;

	vk::PipelineMultisampleStateCreateInfo multisamplingInfo{};
	multisamplingInfo.sampleShadingEnable = false;
	multisamplingInfo.rasterizationSamples = details.sampleCount;
	multisamplingInfo.minSampleShading = 1.0f;
	multisamplingInfo.pSampleMask = nullptr;
	multisamplingInfo.alphaToCoverageEnable = false;
	multisamplingInfo.alphaToOneEnable = false;

	vk::StencilOpState stencilOpState{};
	stencilOpState.failOp = vk::StencilOp::eKeep;
	stencilOpState.passOp = vk::StencilOp::eKeep;
	stencilOpState.depthFailOp = vk::StencilOp::eKeep;
	stencilOpState.compareOp = vk::CompareOp::eNever;
	stencilOpState.compareMask = 0xFF;
	stencilOpState.writeMask = 0xFF;

	vk::PipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.depthTestEnable = true;
	depthStencil.depthWriteEnable = true;
	depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
	depthStencil.depthBoundsTestEnable = false;
	depthStencil.minDepthBounds = 0.0f;
	depthStencil.maxDepthBounds = 1.0f;
	depthStencil.stencilTestEnable = false;
	depthStencil.front = stencilOpState;
	depthStencil.back = stencilOpState;

	vk::PipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA;
	blendAttachment.blendEnable = true;
	blendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
	blendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
	blendAttachment.colorBlendOp = vk::BlendOp::eAdd;
	blendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
	blendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
	blendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

	std::array<float_t, 4> blendConstants{ 0.0f, 0.0f, 0.0f, 0.0f };

	vk::PipelineColorBlendStateCreateInfo blendInfo{};
	blendInfo.logicOpEnable = VK_FALSE;
	blendInfo.logicOp = vk::LogicOp::eCopy;
	blendInfo.attachmentCount = 1;
	blendInfo.pAttachments = &blendAttachment;
	blendInfo.blendConstants = blendConstants;

	vk::Viewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float_t>(details.swapchainExtent.width);
	viewport.height = static_cast<float_t>(details.swapchainExtent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vk::Offset2D offset{};
	offset.x = 0;
	offset.y = 0;

	vk::Rect2D scissor{};
	scissor.offset = offset;
	scissor.extent = details.swapchainExtent;

	vk::PipelineViewportStateCreateInfo viewportInfo{};
	viewportInfo.viewportCount = 1;
	viewportInfo.pViewports = &viewport;
	viewportInfo.scissorCount = 1;
	viewportInfo.pScissors = &scissor;

	vk::GraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.stageCount = renderShaderStages.size();
	pipelineInfo.pStages = renderShaderStages.data();
	pipelineInfo.pVertexInputState = &inputInfo;
	pipelineInfo.pInputAssemblyState = &assemblyInfo;
	pipelineInfo.pRasterizationState = &rasterizerInfo;
	pipelineInfo.pMultisampleState = &multisamplingInfo;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &blendInfo;
	pipelineInfo.pViewportState = &viewportInfo;
	pipelineInfo.pDynamicState = nullptr;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = nullptr;
	pipelineInfo.basePipelineIndex = -1;

	graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;

	depthStencil.stencilTestEnable = true;
	stencilOpState.passOp = vk::StencilOp::eReplace;
	stencilOpState.compareOp = vk::CompareOp::eAlways;

	for (auto index = 0u; index < details.portalCount; index++) {
		stencilOpState.reference = index + 1;
		depthStencil.front = stencilOpState;
		portals.at(index).stencilPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
	}

	stencilOpState.passOp = vk::StencilOp::eKeep;
	stencilOpState.compareOp = vk::CompareOp::eEqual;

	for (auto index = 0u; index < details.portalCount; index++) {
		stencilOpState.reference = index + 1;
		depthStencil.front = stencilOpState;
		portals.at(index).renderPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
	}
}

void createFramebuffers() {
	createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
		details.surfaceFormat.format, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
		vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage.image, colorImage.memory);
	colorImage.view = createImageView(colorImage.image, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);

	createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
		details.depthStencilFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage.image, depthImage.memory);
	depthImage.view = createImageView(depthImage.image, details.depthStencilFormat, vk::ImageAspectFlagBits::eDepth, 1);

	for (auto& swapchainView : swapchainViews) {
		std::array<vk::ImageView, 3> attachments{ colorImage.view, depthImage.view, swapchainView };

		vk::FramebufferCreateInfo framebufferInfo{};
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = details.swapchainExtent.width;
		framebufferInfo.height = details.swapchainExtent.height;
		framebufferInfo.layers = 1;

		framebuffers.push_back(device.createFramebuffer(framebufferInfo));
	}
}

void createElementBuffers() {
	svh::Buffer stagingBuffer{};

	auto indexSize = indices.size() * sizeof(uint16_t);
	createBuffer(indexSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer.buffer, stagingBuffer.memory);

	auto indexData = device.mapMemory(stagingBuffer.memory, 0, indexSize);
	std::memcpy(indexData, indices.data(), indexSize);
	device.unmapMemory(stagingBuffer.memory);

	createBuffer(indexSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer.buffer, indexBuffer.memory);
	copyBuffer(stagingBuffer.buffer, indexBuffer.buffer, indexSize);

	device.destroyBuffer(stagingBuffer.buffer);
	device.freeMemory(stagingBuffer.memory);

	auto vertexSize = vertices.size() * sizeof(svh::Vertex);
	createBuffer(vertexSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer.buffer, stagingBuffer.memory);

	auto vertexData = device.mapMemory(stagingBuffer.memory, 0, vertexSize);
	std::memcpy(vertexData, vertices.data(), vertexSize);
	device.unmapMemory(stagingBuffer.memory);

	createBuffer(vertexSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer.buffer, vertexBuffer.memory);
	copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, vertexSize);

	device.destroyBuffer(stagingBuffer.buffer);
	device.freeMemory(stagingBuffer.memory);

	details.uniformStride = (details.portalCount + 1) * details.uniformAlignment;
	details.uniformSize = details.imageCount * details.uniformStride;
	createBuffer(details.uniformSize, vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		uniformBuffer.buffer, uniformBuffer.memory);
}

void updateDescriptorSet(svh::Mesh &mesh) {
	vk::DescriptorBufferInfo uniformInfo{};
	uniformInfo.buffer = uniformBuffer.buffer;
	uniformInfo.offset = 0;
	uniformInfo.range = details.uniformAlignment;

	vk::DescriptorImageInfo samplerInfo{};
	samplerInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	samplerInfo.imageView = mesh.texture.view;
	samplerInfo.sampler = sampler;

	vk::WriteDescriptorSet uniformWrite{};
	uniformWrite.dstSet = mesh.descriptorSet;
	uniformWrite.dstBinding = 0;
	uniformWrite.dstArrayElement = 0;
	uniformWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
	uniformWrite.descriptorCount = 1;
	uniformWrite.pBufferInfo = &uniformInfo;
	uniformWrite.pImageInfo = nullptr;
	uniformWrite.pTexelBufferView = nullptr;

	vk::WriteDescriptorSet samplerWrite{};
	samplerWrite.dstSet = mesh.descriptorSet;
	samplerWrite.dstBinding = 1;
	samplerWrite.dstArrayElement = 0;
	samplerWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
	samplerWrite.descriptorCount = 1;
	samplerWrite.pBufferInfo = nullptr;
	samplerWrite.pImageInfo = &samplerInfo;
	samplerWrite.pTexelBufferView = nullptr;

	std::array<vk::WriteDescriptorSet, 2> descriptorWrites{ uniformWrite, samplerWrite };
	device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void createDescriptorSets() {
	auto descriptorCount = details.portalCount + details.meshCount;

	vk::DescriptorPoolSize uniformSize{};
	uniformSize.type = vk::DescriptorType::eUniformBufferDynamic;
	uniformSize.descriptorCount = descriptorCount;

	vk::DescriptorPoolSize samplerSize{};
	samplerSize.type = vk::DescriptorType::eCombinedImageSampler;
	samplerSize.descriptorCount = descriptorCount;

	std::array<vk::DescriptorPoolSize, 2> poolSizes{ uniformSize, samplerSize };

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.poolSizeCount = poolSizes.size();
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = descriptorCount;

	descriptorPool = device.createDescriptorPool(poolInfo);

	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts{ descriptorCount, descriptorSetLayout };

	vk::DescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = descriptorSetLayouts.size();
	allocateInfo.pSetLayouts = descriptorSetLayouts.data();

	auto descriptorSets = device.allocateDescriptorSets(allocateInfo);

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
		auto& mesh = portals.at(portalIndex).mesh;
		mesh.descriptorSet = descriptorSets.at(portalIndex);
		updateDescriptorSet(mesh);
	}
		
	for (auto meshIndex = 0u; meshIndex < details.meshCount; meshIndex++) {
		auto& mesh = meshes.at(meshIndex);
		mesh.descriptorSet = descriptorSets.at(details.portalCount + meshIndex);
		updateDescriptorSet(mesh);
	}
}

void createCommandBuffers() {
	vk::CommandBufferAllocateInfo allocateInfo{};
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = vk::CommandBufferLevel::ePrimary;
	allocateInfo.commandBufferCount = details.imageCount;

	commandBuffers = device.allocateCommandBuffers(allocateInfo);

	for (auto imageIndex = 0u; imageIndex < details.imageCount; imageIndex++) {
		auto& commandBuffer = commandBuffers.at(imageIndex);
		auto uniformOffset = imageIndex * details.uniformStride + details.portalCount * details.uniformAlignment;

		vk::DeviceSize bufferOffset = 0;

		vk::Offset2D areaOffset{};
		areaOffset.x = 0;
		areaOffset.y = 0;

		vk::ClearValue colorClear{};
		colorClear.color.float32.at(0) = 0.0f;
		colorClear.color.float32.at(1) = 0.0f;
		colorClear.color.float32.at(2) = 0.0f;
		colorClear.color.float32.at(3) = 1.0f;

		vk::ClearValue depthStencilClear{};
		depthStencilClear.depthStencil.depth = 1.0f;
		depthStencilClear.depthStencil.stencil = 0;

		std::array<vk::ClearValue, 2> clearValues{ colorClear, depthStencilClear };

		vk::Rect2D rect{};
		rect.offset = areaOffset;
		rect.extent = details.swapchainExtent;

		vk::ClearRect clearRect{};
		clearRect.baseArrayLayer = 0;
		clearRect.layerCount = 2;
		clearRect.rect = rect;

		vk::ClearAttachment clearAttachment{};
		clearAttachment.aspectMask = vk::ImageAspectFlagBits::eDepth;
		clearAttachment.clearValue = depthStencilClear;
		clearAttachment.colorAttachment = -1;

		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.pInheritanceInfo = nullptr;

		vk::RenderPassBeginInfo renderPassInfo{};
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = framebuffers.at(imageIndex);
		renderPassInfo.renderArea.offset = areaOffset;
		renderPassInfo.renderArea.extent = details.swapchainExtent;
		renderPassInfo.clearValueCount = clearValues.size();
		renderPassInfo.pClearValues = clearValues.data();

		static_cast<void>(commandBuffer.begin(beginInfo));
		commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
		commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer.buffer, &bufferOffset);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
		commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

		for (auto& mesh : meshes) {
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &mesh.descriptorSet, 1, &uniformOffset);
			commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
		}

		for (auto& portal : portals) {
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, portal.stencilPipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &portal.mesh.descriptorSet, 1, &uniformOffset);
			commandBuffer.drawIndexed(portal.mesh.indexLength, 1, portal.mesh.indexOffset, portal.mesh.vertexOffset, 0);
		}

		commandBuffer.clearAttachments(1, &clearAttachment, 1, &clearRect);

		for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
			auto& portal = portals.at(portalIndex);
			uniformOffset = imageIndex * details.uniformStride + portalIndex * details.uniformAlignment;

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, portal.renderPipeline);

			for (auto& mesh : meshes) {
				commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &mesh.descriptorSet, 1, &uniformOffset);
				commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
			}
		}

		commandBuffer.endRenderPass();
		static_cast<void>(commandBuffer.end());
	}
}

void createSyncObject() {
	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	vk::SemaphoreCreateInfo semaphoreInfo{};

	for (auto imageIndex = 0u; imageIndex < details.imageCount; imageIndex++) {
		frameFences.push_back(device.createFence(fenceInfo));
		orderFences.push_back(nullptr);
		availableSemaphores.push_back(device.createSemaphore(semaphoreInfo));
		finishedSemaphores.push_back(device.createSemaphore(semaphoreInfo));
	}
}

void setup() {
	initializeCore();
	createDevice();
	createScene();
	createSwapchain();
	createRenderPass();
	createPipelineLayout();
	createPipelines();
	createFramebuffers();
	createElementBuffers();
	createDescriptorSets();
	createCommandBuffers();
	createSyncObject();
}

//TODO: Remove constant memory map-unmaps
void updateScene(uint32_t imageIndex) {
	auto moveDelta = state.timeDelta * 12.0f, turnDelta = state.timeDelta * glm::radians(120.0f);
	auto vectorCount = std::abs(controls.keyW - controls.keyS) + std::abs(controls.keyA - controls.keyD);
	auto& camera = controls.observer ? observerCamera : playerCamera;

	if (vectorCount > 0)
		moveDelta /= std::sqrt(vectorCount);

	if (controls.observer) {
		auto left = glm::normalize(glm::cross(camera.up, camera.direction));

		if (controls.keyW || controls.keyS)
			camera.position += (controls.keyW - controls.keyS) * moveDelta * glm::normalize(camera.direction + camera.up);
		if (controls.keyA || controls.keyD)
			camera.position += (controls.keyA - controls.keyD) * moveDelta * left;
		if (controls.keyR || controls.keyF)
			camera.position += (controls.keyR - controls.keyF) * moveDelta * camera.direction;
		if (controls.keyQ || controls.keyE) {
			auto rotation = glm::rotate((controls.keyQ - controls.keyE) * turnDelta, glm::vec3{ 0.0f, 0.0f, 1.0f });

			auto direction = glm::normalize(glm::vec2{ camera.direction });
			camera.position += camera.position.z * glm::vec3{ direction, 0.0f };
			camera.direction = rotation * glm::vec4{ camera.direction, 0.0f };

			direction = glm::normalize(glm::vec2{ camera.direction });
			camera.position -= camera.position.z * glm::vec3{ direction, 0.0f };
			camera.up = rotation * glm::vec4{ camera.up, 0.0f };
		}
	}
	else {
		previousPosition = camera.position;

		auto left = glm::normalize(glm::cross(camera.up, camera.direction));

		camera.direction = glm::normalize(glm::vec3{ glm::rotate(turnDelta * controls.deltaY, left) *
														  glm::rotate(turnDelta * controls.deltaX, camera.up) *
														  glm::vec4{camera.direction, 0.0f} });

		left = glm::normalize(glm::cross(camera.up, camera.direction));

		camera.position += moveDelta * (controls.keyW - controls.keyS) * camera.direction +
			moveDelta * (controls.keyA - controls.keyD) * left;

		controls.deltaX = 0.0f;
		controls.deltaY = 0.0f;

		auto coefficient = 0.0f, distance = glm::length(camera.position - previousPosition);
		auto direction = glm::normalize(camera.position - previousPosition);

		for (auto portal : portals) {
			if (svh::epsilon < distance && glm::intersectRayPlane(previousPosition, direction, portal.origin, portal.normal, coefficient)) {
				auto point = previousPosition + coefficient * direction;

				if (point.x >= portal.minBorders.x && point.y >= portal.minBorders.y && point.z >= portal.minBorders.z &&
					point.x <= portal.maxBorders.x && point.y <= portal.maxBorders.y && point.z <= portal.maxBorders.z &&
					0 <= coefficient && distance >= coefficient) {

					camera.position = portal.transform * glm::vec4{ camera.position, 1.0f };
					camera.direction = portal.transform * glm::vec4{ camera.direction, 0.0f };
					camera.up = portal.transform * glm::vec4{ camera.up, 0.0f };

					previousPosition = camera.position;

					break;
				}
			}
		}
	}

	auto projection = glm::perspective(glm::radians(45.0f),
		static_cast<float_t>(details.swapchainExtent.width) / static_cast<float_t>(details.swapchainExtent.height), 0.001f, 100.0f);
	projection[1][1] *= -1;

	auto data = static_cast<uint8_t*>(device.mapMemory(uniformBuffer.memory, imageIndex * details.uniformStride, details.uniformStride));
	auto transform = projection * glm::lookAt(camera.position, camera.position + camera.direction, camera.up);
	std::memcpy(data + details.portalCount * details.uniformAlignment, &transform, sizeof(glm::mat4));

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
		auto& portal = portals.at(portalIndex);

		svh::Camera portalCamera{};
		portalCamera.position = portal.transform * glm::vec4{ camera.position, 1.0f };
		portalCamera.direction = portal.transform * glm::vec4{ camera.direction, 0.0f };
		portalCamera.up = portal.transform * glm::vec4{ camera.up, 0.0f };

		transform = projection * glm::lookAt(portalCamera.position, portalCamera.position + portalCamera.direction, portalCamera.up);
		std::memcpy(data + portalIndex * details.uniformAlignment, &transform, sizeof(glm::mat4));
	}

	device.unmapMemory(uniformBuffer.memory);
}

//TODO: Split present and retrieve
void draw() {
	state.frameCount = 0;
	state.checkPoint = 0.0f;
	state.currentTime = std::chrono::high_resolution_clock::now();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		state.previousTime = state.currentTime;
		state.currentTime = std::chrono::high_resolution_clock::now();
		state.timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(state.currentTime - state.previousTime).count();
		state.checkPoint += state.timeDelta;
		state.frameCount++;

		updateScene(state.currentImage);

		static_cast<void>(device.waitForFences(1, &frameFences[state.currentImage], true, std::numeric_limits<uint64_t>::max()));
		auto imageIndex = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(),
			availableSemaphores.at(state.currentImage), nullptr).value;

		if (orderFences.at(imageIndex))
			static_cast<void>(device.waitForFences(1, &orderFences.at(imageIndex), true, std::numeric_limits<uint64_t>::max()));

		orderFences.at(imageIndex) = frameFences.at(state.currentImage);

		std::array<vk::Semaphore, 1> waitSemaphores{ availableSemaphores[state.currentImage] };
		std::array<vk::Semaphore, 1> signalSemaphores{ finishedSemaphores[state.currentImage] };
		std::array<vk::PipelineStageFlags, 1> waitStages{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

		vk::SubmitInfo submitInfo{};
		submitInfo.waitSemaphoreCount = waitSemaphores.size();
		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.pWaitDstStageMask = waitStages.data();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers.at(imageIndex);
		submitInfo.signalSemaphoreCount = signalSemaphores.size();
		submitInfo.pSignalSemaphores = signalSemaphores.data();

		static_cast<void>(device.resetFences(1, &frameFences.at(state.currentImage)));
		static_cast<void>(queue.submit(1, &submitInfo, frameFences.at(state.currentImage)));

		vk::PresentInfoKHR presentInfo{};
		presentInfo.waitSemaphoreCount = signalSemaphores.size();
		presentInfo.pWaitSemaphores = signalSemaphores.data();
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		static_cast<void>(queue.presentKHR(presentInfo));
		state.currentImage = (state.currentImage + 1) % details.imageCount;

		if (state.checkPoint > 1.0f) {
			auto title = std::to_string(state.frameCount);
			glfwSetWindowTitle(window, title.c_str());
			state.checkPoint = 0.0f;
			state.frameCount = 0;
		}
	}

	static_cast<void>(device.waitIdle());
}

void clear() {
	for (auto& finishedSemaphore : finishedSemaphores)
		device.destroySemaphore(finishedSemaphore);
	for (auto& availableSemaphore : availableSemaphores)
		device.destroySemaphore(availableSemaphore);
	for (auto& frameFence : frameFences)
		device.destroyFence(frameFence);
	device.destroyDescriptorPool(descriptorPool);
	device.destroyBuffer(uniformBuffer.buffer);
	device.freeMemory(uniformBuffer.memory);
	device.destroyBuffer(vertexBuffer.buffer);
	device.freeMemory(vertexBuffer.memory);
	device.destroyBuffer(indexBuffer.buffer);
	device.freeMemory(indexBuffer.memory);
	for (auto& swapchainFramebuffer : framebuffers)
		device.destroyFramebuffer(swapchainFramebuffer);
	device.destroyImageView(colorImage.view);
	device.destroyImage(colorImage.image);
	device.freeMemory(colorImage.memory);
	device.destroyImageView(depthImage.view);
	device.destroyImage(depthImage.image);
	device.freeMemory(depthImage.memory);
	for (auto& mesh : meshes) {
		device.destroyImageView(mesh.texture.view);
		device.destroyImage(mesh.texture.image);
		device.freeMemory(mesh.texture.memory);
	}
	for (auto& portal : portals) {
		device.destroyPipeline(portal.renderPipeline);
		device.destroyPipeline(portal.stencilPipeline);
		device.destroyImageView(portal.mesh.texture.view);
		device.destroyImage(portal.mesh.texture.image);
		device.freeMemory(portal.mesh.texture.memory);
	}
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroySampler(sampler);
	device.destroyShaderModule(stencilShader);
	device.destroyShaderModule(fragmentShader);
	device.destroyShaderModule(vertexShader);
	device.destroyRenderPass(renderPass);
	for (auto& swapchainView : swapchainViews)
		device.destroyImageView(swapchainView);
	device.destroySwapchainKHR(swapchain);
	device.destroyCommandPool(commandPool);
	device.destroy();
	instance.destroySurfaceKHR(surface);
#ifndef NDEBUG
	instance.destroyDebugUtilsMessengerEXT(messenger, nullptr, functionLoader);
#endif
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

int main() {
	setup();
	draw();
	clear();
}
