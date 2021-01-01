#include "Svanhild.hpp"

GLFWwindow* window;
svh::Controls controls;
svh::State state;
svh::Camera observerCamera, playerCamera;
tinygltf::TinyGLTF objectLoader;

std::vector<uint16_t> indices;
std::vector<svh::Vertex> vertices;
std::vector<svh::Mesh> meshes;

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
	VkDebugUtilsMessageTypeFlagsEXT type,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {
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
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
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
	window = glfwCreateWindow(1280, 720, "", nullptr, nullptr);
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

void createDevice() {
	physicalDevice = pickPhysicalDevice();
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

//TODO: Check format availability
svh::Details generateDetails() {
	svh::Details temporaryDetails;

	temporaryDetails.meshCount = 0;
	temporaryDetails.portalCount = 0;

	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	glfwGetFramebufferSize(window, reinterpret_cast<int*>(&surfaceCapabilities.currentExtent.width),
		reinterpret_cast<int*>(&surfaceCapabilities.currentExtent.height));
	surfaceCapabilities.currentExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
		std::min(surfaceCapabilities.maxImageExtent.width,
			surfaceCapabilities.currentExtent.width));
	surfaceCapabilities.currentExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
		std::min(surfaceCapabilities.maxImageExtent.height,
			surfaceCapabilities.currentExtent.height));

	temporaryDetails.imageCount = std::min(surfaceCapabilities.minImageCount + 1, surfaceCapabilities.maxImageCount);
	temporaryDetails.swapchainExtent = surfaceCapabilities.currentExtent;
	temporaryDetails.swapchainTransform = surfaceCapabilities.currentTransform;

	auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	temporaryDetails.surfaceFormat = surfaceFormats.front();

	for (auto& surfaceFormat : surfaceFormats)
		if (surfaceFormat.format == vk::Format::eB8G8R8A8Srgb &&
			surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
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

//TODO: Implement swapchain recreation on window resize
void createSwapchain() {
	details = generateDetails();

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
		swapchainViews.push_back(
			createImageView(swapchainImage, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
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
svh::Image
loadTexture(uint32_t width, uint32_t height, uint32_t levels, vk::Format format, std::vector<uint8_t>& data) {
	svh::Image image{};
	svh::Buffer buffer{};

	createBuffer(data.size(), vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer.buffer,
		buffer.memory);

	auto imageData = device.mapMemory(buffer.memory, 0, data.size());
	std::memcpy(imageData, data.data(), data.size());
	device.unmapMemory(buffer.memory);

	createImage(width, height, levels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal, image.image, image.memory);
	transitionImageLayout(image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, levels);
	copyBufferToImage(buffer.buffer, image.image, width, height);
	transitionImageLayout(image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		levels);
	image.view = createImageView(image.image, format, vk::ImageAspectFlagBits::eColor, levels);

	device.destroyBuffer(buffer.buffer);
	device.freeMemory(buffer.memory);

	return image;
}

void loadMesh(tinygltf::Model& modelData, tinygltf::Mesh& meshData, glm::mat4 transform) {
	for (auto& primitive : meshData.primitives) {
		auto& indexReference = modelData.bufferViews.at(primitive.indices);
		auto& indexData = modelData.buffers.at(indexReference.buffer);

		svh::Mesh mesh{};

		mesh.indexOffset = indices.size();
		mesh.indexLength = indexReference.byteLength / sizeof(uint16_t);

		indices.resize(mesh.indexOffset + mesh.indexLength);
		std::memcpy(indices.data() + mesh.indexOffset, indexData.data.data() + indexReference.byteOffset,
			indexReference.byteLength);

		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> texcoords;

		for (auto& attribute : primitive.attributes) {
			auto& accessor = modelData.accessors.at(attribute.second);
			auto& primitiveView = modelData.bufferViews.at(accessor.bufferView);
			auto& primitiveBuffer = modelData.buffers.at(primitiveView.buffer);

			if (attribute.first.compare("POSITION") == 0) {
				positions.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(positions.data(),
					primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
			else if (attribute.first.compare("NORMAL") == 0) {
				normals.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(normals.data(),
					primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
			else if (attribute.first.compare("TEXCOORD_0") == 0) {
				texcoords.resize(primitiveView.byteLength / sizeof(glm::vec2));
				std::memcpy(texcoords.data(),
					primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
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

		mesh.transform = transform;

		auto& material = modelData.materials.at(primitive.material);
		for (auto& value : material.values) {
			if (!value.first.compare("baseColorTexture")) {
				auto& image = modelData.images.at(value.second.TextureIndex());
				mesh.texture = loadTexture(image.width, image.height, details.mipLevels, details.imageFormat,
					image.image);
				break;
			}
		}

		meshes.push_back(mesh);
	}
}

void loadNode(tinygltf::Model& modelData, tinygltf::Node& nodeData, glm::mat4 transform) {
	glm::mat4 scale{ 1.0f }, rotation{ 1.0f }, translation{ 1.0f };

	if (!nodeData.rotation.empty())
		rotation = glm::toMat4(glm::qua{
			nodeData.rotation.at(3), nodeData.rotation.at(0), nodeData.rotation.at(1), nodeData.rotation.at(2)});
	for (auto index = 0u; index < nodeData.scale.size(); index++)
		scale[index][index] = nodeData.scale.at(index);
	for (auto index = 0u; index < nodeData.translation.size(); index++)
		translation[3][index] = nodeData.translation.at(index);

	transform = transform * translation * rotation * scale;

	if (nodeData.mesh >= 0 && nodeData.mesh < modelData.meshes.size())
		loadMesh(modelData, modelData.meshes.at(nodeData.mesh), transform);

	for (auto& childIndex : nodeData.children)
		loadNode(modelData, modelData.nodes.at(childIndex), transform);
}

void loadModel(std::string filename) {
	std::string error, warning;
	tinygltf::Model modelData;

	auto result = objectLoader.LoadBinaryFromFile(&modelData, &error, &warning, filename);

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
		loadNode(modelData, modelData.nodes.at(nodeIndex), glm::mat4{ 1.0f });
}

void createScene() {
	observerCamera.position = glm::vec4{ 0.0f, 10.0f, 10.0f, 1.0f };
	observerCamera.direction = glm::vec4{ 0.0f, -std::sqrt(2.0f) / 2.0f, -std::sqrt(2.0f) / 2.0f, 0.0f };
	observerCamera.up = glm::vec4{ 0.0f, -std::sqrt(2.0f) / 2.0f, std::sqrt(2.0f) / 2.0f, 0.0f };

	playerCamera.position = glm::vec4{ 0.0f, 4.0f, 2.0f, 1.0f };
	playerCamera.direction = glm::vec4{ 0.0f, -1.0f, 0.0f, 0.0f };
	playerCamera.up = glm::vec4{ 0.0f, 0.0f, 1.0f, 0.0f };

	loadModel("models/Portal.glb");
	details.portalCount = meshes.size();

	loadModel("models/Room1.glb");
	loadModel("models/Room2.glb");
	details.meshCount = meshes.size();
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
	dependency.srcStageMask =
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
	dependency.srcAccessMask = vk::AccessFlags{};
	dependency.dstStageMask =
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
	dependency.dstAccessMask =
		vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.attachmentCount = attachments.size();
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	renderPass = device.createRenderPass(renderPassInfo, nullptr);
}

vk::ShaderModule loadShader(std::string name) {
	auto path = std::filesystem::current_path() / name;
	auto size = std::filesystem::file_size(path);

	std::ifstream file(name, std::ios::binary);
	std::vector<uint32_t> data(size / sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(data.data()), size);

	vk::ShaderModuleCreateInfo shaderInfo{};
	shaderInfo.flags = vk::ShaderModuleCreateFlags{};
	shaderInfo.codeSize = size;
	shaderInfo.pCode = data.data();

	return device.createShaderModule(shaderInfo);
}

void createPipelineLayout() {
	vertexShader = loadShader("shaders/vertex.spv");
	fragmentShader = loadShader("shaders/fragment.spv");

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

	std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{
			positionDescription, normalDescription, textureDescription };

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
	viewport.width = details.swapchainExtent.width;
	viewport.height = details.swapchainExtent.height;
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
		meshes.at(index).stencilPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
	}

	stencilOpState.passOp = vk::StencilOp::eKeep;
	stencilOpState.compareOp = vk::CompareOp::eEqual;

	for (auto index = 0u; index < details.portalCount; index++) {
		stencilOpState.reference = index + 1;
		depthStencil.front = stencilOpState;
		meshes.at(index).renderPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
	}
}

void createFramebuffers() {
	createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
		details.surfaceFormat.format, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
		vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage.image, colorImage.memory);
	colorImage.view = createImageView(colorImage.image, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor,
		1);

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

void createDescriptorSets() {
	vk::DescriptorPoolSize uniformSize{};
	uniformSize.type = vk::DescriptorType::eUniformBufferDynamic;
	uniformSize.descriptorCount = details.meshCount;

	vk::DescriptorPoolSize samplerSize{};
	samplerSize.type = vk::DescriptorType::eCombinedImageSampler;
	samplerSize.descriptorCount = details.meshCount;

	std::array<vk::DescriptorPoolSize, 2> poolSizes{ uniformSize, samplerSize };

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.poolSizeCount = poolSizes.size();
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = details.meshCount;

	descriptorPool = device.createDescriptorPool(poolInfo);

	for (auto& mesh : meshes) {
		std::vector<vk::DescriptorSetLayout> layouts{ details.meshCount, descriptorSetLayout };

		vk::DescriptorSetAllocateInfo allocateInfo{};
		allocateInfo.descriptorPool = descriptorPool;
		allocateInfo.descriptorSetCount = 1;
		allocateInfo.pSetLayouts = &descriptorSetLayout;

		mesh.descriptorSet = device.allocateDescriptorSets(allocateInfo).front();

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
}

//TODO: Implement an indexing function
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

		for (auto meshIndex = details.portalCount; meshIndex < details.meshCount; meshIndex++) {
			auto& mesh = meshes.at(meshIndex);

			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
				&mesh.descriptorSet, 1, &uniformOffset);
			commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
		}

		for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
			auto& portal = meshes.at(portalIndex);

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, portal.stencilPipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
				&portal.descriptorSet, 1, &uniformOffset);
			commandBuffer.drawIndexed(portal.indexLength, 1, portal.indexOffset, portal.vertexOffset, 0);
		}

		commandBuffer.clearAttachments(1, &clearAttachment, 1, &clearRect);

		for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
			auto& portal = meshes.at(portalIndex);
			uniformOffset = imageIndex * details.uniformStride + portalIndex * details.uniformAlignment;

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, portal.renderPipeline);

			for (auto meshIndex = details.portalCount; meshIndex < details.meshCount; meshIndex++) {
				auto& mesh = meshes.at(meshIndex);

				commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
					&mesh.descriptorSet, 1, &uniformOffset);
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
	createSwapchain();
	createScene();
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
	auto moveDelta = state.timeDelta * 12.0f, turnDelta = state.timeDelta * glm::radians(90.0f);
	auto vectorCount = std::abs(controls.keyW - controls.keyS) + std::abs(controls.keyA - controls.keyD);

	if (vectorCount > 0)
		moveDelta /= std::sqrt(vectorCount);

	//TODO: Refactor here like the player camera
	if (controls.observer) {
		auto left = glm::normalize(glm::cross(observerCamera.up, observerCamera.direction));

		if (controls.keyW || controls.keyS)
			observerCamera.position += (controls.keyW - controls.keyS) * moveDelta *
			(observerCamera.direction + observerCamera.up) / std::sqrt(2.0f);
		if (controls.keyA || controls.keyD)
			observerCamera.position += (controls.keyA - controls.keyD) * moveDelta * left;
		if (controls.keyR || controls.keyF)
			observerCamera.position += (controls.keyR - controls.keyF) * moveDelta * observerCamera.direction;
		if (controls.keyQ || controls.keyE) {
			glm::mat4 rotation = glm::rotate((controls.keyQ - controls.keyE) * turnDelta, glm::vec3{ 0.0f, 0.0f, 1.0f });

			auto direction = glm::normalize(glm::vec2{ observerCamera.direction });
			observerCamera.position += observerCamera.position.z * glm::vec3{ direction, 0.0f };
			observerCamera.direction = rotation * glm::vec4{ observerCamera.direction, 0.0f };

			direction = glm::normalize(glm::vec2{ observerCamera.direction });
			observerCamera.position -= observerCamera.position.z * glm::vec3{ direction, 0.0f };
			observerCamera.up = rotation * glm::vec4{ observerCamera.up, 0.0f };
		}
	}
	else {
		auto left = glm::normalize(glm::cross(playerCamera.up, playerCamera.direction));

		playerCamera.direction = glm::normalize(glm::vec3{ glm::rotate(turnDelta * controls.deltaY, left) *
														  glm::rotate(turnDelta * controls.deltaX, playerCamera.up) *
														  glm::vec4{playerCamera.direction, 0.0f} });

		left = glm::normalize(glm::cross(playerCamera.up, playerCamera.direction));

		playerCamera.position += moveDelta * (controls.keyW - controls.keyS) * playerCamera.direction +
			moveDelta * (controls.keyA - controls.keyD) * left;

		controls.deltaX = 0.0f;
		controls.deltaY = 0.0f;
	}

	auto& camera = controls.observer ? observerCamera : playerCamera;
	auto projection = glm::perspective(glm::radians(45.0f),
		static_cast<float_t>(details.swapchainExtent.width) /
		static_cast<float_t>(details.swapchainExtent.height), 0.01f, 100.0f);
	projection[1][1] *= -1;

	auto data = static_cast<uint8_t*>(device.mapMemory(uniformBuffer.memory, imageIndex * details.uniformStride,
		details.uniformStride));
	auto transform = projection * glm::lookAt(camera.position, camera.position + camera.direction, camera.up);
	std::memcpy(data + details.portalCount * details.uniformAlignment, &transform, sizeof(glm::mat4));

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
		auto& sourceTransform = meshes.at(portalIndex).transform,
			& destinationTransform = meshes.at(portalIndex + 1 - portalIndex % 2 * 2).transform;
		auto cameraTransform = sourceTransform * glm::rotate(glm::radians(180.0f), glm::vec3{ 0.0f, 0.0f, 1.0f }) *
			glm::inverse(destinationTransform);

		svh::Camera portalCamera{};
		portalCamera.position = cameraTransform * glm::vec4{ camera.position, 1.0f };
		portalCamera.direction = cameraTransform * glm::vec4{ camera.direction, 0.0f };
		portalCamera.up = cameraTransform * glm::vec4{ camera.up, 0.0f };

		transform = projection *
			glm::lookAt(portalCamera.position, portalCamera.position + portalCamera.direction, portalCamera.up);
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
		state.timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(
			state.currentTime - state.previousTime).count();
		state.checkPoint += state.timeDelta;
		state.frameCount++;

		static_cast<void>(device.waitForFences(1, &frameFences[state.currentImage], true,
			std::numeric_limits<uint64_t>::max()));
		auto imageIndex = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(),
			availableSemaphores.at(state.currentImage), nullptr).value;

		if (orderFences.at(imageIndex))
			static_cast<void>(device.waitForFences(1, &orderFences.at(imageIndex), true,
				std::numeric_limits<uint64_t>::max()));

		orderFences.at(imageIndex) = frameFences.at(state.currentImage);

		std::array<vk::Semaphore, 1> waitSemaphores{ availableSemaphores[state.currentImage] };
		std::array<vk::Semaphore, 1> signalSemaphores{ finishedSemaphores[state.currentImage] };
		std::array<vk::PipelineStageFlags, 1> waitStages{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

		updateScene(state.currentImage);

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
		device.destroyPipeline(mesh.renderPipeline);
		device.destroyPipeline(mesh.stencilPipeline);
		device.destroyImageView(mesh.texture.view);
		device.destroyImage(mesh.texture.image);
		device.freeMemory(mesh.texture.memory);
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
