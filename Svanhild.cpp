#include "Svanhild.hpp"

GLFWwindow* window;
tinygltf::TinyGLTF objectLoader;
shaderc::Compiler shaderCompiler;
shaderc::CompileOptions shaderOptions;

svh::Controls controls;
svh::State state;
svh::Camera camera;
std::vector<uint16_t> indices;
std::vector<svh::Vertex> vertices;
std::vector<std::string> imageNames;
std::vector<svh::Image> textures;
std::vector<svh::Mesh> meshes;
std::vector<svh::Portal> portals;

vk::Instance instance;
vk::SurfaceKHR surface;
vk::PhysicalDevice physicalDevice;
vk::Device device;
uint32_t queueFamilyIndex;
vk::Queue mainQueue;
vk::CommandPool mainCommandPool;
svh::Details details;
vk::SwapchainKHR swapchain;
std::vector<vk::Image> swapchainImages;
std::vector<vk::ImageView> swapchainViews;
vk::Sampler sampler;
vk::RenderPass renderPass;
vk::ShaderModule computeShader, vertexShader, fragmentShader;
vk::DescriptorSetLayout computeSetLayout, graphicsSetLayout;
vk::PipelineLayout computePipelineLayout, graphicsPipelineLayout;
vk::Pipeline computePipeline, graphicsPipeline, stencilPipeline, renderPipeline;
std::vector <svh::Image> colorImages, depthImages;
std::vector<vk::Framebuffer> framebuffers;
svh::Buffer indexBuffer, vertexBuffer, uniformBuffer;
vk::DescriptorPool descriptorPool;

uint8_t* deviceMemory;

std::vector<vk::Queue> concurrentQueues;
std::vector<vk::CommandPool> threadCommandPools;
std::vector<vk::CommandBuffer> commandBuffers;
std::vector<svh::Status> commandBufferStatuses;

std::vector<std::thread> recordThreads, renderThreads;
std::vector<vk::Fence> submitFences;
std::vector<vk::Semaphore> submitSemaphores, presentSemaphores;
std::vector<std::unique_ptr<std::binary_semaphore>> swapchainSemaphores;
std::mutex acquireMutex;
std::vector<std::unique_ptr<std::mutex>> swapchainMutexes;
std::counting_semaphore<std::numeric_limits<ptrdiff_t>::max()> recordingSemaphore(0);
std::shared_mutex uniformMutex;

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
			controls.keyW = false;
		else if (key == GLFW_KEY_S)
			controls.keyS = false;
		else if (key == GLFW_KEY_A)
			controls.keyA = false;
		else if (key == GLFW_KEY_D)
			controls.keyD = false;
	}
	else if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_W)
			controls.keyW = true;
		else if (key == GLFW_KEY_S)
			controls.keyS = true;
		else if (key == GLFW_KEY_A)
			controls.keyA = true;
		else if (key == GLFW_KEY_D)
			controls.keyD = true;
		else if (key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(handle, 1);
	}
}

void resizeEvent(GLFWwindow* handle, int width, int height) {
	static_cast<void>(width);
	static_cast<void>(height);

	glfwSetWindowSize(handle, details.swapchainExtent.width, details.swapchainExtent.width);
}

void initializeControls() {
	controls.keyW = false;
	controls.keyA = false;
	controls.keyS = false;
	controls.keyD = false;

	controls.deltaX = 0.0f;
	controls.deltaY = 0.0f;
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
		auto& flags = queueFamilies.at(index).queueFlags;
		auto computeSupport = flags & vk::QueueFlagBits::eCompute;
		auto graphicsSupport = flags & vk::QueueFlagBits::eGraphics;
		auto presentSupport = physicalDevice.getSurfaceSupportKHR(index, surface);

		if (computeSupport && graphicsSupport && presentSupport)
			return index;
	}

	return std::numeric_limits<uint32_t>::max();
}

//TODO: Check format availability and generate mipmaps
svh::Details generateDetails() {
	svh::Details temporaryDetails;

	temporaryDetails.meshCount = 0;
	temporaryDetails.portalCount = 0;
	temporaryDetails.commandBufferPerImage = 3;

	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	glfwGetFramebufferSize(window, reinterpret_cast<int32_t*>(&surfaceCapabilities.currentExtent.width),
		reinterpret_cast<int32_t*>(&surfaceCapabilities.currentExtent.height));

	surfaceCapabilities.currentExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
		std::min(surfaceCapabilities.maxImageExtent.width, surfaceCapabilities.currentExtent.width));
	surfaceCapabilities.currentExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
		std::min(surfaceCapabilities.maxImageExtent.height, surfaceCapabilities.currentExtent.height));

	uint32_t extraImages = 1;
	temporaryDetails.minImageCount = surfaceCapabilities.minImageCount;
	temporaryDetails.maxImageCount = surfaceCapabilities.maxImageCount ? surfaceCapabilities.maxImageCount : surfaceCapabilities.minImageCount + extraImages;
	temporaryDetails.imageCount = std::min(temporaryDetails.minImageCount + extraImages, temporaryDetails.maxImageCount);
	temporaryDetails.concurrentImageCount = temporaryDetails.imageCount - temporaryDetails.minImageCount + 1;
	temporaryDetails.queueCount = temporaryDetails.concurrentImageCount + 1;

	temporaryDetails.swapchainExtent = surfaceCapabilities.currentExtent;
	temporaryDetails.swapchainTransform = surfaceCapabilities.currentTransform;

	auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	temporaryDetails.surfaceFormat = surfaceFormats.front();

	for (auto& surfaceFormat : surfaceFormats)
		if (surfaceFormat.format == vk::Format::eR8G8B8A8Srgb && surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
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

	auto uniformAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;
	temporaryDetails.uniformAlignment = ((sizeof(glm::mat4) - 1) / uniformAlignment + 1) * uniformAlignment;

	return temporaryDetails;
}

void createDevice() {
	physicalDevice = pickPhysicalDevice();
	queueFamilyIndex = selectQueueFamily();
	details = generateDetails();

	vk::PhysicalDeviceFeatures deviceFeatures{};
	std::vector<const char*> extensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	std::vector<float> queuePriorities( details.queueCount, 1.0f );

	vk::DeviceQueueCreateInfo queueInfo{};
	queueInfo.queueFamilyIndex = queueFamilyIndex;
	queueInfo.queueCount = details.queueCount;
	queueInfo.pQueuePriorities = queuePriorities.data();

	vk::DeviceCreateInfo deviceInfo{};
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	deviceInfo.pEnabledFeatures = &deviceFeatures;
	deviceInfo.enabledExtensionCount = extensions.size();
	deviceInfo.ppEnabledExtensionNames = extensions.data();

	vk::CommandPoolCreateInfo poolInfo{};
	poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = queueFamilyIndex;

	device = physicalDevice.createDevice(deviceInfo);
	mainQueue = device.getQueue(queueFamilyIndex, 0);

	concurrentQueues.reserve(details.queueCount - 1);
	for (auto queueIndex = 1u; queueIndex < details.queueCount; queueIndex++)
		concurrentQueues.push_back(device.getQueue(queueFamilyIndex, queueIndex));

	mainCommandPool = device.createCommandPool(poolInfo);
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
	allocateInfo.commandPool = mainCommandPool;
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

	static_cast<void>(mainQueue.submit(1, &submitInfo, nullptr));
	static_cast<void>(mainQueue.waitIdle());
	device.freeCommandBuffers(mainCommandPool, 1, &commandBuffer);
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

uint32_t getTextureIndex(const tinygltf::Model& model, const tinygltf::Mesh& mesh) {
	auto& material = model.materials.at(mesh.primitives.front().material);

	for (auto& value : material.values)
		if (!value.first.compare("baseColorTexture"))
			return std::distance(imageNames.begin(), std::find(imageNames.begin(), imageNames.end(), model.images.at(value.second.TextureIndex()).name));

	return std::numeric_limits<uint32_t>::max();
}

glm::mat4 getNodeTranslation(const tinygltf::Node& node) {
	glm::mat4 translation{ 1.0f };

	for (auto index = 0u; index < node.translation.size(); index++)
		translation[3][index] = node.translation.at(index);

	return translation;
}

glm::mat4 getNodeRotation(const tinygltf::Node& node) {
	glm::mat4 rotation{ 1.0f };

	if (!node.rotation.empty())
		rotation = glm::toMat4(glm::qua{ node.rotation.at(3), node.rotation.at(0), node.rotation.at(1), node.rotation.at(2) });

	return rotation;
}

glm::mat4 getNodeScale(const tinygltf::Node& node) {
	glm::mat4 scale{ 1.0f };

	for (auto index = 0u; index < node.scale.size(); index++)
		scale[index][index] = node.scale.at(index);

	return scale;
}

glm::mat4 getNodeTransformation(const tinygltf::Node& node) {
	return getNodeTranslation(node) * getNodeRotation(node) * getNodeScale(node);
}

void createCameraFromMatrix(svh::Camera& camera, const glm::mat4& transformation, uint32_t room = 0) {
	camera.room = room;
	camera.position = transformation * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
	camera.direction = transformation * glm::vec4{ 0.0f, -1.0f, 0.0f, 0.0f };
	camera.up = transformation * glm::vec4{ 0.0f, 0.0f, 1.0f, 0.0f };
	camera.previous = camera.position;
}

//TODO: Use continuous memory for images
void loadTexture(std::string name, uint32_t levels, vk::Format format) {
	svh::Image image{};
	svh::Buffer buffer{};

	auto width = 0, height = 0, channel = 0;
	auto pixels = stbi_load(("assets/" + name + ".jpg").c_str(), &width, &height, &channel, STBI_rgb_alpha);
	auto size = width * height * 4;

	createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer.buffer, buffer.memory);

	auto memory = device.mapMemory(buffer.memory, 0, size);
	std::memcpy(memory, pixels, size);
	device.unmapMemory(buffer.memory);

	std::free(pixels);

	createImage(width, height, levels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal, image.image, image.memory);
	transitionImageLayout(image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, levels);
	copyBufferToImage(buffer.buffer, image.image, width, height);
	transitionImageLayout(image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, levels);
	image.view = createImageView(image.image, format, vk::ImageAspectFlagBits::eColor, levels);

	device.destroyBuffer(buffer.buffer);
	device.freeMemory(buffer.memory);

	textures.push_back(image);
}

void loadMesh(const tinygltf::Model& modelData, const tinygltf::Mesh& meshData, svh::Type type, uint32_t textureIndex,
	const glm::mat4& translation, const glm::mat4& rotation, const glm::mat4& scale, uint8_t sourceRoom = 0, uint8_t targetRoom = 0) {
	auto& primitive = meshData.primitives.front();
	auto& indexReference = modelData.bufferViews.at(primitive.indices);
	auto& indexData = modelData.buffers.at(indexReference.buffer);

	svh::Mesh mesh{};

	mesh.indexOffset = indices.size();
	mesh.indexLength = indexReference.byteLength / sizeof(uint16_t);
	mesh.sourceTransform = translation * rotation * scale;

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
	mesh.textureIndex = textureIndex;

	for (auto index = 0u; index < mesh.vertexLength; index++) {
		svh::Vertex vertex{};
		vertex.position = mesh.sourceTransform * glm::vec4{ positions.at(index), 1.0f };
		vertex.normal = glm::normalize(glm::vec3{ mesh.sourceTransform * glm::vec4{ glm::normalize(normals.at(index)), 0.0f } });
		vertex.texture = texcoords.at(index);
		vertices.push_back(vertex);
	}

	auto origin = glm::vec3{ 0.0f }, normal = glm::vec3{ 0.0f };
	auto min = glm::vec3{ std::numeric_limits<float_t>::max() }, max = glm::vec3{ -std::numeric_limits<float_t>::max() };

	for (auto index = 0u; index < mesh.vertexLength; index++) {
		auto& vertex = vertices.at(mesh.vertexOffset + index);

		origin += vertex.position;
		//normal += vertex.normal;

		min.x = std::min(min.x, vertex.position.x);
		min.y = std::min(min.y, vertex.position.y);
		min.z = std::min(min.z, vertex.position.z);

		max.x = std::max(max.x, vertex.position.x);
		max.y = std::max(max.y, vertex.position.y);
		max.z = std::max(max.z, vertex.position.z);
	}

	mesh.origin = origin / static_cast<float_t>(mesh.vertexLength);
	//mesh.normal = glm::normalize(normal);
	mesh.origin = glm::vec3{ translation * scale * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f } };
	mesh.minBorders = min;
	mesh.maxBorders = max;

	//if(type == svh::Type::Portal)
	//	for(int i = 0; i < 4; i++)
	//		std::cout << origin[i] << " " << mesh.origin[i] << std::endl;

	if (type == svh::Type::Mesh) {
		details.meshCount++;
		meshes.push_back(mesh);
	}

	else if (type == svh::Type::Portal) {
		svh::Portal portal{};

		portal.mesh = mesh;
		portal.direction = glm::normalize(rotation * glm::vec4{ 0.0f, 1.0f, 0.0f, 0.0f });

		details.portalCount++;
		portals.push_back(portal);
	}
}

void loadModel(const std::string name, svh::Type type, uint8_t sourceRoom = 0, uint8_t targetRoom = 0) {
	std::string error, warning;
	tinygltf::Model model;

	auto result = objectLoader.LoadASCIIFromFile(&model, &error, &warning, "assets/" + name + ".gltf");

#ifndef NDEBUG
	if (!warning.empty())
		std::cout << "GLTF Warning: " << warning << std::endl;
	if (!error.empty())
		std::cout << "GLTF Error: " << error << std::endl;
	if (!result)
		return;
#endif

	if (type == svh::Type::Camera)
		createCameraFromMatrix(camera, getNodeTransformation(model.nodes.front()), sourceRoom);

	else {
		for (auto& image : model.images) {
			if (std::find(imageNames.begin(), imageNames.end(), image.name) == imageNames.end()) {
				imageNames.push_back(image.name);
				loadTexture(image.name, details.mipLevels, details.imageFormat);
			}
		}

		for (auto& node : model.nodes) {
			auto& mesh = model.meshes.at(node.mesh);
			loadMesh(model, mesh, type, getTextureIndex(model, mesh), getNodeTranslation(node), getNodeRotation(node), getNodeScale(node));
		}

		if (type == svh::Type::Portal) {
			auto& bluePortal = portals.at(portals.size() - 2);
			auto& orangePortal = portals.at(portals.size() - 1);

			bluePortal.targetRoom = orangePortal.mesh.sourceRoom;
			orangePortal.targetRoom = bluePortal.mesh.sourceRoom;

			bluePortal.targetTransform = orangePortal.mesh.sourceTransform;
			orangePortal.targetTransform = bluePortal.mesh.sourceTransform;

			bluePortal.cameraTransform = bluePortal.targetTransform * glm::rotate(glm::radians(180.0f), glm::vec3{ 0.0f, 0.0f, 1.0f }) * glm::inverse(bluePortal.mesh.sourceTransform);
			orangePortal.cameraTransform = orangePortal.targetTransform * glm::rotate(glm::radians(180.0f), glm::vec3{ 0.0f, 0.0f, 1.0f }) * glm::inverse(orangePortal.mesh.sourceTransform);
		}
	}
}

void createScene() {
	loadModel("camera", svh::Type::Camera, 1);
	
	loadModel("portal12", svh::Type::Portal, 1, 2);
	loadModel("portal13", svh::Type::Portal, 1, 3);
	loadModel("portal14", svh::Type::Portal, 1, 4);
	loadModel("portal15", svh::Type::Portal, 1, 5);
	loadModel("portal26", svh::Type::Portal, 2, 6);

	loadModel("room1", svh::Type::Mesh, 1);
	loadModel("room2", svh::Type::Mesh, 2);
	loadModel("room3", svh::Type::Mesh, 3);
	loadModel("room4", svh::Type::Mesh, 4);
	loadModel("room5", svh::Type::Mesh, 5);
	loadModel("room6", svh::Type::Mesh, 6);
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
	details.concurrentImageCount = details.imageCount - details.minImageCount + 1;

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
	std::string extension{".glsl"};

	if (kind == shaderc_glsl_compute_shader)
		extension = ".comp";
	else if (kind == shaderc_glsl_vertex_shader)
		extension = ".vert";
	else if (kind == shaderc_glsl_fragment_shader)
		extension = ".frag";

	auto path = std::filesystem::current_path() / "shaders" / (name + extension);
	auto size = std::filesystem::file_size(path);

	std::ifstream file(path);
	std::vector<char> code(size);
	file.read(code.data(), size);

	auto shaderModule = shaderCompiler.CompileGlslToSpv(code.data(), code.size(), kind, name.c_str(), "main", shaderOptions);

#ifndef NDEBUG
	if (shaderModule.GetCompilationStatus() != shaderc_compilation_status_success) {
		if (!shaderModule.GetErrorMessage().empty())
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
	computeShader = loadShader("compute", shaderc_glsl_compute_shader);
	vertexShader = loadShader("vertex", shaderc_glsl_vertex_shader);
	fragmentShader = loadShader("fragment", shaderc_glsl_fragment_shader);

	vk::DescriptorSetLayoutBinding frustumPlanesBinding{};
	frustumPlanesBinding.binding = 0;
	frustumPlanesBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
	frustumPlanesBinding.descriptorCount = 1;
	frustumPlanesBinding.stageFlags = vk::ShaderStageFlagBits::eCompute;
	frustumPlanesBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding instanceDataBinding{};
	instanceDataBinding.binding = 1;
	instanceDataBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
	instanceDataBinding.descriptorCount = 1;
	instanceDataBinding.stageFlags = vk::ShaderStageFlagBits::eCompute;
	instanceDataBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding indirectDrawsBinding{};
	indirectDrawsBinding.binding = 2;
	indirectDrawsBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
	indirectDrawsBinding.descriptorCount = 1;
	indirectDrawsBinding.stageFlags = vk::ShaderStageFlagBits::eCompute;
	indirectDrawsBinding.pImmutableSamplers = nullptr;

	std::array<vk::DescriptorSetLayoutBinding, 3> computeBindings{ frustumPlanesBinding, instanceDataBinding, indirectDrawsBinding };

	vk::DescriptorSetLayoutCreateInfo computeDescriptorInfo{};
	computeDescriptorInfo.bindingCount = computeBindings.size();
	computeDescriptorInfo.pBindings = computeBindings.data();

	computeSetLayout = device.createDescriptorSetLayout(computeDescriptorInfo);

	vk::PipelineLayoutCreateInfo computeLayoutInfo{};
	computeLayoutInfo.setLayoutCount = 1;
	computeLayoutInfo.pSetLayouts = &computeSetLayout;
	computeLayoutInfo.pushConstantRangeCount = 0;
	computeLayoutInfo.pPushConstantRanges = nullptr;

	computePipelineLayout = device.createPipelineLayout(computeLayoutInfo);
	
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

	std::array<vk::DescriptorSetLayoutBinding, 2> graphicsBindings{ uniformLayoutBinding, samplerLayoutBinding };

	vk::DescriptorSetLayoutCreateInfo graphicsDescriptorInfo{};
	graphicsDescriptorInfo.bindingCount = graphicsBindings.size();
	graphicsDescriptorInfo.pBindings = graphicsBindings.data();

	graphicsSetLayout = device.createDescriptorSetLayout(graphicsDescriptorInfo);

	vk::PipelineLayoutCreateInfo graphicsLayoutInfo{};
	graphicsLayoutInfo.setLayoutCount = 1;
	graphicsLayoutInfo.pSetLayouts = &graphicsSetLayout;
	graphicsLayoutInfo.pushConstantRangeCount = 0;
	graphicsLayoutInfo.pPushConstantRanges = nullptr;

	graphicsPipelineLayout = device.createPipelineLayout(graphicsLayoutInfo);

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
}

void createPipelines() {
	vk::PipelineShaderStageCreateInfo computeShaderInfo{};
	computeShaderInfo.stage = vk::ShaderStageFlagBits::eCompute;
	computeShaderInfo.module = computeShader;
	computeShaderInfo.pName = "main";
	computeShaderInfo.pSpecializationInfo = nullptr;

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

	vk::ComputePipelineCreateInfo computePipelineInfo{};
	computePipelineInfo.basePipelineIndex = -1;
	computePipelineInfo.basePipelineHandle = nullptr;
	computePipelineInfo.layout = computePipelineLayout;
	computePipelineInfo.stage = computeShaderInfo;

	computePipeline = device.createComputePipeline(nullptr, computePipelineInfo).value;

	vk::GraphicsPipelineCreateInfo graphicsPipelineInfo{};
	graphicsPipelineInfo.stageCount = renderShaderStages.size();
	graphicsPipelineInfo.pStages = renderShaderStages.data();
	graphicsPipelineInfo.pVertexInputState = &inputInfo;
	graphicsPipelineInfo.pInputAssemblyState = &assemblyInfo;
	graphicsPipelineInfo.pRasterizationState = &rasterizerInfo;
	graphicsPipelineInfo.pMultisampleState = &multisamplingInfo;
	graphicsPipelineInfo.pDepthStencilState = &depthStencil;
	graphicsPipelineInfo.pColorBlendState = &blendInfo;
	graphicsPipelineInfo.pViewportState = &viewportInfo;
	graphicsPipelineInfo.pDynamicState = nullptr;
	graphicsPipelineInfo.layout = graphicsPipelineLayout;
	graphicsPipelineInfo.renderPass = renderPass;
	graphicsPipelineInfo.subpass = 0;
	graphicsPipelineInfo.basePipelineHandle = nullptr;
	graphicsPipelineInfo.basePipelineIndex = -1;

	graphicsPipeline = device.createGraphicsPipeline(nullptr, graphicsPipelineInfo).value;

	std::array<vk::DynamicState, 1> dynamicStates = { vk::DynamicState::eStencilReference };

	vk::PipelineDynamicStateCreateInfo dynamicInfo{};
	dynamicInfo.dynamicStateCount = dynamicStates.size();
	dynamicInfo.pDynamicStates = dynamicStates.data();

	graphicsPipelineInfo.pDynamicState = &dynamicInfo;

	depthStencil.stencilTestEnable = true;
	stencilOpState.passOp = vk::StencilOp::eReplace;
	stencilOpState.compareOp = vk::CompareOp::eAlways;

	stencilOpState.reference = 0;
	depthStencil.front = stencilOpState;
	stencilPipeline = device.createGraphicsPipeline(nullptr, graphicsPipelineInfo).value;

	stencilOpState.passOp = vk::StencilOp::eKeep;
	stencilOpState.compareOp = vk::CompareOp::eEqual;

	stencilOpState.reference = 0;
	depthStencil.front = stencilOpState;
	renderPipeline = device.createGraphicsPipeline(nullptr, graphicsPipelineInfo).value;
}

void createFramebuffers() {
	colorImages.reserve(details.imageCount);
	depthImages.reserve(details.imageCount);

	for (auto& swapchainView : swapchainViews) {
		svh::Image colorImage, depthImage;

		createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
			details.surfaceFormat.format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage.image, colorImage.memory);
		colorImage.view = createImageView(colorImage.image, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);

		createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
			details.depthStencilFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage.image, depthImage.memory);
		depthImage.view = createImageView(depthImage.image, details.depthStencilFormat, vk::ImageAspectFlagBits::eDepth, 1);
		
		std::array<vk::ImageView, 3> attachments{ colorImage.view, depthImage.view, swapchainView };

		vk::FramebufferCreateInfo framebufferInfo{};
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = details.swapchainExtent.width;
		framebufferInfo.height = details.swapchainExtent.height;
		framebufferInfo.layers = 1;

		colorImages.push_back(colorImage);
		depthImages.push_back(depthImage);
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

	details.uniformQueueStride = (details.portalCount + 1) * details.uniformAlignment;
	details.uniformFrameStride = details.commandBufferPerImage * details.uniformQueueStride;
	details.uniformSize = details.imageCount * details.uniformFrameStride;

	createBuffer(details.uniformSize, vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		uniformBuffer.buffer, uniformBuffer.memory);
}

void createDescriptorSets() {
	auto descriptorCount = textures.size();

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

	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts{ descriptorCount, graphicsSetLayout };

	vk::DescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = descriptorSetLayouts.size();
	allocateInfo.pSetLayouts = descriptorSetLayouts.data();

	auto descriptorSets = device.allocateDescriptorSets(allocateInfo);

	for (auto descriptorIndex = 0u; descriptorIndex < descriptorCount; descriptorIndex++) {
		auto& texture = textures.at(descriptorIndex);
		texture.descriptor = descriptorSets.at(descriptorIndex);

		vk::DescriptorBufferInfo uniformInfo{};
		uniformInfo.buffer = uniformBuffer.buffer;
		uniformInfo.offset = 0;
		uniformInfo.range = details.uniformAlignment;

		vk::DescriptorImageInfo samplerInfo{};
		samplerInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		samplerInfo.imageView = texture.view;
		samplerInfo.sampler = sampler;

		vk::WriteDescriptorSet uniformWrite{};
		uniformWrite.dstSet = texture.descriptor;
		uniformWrite.dstBinding = 0;
		uniformWrite.dstArrayElement = 0;
		uniformWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
		uniformWrite.descriptorCount = 1;
		uniformWrite.pBufferInfo = &uniformInfo;
		uniformWrite.pImageInfo = nullptr;
		uniformWrite.pTexelBufferView = nullptr;

		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.dstSet = texture.descriptor;
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

void createCommandBuffers() {
	auto commandBufferCount = details.imageCount * details.commandBufferPerImage;

	threadCommandPools.reserve(details.imageCount);
	commandBuffers.reserve(commandBufferCount);
	commandBufferStatuses.reserve(commandBufferCount);

	for (auto commandBufferIndex = 0u; commandBufferIndex < details.imageCount; commandBufferIndex++) {
		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		poolInfo.queueFamilyIndex = queueFamilyIndex;

		threadCommandPools.push_back(device.createCommandPool(poolInfo));

		vk::CommandBufferAllocateInfo allocateInfo{};
		allocateInfo.commandPool = threadCommandPools.back();
		allocateInfo.level = vk::CommandBufferLevel::ePrimary;
		allocateInfo.commandBufferCount = details.commandBufferPerImage;

		auto imageCommandBuffers = device.allocateCommandBuffers(allocateInfo);
		commandBuffers.insert(commandBuffers.end(), imageCommandBuffers.begin(), imageCommandBuffers.end());
	}

	for (auto statusIndex = 0u; statusIndex < commandBufferCount; statusIndex++)
		commandBufferStatuses.push_back(svh::Status::NotRecorded);
}

void createSyncObjects() {
	vk::SemaphoreCreateInfo semaphoreInfo{};

	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	submitFences.reserve(details.concurrentImageCount);
	submitSemaphores.reserve(details.concurrentImageCount);
	presentSemaphores.reserve(details.concurrentImageCount);

	swapchainMutexes.reserve(details.imageCount);
	swapchainSemaphores.reserve(details.imageCount);
	
	for (auto imageIndex = 0u; imageIndex < details.concurrentImageCount; imageIndex++) {
		submitFences.push_back(device.createFence(fenceInfo));
		submitSemaphores.push_back(device.createSemaphore(semaphoreInfo));
		presentSemaphores.push_back(device.createSemaphore(semaphoreInfo));
	}

	for (auto imageIndex = 0u; imageIndex < details.imageCount; imageIndex++) {
		swapchainMutexes.push_back(std::make_unique<std::mutex>());
		swapchainSemaphores.push_back(std::make_unique<std::binary_semaphore>(1));
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
	createSyncObjects();
}

//TODO: Remove constant memory map-unmaps
void updateUniformBuffer(uint32_t imageIndex, uint32_t queueIndex) {
	auto uniformOffset = imageIndex * details.uniformFrameStride + queueIndex * details.uniformQueueStride;

	std::shared_lock<std::shared_mutex> readLock{ uniformMutex };

	std::memcpy(deviceMemory + uniformOffset + details.portalCount * details.uniformAlignment, &camera.transform, sizeof(glm::mat4));

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++)
		std::memcpy(deviceMemory + uniformOffset + portalIndex * details.uniformAlignment, &portals.at(portalIndex).transform, sizeof(glm::mat4));
}

void updateCommandBuffer(uint32_t imageIndex, uint32_t queueIndex) {
	auto& commandBuffer = commandBuffers.at(imageIndex * details.commandBufferPerImage + queueIndex);
	auto uniformOffset = imageIndex * details.uniformFrameStride + queueIndex * details.uniformQueueStride;

	vk::DeviceSize bufferOffset = 0;

	vk::ClearValue colorClear{};
	colorClear.color.float32.at(0) = 0.0f;
	colorClear.color.float32.at(1) = 0.0f;
	colorClear.color.float32.at(2) = 0.0f;
	colorClear.color.float32.at(3) = 1.0f;

	vk::ClearValue depthStencilClear{};
	depthStencilClear.depthStencil.depth = 1.0f;
	depthStencilClear.depthStencil.stencil = 0;

	std::array<vk::ClearValue, 2> clearValues{ colorClear, depthStencilClear };

	vk::Offset2D areaOffset{};
	areaOffset.x = 0;
	areaOffset.y = 0;

	vk::Rect2D area{};
	area.offset = areaOffset;
	area.extent = details.swapchainExtent;

	vk::ClearRect clearArea{};
	clearArea.baseArrayLayer = 0;
	clearArea.layerCount = 1;
	clearArea.rect = area;

	vk::ClearAttachment depthClearAttachment{};
	depthClearAttachment.aspectMask = vk::ImageAspectFlagBits::eDepth;
	depthClearAttachment.clearValue = depthStencilClear;
	depthClearAttachment.colorAttachment = -1;

	vk::ClearAttachment stencilClearAttachment{};
	stencilClearAttachment.aspectMask = vk::ImageAspectFlagBits::eStencil;
	stencilClearAttachment.clearValue = depthStencilClear;
	stencilClearAttachment.colorAttachment = -1;

	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.pInheritanceInfo = nullptr;

	vk::RenderPassBeginInfo renderPassInfo{};
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.framebuffer = framebuffers.at(imageIndex);
	renderPassInfo.renderArea = area;
	renderPassInfo.clearValueCount = clearValues.size();
	renderPassInfo.pClearValues = clearValues.data();
	
	commandBuffer.begin(beginInfo);
	commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
	commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer.buffer, &bufferOffset);
	commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

	auto uniformLocation = uniformOffset + details.portalCount * details.uniformAlignment;

	for (auto textureIndex = 1u; textureIndex < textures.size(); textureIndex++) {
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 0, 1, &textures.at(textureIndex).descriptor, 1, &uniformLocation);
		
		for (auto& mesh : meshes)
			if(mesh.textureIndex == textureIndex)
				commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
	}

	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 0, 1, &textures.front().descriptor, 1, &uniformLocation);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, stencilPipeline);

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
		auto& portal = portals.at(portalIndex);

		commandBuffer.setStencilReference(vk::StencilFaceFlagBits::eFront, portalIndex + 1);
		commandBuffer.drawIndexed(portal.mesh.indexLength, 1, portal.mesh.indexOffset, portal.mesh.vertexOffset, 0);
	}

	commandBuffer.clearAttachments(1, &depthClearAttachment, 1, &clearArea);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, renderPipeline);

	for (auto portalIndex = 0u; portalIndex < details.portalCount; portalIndex++) {
		auto& portal = portals.at(portalIndex);

		auto portalUniformLocation = uniformOffset + portalIndex * details.uniformAlignment;

		commandBuffer.setStencilReference(vk::StencilFaceFlagBits::eFront, portalIndex + 1);

		for (auto textureIndex = 1u; textureIndex < textures.size(); textureIndex++) {
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 0, 1, &textures.at(textureIndex).descriptor, 1, &portalUniformLocation);
			
			for (auto& mesh : meshes)
				if (mesh.textureIndex == textureIndex)
					commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
		}
	}

	commandBuffer.clearAttachments(1, &stencilClearAttachment, 1, &clearArea);
	commandBuffer.endRenderPass();
	commandBuffer.end();
}

uint32_t getCommandBufferIndex(uint32_t imageIndex) {
	auto offset = imageIndex * details.commandBufferPerImage;
	auto &commandBufferMutex = swapchainMutexes.at(imageIndex);

	commandBufferMutex->lock();

	for (auto index = offset; index < offset + details.commandBufferPerImage; index++) {
		auto& status = commandBufferStatuses.at(index);

		if (status == svh::Status::Ready || status == svh::Status::Used) {
			status = svh::Status::InUse;
			commandBufferMutex->unlock();
			return index;
		}
	}

	commandBufferMutex->unlock();
	return std::numeric_limits<uint32_t>::max();
}

void recordCommands(uint32_t imageIndex) {
	auto commandBufferOffset = imageIndex * details.commandBufferPerImage;
	auto& swapchainMutex = swapchainMutexes.at(imageIndex);

	updateCommandBuffer(imageIndex, 0);

	commandBufferStatuses.at(commandBufferOffset) = svh::Status::Ready;
	auto previousIndex = std::numeric_limits<uint32_t>::max(), currentIndex = commandBufferOffset;

	if (++state.recordingCount == details.imageCount)
		recordingSemaphore.release(details.concurrentImageCount);

	while (state.threadsActive) {
		previousIndex = currentIndex;
		currentIndex = std::numeric_limits<uint32_t>::max();
		
		auto notRecorded = std::numeric_limits<uint32_t>::max();
		auto invalidated = std::numeric_limits<uint32_t>::max();
		auto used = std::numeric_limits<uint32_t>::max();

		swapchainMutex->lock();
		
		for (auto index = commandBufferOffset; index < commandBufferOffset + details.commandBufferPerImage; index++) {
			if (index == previousIndex)
				continue;

			auto& status = commandBufferStatuses.at(index);

			if (status == svh::Status::NotRecorded)
				notRecorded = index;
			if (status == svh::Status::Invalidated)
				invalidated = index;
			if (status == svh::Status::Used)
				used = index;
		}

		if (notRecorded != std::numeric_limits<uint32_t>::max())
			currentIndex = notRecorded;
		else if (invalidated != std::numeric_limits<uint32_t>::max())
			currentIndex = invalidated;
		else if (used != std::numeric_limits<uint32_t>::max())
			currentIndex = used;

		commandBufferStatuses.at(currentIndex) = svh::Status::Recording;
		swapchainMutex->unlock();

		auto queueIndex = currentIndex - commandBufferOffset;

		updateCommandBuffer(imageIndex, queueIndex);

		swapchainMutex->lock();

		for(auto index = commandBufferOffset; index < commandBufferOffset + details.commandBufferPerImage; index++)
			if(commandBufferStatuses.at(index) == svh::Status::Ready || commandBufferStatuses.at(index) == svh::Status::Used)
				commandBufferStatuses.at(index) = svh::Status::Invalidated;

		commandBufferStatuses.at(currentIndex) = svh::Status::Ready;

		swapchainMutex->unlock();
	}
}

void renderImage(uint32_t threadIndex) {
	auto& queue = concurrentQueues.at(threadIndex);
	auto& submitFence = submitFences.at(threadIndex);
	auto& submitSemaphore = submitSemaphores.at(threadIndex);
	auto& presentSemaphore = presentSemaphores.at(threadIndex);

	auto imageIndex = std::numeric_limits<uint32_t>::max();
	auto commandBufferIndex = std::numeric_limits<uint32_t>::max();
	vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	vk::SubmitInfo submitInfo{};
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &presentSemaphore;
	submitInfo.pWaitDstStageMask = &waitStage;
	submitInfo.commandBufferCount = 1;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &submitSemaphore;

	vk::PresentInfoKHR presentInfo{};
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &submitSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchain;

	recordingSemaphore.acquire();

	while (state.threadsActive) {
		device.waitForFences(1, &submitFence, true, std::numeric_limits<uint64_t>::max());
		device.resetFences(1, &submitFence);

		if (imageIndex != std::numeric_limits<uint32_t>::max()) {
			swapchainMutexes.at(imageIndex)->lock();
			commandBufferStatuses.at(commandBufferIndex) = svh::Status::Used;
			swapchainMutexes.at(imageIndex)->unlock();
			swapchainSemaphores.at(imageIndex)->release();
		}

		acquireMutex.lock();

		imageIndex = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(), presentSemaphore, nullptr);

		swapchainSemaphores.at(imageIndex)->acquire();

		commandBufferIndex = getCommandBufferIndex(imageIndex);
		auto queueIndex = commandBufferIndex - imageIndex * details.commandBufferPerImage;
		updateUniformBuffer(imageIndex, queueIndex);

		submitInfo.pCommandBuffers = &commandBuffers.at(commandBufferIndex);
		queue.submit(1, &submitInfo, submitFence);

		presentInfo.pImageIndices = &imageIndex;
		queue.presentKHR(presentInfo);

		acquireMutex.unlock();

		state.frameCount++;
	}
}

void gameTick() {
	state.previousTime = state.currentTime;
	state.currentTime = std::chrono::high_resolution_clock::now();
	state.timeDelta = std::chrono::duration<double_t, std::chrono::seconds::period>(state.currentTime - state.previousTime).count();
	state.checkPoint += state.timeDelta;

	auto moveDelta = static_cast<float_t>(state.timeDelta) * 12.0f, turnDelta = static_cast<float_t>(state.timeDelta) * glm::radians(240.0f);
	auto vectorCount = std::abs(controls.keyW - controls.keyS) + std::abs(controls.keyA - controls.keyD);

	if (vectorCount > 0)
		moveDelta /= std::sqrt(vectorCount);
	
	camera.previous = camera.position;

	auto left = glm::normalize(glm::cross(camera.up, camera.direction));

	camera.direction = glm::normalize(glm::vec3{ glm::rotate(turnDelta * controls.deltaY, left) *
														glm::rotate(turnDelta * controls.deltaX, camera.up) *
														glm::vec4{camera.direction, 0.0f} });

	left = glm::normalize(glm::cross(camera.up, camera.direction));

	camera.position += moveDelta * (controls.keyW - controls.keyS) * camera.direction +
		moveDelta * (controls.keyA - controls.keyD) * left;

	auto coefficient = 0.0f, distance = glm::length(camera.position - camera.previous);
	auto direction = glm::normalize(camera.position - camera.previous);

	for (auto& portal : portals) {
		if (svh::epsilon < distance && glm::intersectRayPlane(camera.previous, direction, portal.mesh.origin, portal.direction, coefficient)) {
			auto point = camera.previous + coefficient * direction;

			if (point.x >= portal.mesh.minBorders.x && point.y >= portal.mesh.minBorders.y && point.z >= portal.mesh.minBorders.z &&
				point.x <= portal.mesh.maxBorders.x && point.y <= portal.mesh.maxBorders.y && point.z <= portal.mesh.maxBorders.z &&
				0 <= coefficient && distance >= coefficient) {

				camera.position = portal.cameraTransform * glm::vec4{ camera.position, 1.0f };
				camera.direction = portal.cameraTransform * glm::vec4{ camera.direction, 0.0f };
				camera.up = portal.cameraTransform * glm::vec4{ camera.up, 0.0f };

				camera.previous = camera.position;

				break;
			}
		}
	}

	controls.deltaX = 0.0f;
	controls.deltaY = 0.0f;

	auto view = glm::lookAt(camera.position, camera.position + camera.direction, camera.up);
	auto projection = glm::perspective(glm::radians(45.0f),
		static_cast<float_t>(details.swapchainExtent.width) / static_cast<float_t>(details.swapchainExtent.height), 0.001f, 100.0f);
	projection[1][1] *= -1;

	camera.transform = projection * view;

	std::unique_lock<std::shared_mutex> writeLock{ uniformMutex };

	for (auto& portal : portals) {
		svh::Camera portalCamera = camera;
		portalCamera.position = portal.cameraTransform * glm::vec4{ portalCamera.position, 1.0f };

		auto portalView = glm::lookAt(portalCamera.position, portalCamera.position + portalCamera.direction, portalCamera.up);
		auto portalProjection = glm::perspective(glm::radians(45.0f),
			static_cast<float_t>(details.swapchainExtent.width) / static_cast<float_t>(details.swapchainExtent.height), 0.001f, 100.0f);
		/*
		auto plane = glm::vec4{ portal.normal, glm::dot(-portal.normal, portal.origin) };

		glm::vec4 quaternion{};
		quaternion.x = ((0.0f < plane.x) - (plane.x < 0.0f) + portalProjection[2][0]) / portalProjection[0][0];
		quaternion.y = ((0.0f < plane.y) - (plane.y < 0.0f) + portalProjection[2][1]) / portalProjection[1][1];
		quaternion.z = -1.0f;
		quaternion.w = portalProjection[2][2] / portalProjection[3][2];

		auto clip = plane * (1.0f / glm::dot(plane, quaternion));

		portalProjection[0][2] = clip.x;
		portalProjection[1][2] = clip.y;
		portalProjection[2][2] = clip.z;
		portalProjection[3][2] = clip.w;
		*/
		portalProjection[1][1] *= -1;

		portal.transform = portalProjection * portalView;
	}
}

//TODO: Split present and retrieve
void draw() {
	state.frameCount = 0;
	state.totalFrameCount = 0;
	state.currentImage = 0;
	state.checkPoint = 0.0f;
	state.recordingCount = 0;
	state.threadsActive = true;
	state.currentTime = std::chrono::high_resolution_clock::now();

	deviceMemory = static_cast<uint8_t*>(device.mapMemory(uniformBuffer.memory, 0, details.uniformSize));

	for (auto imageIndex = 0u; imageIndex < details.imageCount; imageIndex++)
		recordThreads.push_back(std::thread(recordCommands, imageIndex));

	for (auto imageIndex = 0u; imageIndex < details.concurrentImageCount; imageIndex++)
		renderThreads.push_back(std::thread(renderImage, imageIndex));
	
	while (true) {
		glfwPollEvents();
		if (glfwWindowShouldClose(window))
			break;

		gameTick();

		if (state.checkPoint > 1.0) {
			state.totalFrameCount += state.frameCount;
			auto title = std::to_string(state.frameCount) + " - " + std::to_string(state.totalFrameCount);
			state.frameCount = 0;
			state.checkPoint = 0.0;
			glfwSetWindowTitle(window, title.c_str());
		}

		//std::this_thread::sleep_for(std::chrono::microseconds(10));
		//std::this_thread::sleep_until(state.currentTime + std::chrono::milliseconds(1));
	}

	state.threadsActive = false;
	for (auto& thread : renderThreads)
		thread.join();
	for (auto& thread : recordThreads)
		thread.join();

	device.unmapMemory(uniformBuffer.memory);
	device.waitIdle();
}

void clear() {
	for (auto& semaphore : presentSemaphores)
		device.destroySemaphore(semaphore);
	for (auto& semaphore : submitSemaphores)
		device.destroySemaphore(semaphore);
	for (auto& fence : submitFences)
		device.destroyFence(fence);
	device.destroyDescriptorPool(descriptorPool);
	device.destroyBuffer(uniformBuffer.buffer);
	device.freeMemory(uniformBuffer.memory);
	device.destroyBuffer(vertexBuffer.buffer);
	device.freeMemory(vertexBuffer.memory);
	device.destroyBuffer(indexBuffer.buffer);
	device.freeMemory(indexBuffer.memory);
	for (auto& swapchainFramebuffer : framebuffers)
		device.destroyFramebuffer(swapchainFramebuffer);
	for (auto& colorImage : colorImages) {
		device.destroyImageView(colorImage.view);
		device.destroyImage(colorImage.image);
		device.freeMemory(colorImage.memory);
	}
	for (auto& depthImage : depthImages) {
		device.destroyImageView(depthImage.view);
		device.destroyImage(depthImage.image);
		device.freeMemory(depthImage.memory);
	}
	for (auto& texture : textures) {
		device.destroyImageView(texture.view);
		device.destroyImage(texture.image);
		device.freeMemory(texture.memory);
	}
	device.destroyPipeline(renderPipeline);
	device.destroyPipeline(stencilPipeline);
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipeline(computePipeline);
	device.destroyPipelineLayout(graphicsPipelineLayout);
	device.destroyPipelineLayout(computePipelineLayout);
	device.destroyDescriptorSetLayout(graphicsSetLayout);
	device.destroyDescriptorSetLayout(computeSetLayout);
	device.destroySampler(sampler);
	device.destroyShaderModule(fragmentShader);
	device.destroyShaderModule(vertexShader);
	device.destroyShaderModule(computeShader);
	device.destroyRenderPass(renderPass);
	for (auto& swapchainView : swapchainViews)
		device.destroyImageView(swapchainView);
	device.destroySwapchainKHR(swapchain);
	for(auto& commandPool : threadCommandPools)
		device.destroyCommandPool(commandPool);
	device.destroyCommandPool(mainCommandPool);
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
