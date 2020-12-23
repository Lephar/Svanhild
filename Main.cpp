#include "Silibrand.hpp"

GLFWwindow *window;
svh::Controls controls;
svh::State state;
svh::Camera observerCamera, playerCamera;
tinygltf::TinyGLTF objectLoader;

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
vk::RenderPass renderPass, offscreenPass;
vk::ShaderModule vertexShader, fragmentShader;
vk::Sampler sampler;
vk::DescriptorSetLayout descriptorSetLayout;
vk::PipelineLayout pipelineLayout;
vk::Pipeline graphicsPipeline, offscreenPipeline;
svh::Image colorImage, depthImage;
std::vector<svh::Image> offscreenImages;
std::vector<vk::Framebuffer> swapchainFramebuffers, offscreenFramebuffers;

std::vector<uint16_t> indices;
std::vector<svh::Vertex> vertices;
std::vector<glm::mat4> matrices;
std::vector<svh::Image> textures;
std::vector<svh::Mesh> meshes;
std::vector<svh::Model> models;
std::vector<svh::Asset> assets;
std::vector<svh::Portal> portals;

svh::Buffer indexBuffer, vertexBuffer;
std::vector<svh::Buffer> cameraBuffers, transformBuffers;

vk::DescriptorPool descriptorPool;
std::vector<vk::DescriptorSet> descriptorSets;
std::vector<vk::CommandBuffer> commandBuffers;

std::vector<vk::Fence> frameFences, orderFences;
std::vector<vk::Semaphore> availableSemaphores, finishedSemaphores;

#ifndef NDEBUG

vk::DebugUtilsMessengerEXT messenger;
vk::DispatchLoaderDynamic functionLoader;

VKAPI_ATTR VkBool32 VKAPI_CALL messageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
											   VkDebugUtilsMessageTypeFlagsEXT type,
											   const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
											   void *pUserData) {
	static_cast<void>(type);
	static_cast<void>(pUserData);

	if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		std::cout << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

#endif

void mouseCallback(GLFWwindow *handle, double x, double y) {
	controls.deltaX = controls.mouseX - x;
	controls.deltaY = y - controls.mouseY;
	controls.mouseX = x;
	controls.mouseY = y;
}

void keyboardCallback(GLFWwindow *handle, int key, int scancode, int action, int mods) {
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
	} else if (action == GLFW_PRESS) {
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

void resizeEvent(GLFWwindow *handle, int width, int height) {
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
	std::vector<const char *> layers{}, extensions{extensionNames, extensionNames + extensionCount};

#ifndef NDEBUG
	layers.push_back("VK_LAYER_KHRONOS_validation");
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	vk::ApplicationInfo applicationInfo{};
	applicationInfo.pApplicationName = "Silibrand";
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
	functionLoader = vk::DispatchLoaderDynamic{instance, vkGetInstanceProcAddr};
	messenger = instance.createDebugUtilsMessengerEXT(messengerInfo, nullptr, functionLoader);
#endif

	VkSurfaceKHR surfaceHandle;
	glfwCreateWindowSurface(instance, window, nullptr, &surfaceHandle);
	surface = vk::SurfaceKHR{surfaceHandle};
}

//TODO: Implement a better device selection
vk::PhysicalDevice pickPhysicalDevice() {
	auto physicalDevices = instance.enumeratePhysicalDevices();

	for (auto &temporaryDevice : physicalDevices) {
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
	std::vector<const char *> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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

svh::Details generateDetails() {
	svh::Details temporaryDetails;

	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	glfwGetFramebufferSize(window, reinterpret_cast<int *>(&surfaceCapabilities.currentExtent.width),
						   reinterpret_cast<int *>(&surfaceCapabilities.currentExtent.height));
	surfaceCapabilities.currentExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
													   std::min(surfaceCapabilities.maxImageExtent.width,
																surfaceCapabilities.currentExtent.width));
	surfaceCapabilities.currentExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
														std::min(surfaceCapabilities.maxImageExtent.height,
																 surfaceCapabilities.currentExtent.height));

	temporaryDetails.imageCount = std::min(surfaceCapabilities.minImageCount + 1, surfaceCapabilities.maxImageCount);
	temporaryDetails.swapchainExtent = surfaceCapabilities.currentExtent;
	temporaryDetails.swapchainTransform = surfaceCapabilities.currentTransform;

	temporaryDetails.depthStencilFormat = vk::Format::eD32Sfloat;
	auto formatProperties = physicalDevice.getFormatProperties(vk::Format::eD32SfloatS8Uint);

	if (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
		temporaryDetails.depthStencilFormat = vk::Format::eD32SfloatS8Uint;

	auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	temporaryDetails.surfaceFormat = surfaceFormats.front();

	for (auto &surfaceFormat : surfaceFormats)
		if (surfaceFormat.format == vk::Format::eB8G8R8A8Srgb &&
			surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			temporaryDetails.surfaceFormat = surfaceFormat;

	auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
	auto immediateSupport = false;
	temporaryDetails.presentMode = vk::PresentModeKHR::eFifo;

	for (auto &presentMode : presentModes) {
		if (presentMode == vk::PresentModeKHR::eMailbox) {
			temporaryDetails.presentMode = presentMode;
			break;
		} else if (presentMode == vk::PresentModeKHR::eImmediate)
			immediateSupport = true;
	}

	if (immediateSupport && temporaryDetails.presentMode != vk::PresentModeKHR::eMailbox)
		temporaryDetails.presentMode = vk::PresentModeKHR::eImmediate;

	temporaryDetails.imageFormat = vk::Format::eR8G8B8A8Srgb;

	auto deviceProperties = physicalDevice.getProperties();
	auto sampleCount = deviceProperties.limits.framebufferColorSampleCounts;
	if (sampleCount > deviceProperties.limits.framebufferDepthSampleCounts)
		sampleCount = deviceProperties.limits.framebufferDepthSampleCounts;

	if (sampleCount & vk::SampleCountFlagBits::e64)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e64;
	else if (sampleCount & vk::SampleCountFlagBits::e32)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e32;
	else if (sampleCount & vk::SampleCountFlagBits::e16)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e16;
	else if (sampleCount & vk::SampleCountFlagBits::e8)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e8;
	else if (sampleCount & vk::SampleCountFlagBits::e4)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e4;
	else if (sampleCount & vk::SampleCountFlagBits::e2)
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e2;
	else
		temporaryDetails.sampleCount = vk::SampleCountFlagBits::e1;

	temporaryDetails.mipLevels = 1;
	temporaryDetails.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;

	temporaryDetails.renderTransformCount = 0;
	temporaryDetails.offscreenTransformCount = 0;

	temporaryDetails.bufferAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;
	while (temporaryDetails.bufferAlignment < 2 * sizeof(glm::mat4))
		temporaryDetails.bufferAlignment += deviceProperties.limits.minUniformBufferOffsetAlignment;

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

	for (auto &swapchainImage : swapchainImages)
		swapchainViews.push_back(
				createImageView(swapchainImage, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
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

	std::array<vk::AttachmentDescription, 3> attachments{colorAttachment, depthAttachment, resolveAttachment};

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

	vk::SubpassDependency sourceExternal{};
	sourceExternal.srcSubpass = VK_SUBPASS_EXTERNAL;
	sourceExternal.dstSubpass = 0;
	sourceExternal.srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
	sourceExternal.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	sourceExternal.srcAccessMask = vk::AccessFlagBits::eMemoryRead;
	sourceExternal.dstAccessMask =
			vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
	sourceExternal.dependencyFlags = vk::DependencyFlagBits::eByRegion;

	vk::SubpassDependency destinationExternal{};
	destinationExternal.srcSubpass = 0;
	destinationExternal.dstSubpass = VK_SUBPASS_EXTERNAL;
	destinationExternal.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	destinationExternal.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
	destinationExternal.srcAccessMask =
			vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
	destinationExternal.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
	destinationExternal.dependencyFlags = vk::DependencyFlagBits::eByRegion;

	std::array<vk::SubpassDependency, 2> dependencies{sourceExternal, destinationExternal};

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.attachmentCount = attachments.size();
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = dependencies.size();
	renderPassInfo.pDependencies = dependencies.data();

	renderPass = device.createRenderPass(renderPassInfo, nullptr);
}

void createOffscreenPass() {
	vk::AttachmentDescription colorAttachment{};
	colorAttachment.format = details.imageFormat;
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
	resolveAttachment.format = details.imageFormat;
	resolveAttachment.samples = vk::SampleCountFlagBits::e1;
	resolveAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
	resolveAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	resolveAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	resolveAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	resolveAttachment.initialLayout = vk::ImageLayout::eUndefined;
	resolveAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	std::array<vk::AttachmentDescription, 3> attachments{colorAttachment, depthAttachment, resolveAttachment};

	vk::AttachmentReference colorReference{};
	colorReference.attachment = 0;
	colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentReference depthReference{};
	depthReference.attachment = 1;
	depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::AttachmentReference resolveReference{};
	resolveReference.attachment = 2;
	resolveReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpassDescription{};
	subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.pResolveAttachments = &resolveReference;

	vk::SubpassDependency sourceExternal{};
	sourceExternal.srcSubpass = VK_SUBPASS_EXTERNAL;
	sourceExternal.dstSubpass = 0;
	sourceExternal.srcStageMask = vk::PipelineStageFlagBits::eFragmentShader;
	sourceExternal.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	sourceExternal.srcAccessMask = vk::AccessFlagBits::eShaderRead;
	sourceExternal.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	sourceExternal.dependencyFlags = vk::DependencyFlagBits::eByRegion;

	vk::SubpassDependency destinationExternal{};
	destinationExternal.srcSubpass = 0;
	destinationExternal.dstSubpass = VK_SUBPASS_EXTERNAL;
	destinationExternal.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	destinationExternal.dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
	destinationExternal.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	destinationExternal.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	destinationExternal.dependencyFlags = vk::DependencyFlagBits::eByRegion;

	std::array<vk::SubpassDependency, 2> dependencies{sourceExternal, destinationExternal};

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.attachmentCount = attachments.size();
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = dependencies.size();
	renderPassInfo.pDependencies = dependencies.data();

	offscreenPass = device.createRenderPass(renderPassInfo, nullptr);
}

vk::ShaderModule loadShader(std::string path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	auto size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<uint32_t> data(size / sizeof(uint32_t));
	file.read(reinterpret_cast<char *>(data.data()), size);

	vk::ShaderModuleCreateInfo shaderInfo{};
	shaderInfo.flags = vk::ShaderModuleCreateFlags{};
	shaderInfo.codeSize = size;
	shaderInfo.pCode = data.data();

	return device.createShaderModule(shaderInfo);
}

void createPipelineLayout() {
	vertexShader = loadShader("shaders/vert.spv");
	fragmentShader = loadShader("shaders/frag.spv");

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

	vk::DescriptorSetLayoutBinding transformLayoutBinding{};
	transformLayoutBinding.binding = 0;
	transformLayoutBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
	transformLayoutBinding.descriptorCount = 1;
	transformLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
	transformLayoutBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding cameraLayoutBinding{};
	cameraLayoutBinding.binding = 1;
	cameraLayoutBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
	cameraLayoutBinding.descriptorCount = 1;
	cameraLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
	cameraLayoutBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 2;
	samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
	samplerLayoutBinding.pImmutableSamplers = nullptr;

	std::array<vk::DescriptorSetLayoutBinding, 3> bindings{
			transformLayoutBinding, cameraLayoutBinding, samplerLayoutBinding};

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

void createGraphicsPipeline() {
	vk::PipelineShaderStageCreateInfo vertexInfo{};
	vertexInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertexInfo.module = vertexShader;
	vertexInfo.pName = "main";
	vertexInfo.pSpecializationInfo = nullptr;

	vk::PipelineShaderStageCreateInfo fragmentInfo{};
	fragmentInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragmentInfo.module = fragmentShader;
	fragmentInfo.pName = "main";
	fragmentInfo.pSpecializationInfo = nullptr;

	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{vertexInfo, fragmentInfo};

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
			positionDescription, normalDescription, textureDescription};

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

	vk::PipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.depthTestEnable = true;
	depthStencil.depthWriteEnable = true;
	depthStencil.depthCompareOp = vk::CompareOp::eLess;
	depthStencil.depthBoundsTestEnable = false;
	depthStencil.minDepthBounds = 0.0f;
	depthStencil.maxDepthBounds = 1.0f;
	depthStencil.stencilTestEnable = false;
	depthStencil.front = vk::StencilOpState{};
	depthStencil.back = vk::StencilOpState{};

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

	std::array<float_t, 4> blendConstants{0.0f, 0.0f, 0.0f, 0.0f};

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
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
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
}

void createOffscreenPipeline() {
	vk::PipelineShaderStageCreateInfo vertexInfo{};
	vertexInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertexInfo.module = vertexShader;
	vertexInfo.pName = "main";
	vertexInfo.pSpecializationInfo = nullptr;

	vk::PipelineShaderStageCreateInfo fragmentInfo{};
	fragmentInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragmentInfo.module = fragmentShader;
	fragmentInfo.pName = "main";
	fragmentInfo.pSpecializationInfo = nullptr;

	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{vertexInfo, fragmentInfo};

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
			positionDescription, normalDescription, textureDescription};

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

	vk::PipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.depthTestEnable = true;
	depthStencil.depthWriteEnable = true;
	depthStencil.depthCompareOp = vk::CompareOp::eLess;
	depthStencil.depthBoundsTestEnable = false;
	depthStencil.minDepthBounds = 0.0f;
	depthStencil.maxDepthBounds = 1.0f;
	depthStencil.stencilTestEnable = false;
	depthStencil.front = vk::StencilOpState{};
	depthStencil.back = vk::StencilOpState{};

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

	std::array<float_t, 4> blendConstants{0.0f, 0.0f, 0.0f, 0.0f};

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

	std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

	vk::PipelineDynamicStateCreateInfo dynamicInfo{};
	dynamicInfo.dynamicStateCount = dynamicStates.size();
	dynamicInfo.pDynamicStates = dynamicStates.data();

	vk::GraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &inputInfo;
	pipelineInfo.pInputAssemblyState = &assemblyInfo;
	pipelineInfo.pRasterizationState = &rasterizerInfo;
	pipelineInfo.pMultisampleState = &multisamplingInfo;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &blendInfo;
	pipelineInfo.pViewportState = &viewportInfo;
	pipelineInfo.pDynamicState = &dynamicInfo;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = offscreenPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = nullptr;
	pipelineInfo.basePipelineIndex = -1;

	offscreenPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
}

uint32_t chooseMemoryType(uint32_t filter, vk::MemoryPropertyFlags flags) {
	auto memoryProperties = physicalDevice.getMemoryProperties();

	for (auto index = 0u; index < memoryProperties.memoryTypeCount; index++)
		if ((filter & (1 << index)) && (memoryProperties.memoryTypes[index].propertyFlags & flags) == flags)
			return index;

	return std::numeric_limits<uint32_t>::max();
}

void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
				  vk::Buffer &buffer, vk::DeviceMemory &bufferMemory) {
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
	device.bindBufferMemory(buffer, bufferMemory, 0);
}

void createImage(uint32_t imageWidth, uint32_t imageHeight, uint32_t mipLevels, vk::SampleCountFlagBits samples,
				 vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
				 vk::MemoryPropertyFlags properties, vk::Image &image, vk::DeviceMemory &imageMemory) {
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
	device.bindImageMemory(image, imageMemory, 0);
}

vk::CommandBuffer beginSingleTimeCommand() {
	vk::CommandBufferAllocateInfo allocateInfo{};
	allocateInfo.level = vk::CommandBufferLevel::ePrimary;
	allocateInfo.commandPool = commandPool;
	allocateInfo.commandBufferCount = 1;

	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto commandBuffer = device.allocateCommandBuffers(allocateInfo).at(0);
	commandBuffer.begin(beginInfo);
	return commandBuffer;
}

void endSingleTimeCommand(vk::CommandBuffer commandBuffer) {
	commandBuffer.end();

	vk::SubmitInfo submitInfo{};
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	static_cast<void>(queue.submit(1, &submitInfo, nullptr));
	queue.waitIdle();
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
	} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
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
void loadTexture(uint32_t width, uint32_t height, uint32_t levels, vk::Format format, std::vector<uint8_t> &data) {
	svh::Image image{};
	svh::Buffer buffer{};

	image.width = width;
	image.height = height;

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

	textures.push_back(image);
}

void loadGateway(uint32_t width, uint32_t height, uint32_t levels, vk::Format format, std::vector<uint8_t> &data) {
	svh::Image image{};
	image.width = width;
	image.height = height;

	createImage(width, height, levels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, image.image, image.memory);
	image.view = createImageView(image.image, format, vk::ImageAspectFlagBits::eColor, levels);

	textures.push_back(image);
}

void loadMesh(tinygltf::Model &modelData, tinygltf::Mesh &meshData) {
	for (auto &primitive : meshData.primitives) {
		auto &indexReference = modelData.bufferViews.at(primitive.indices);
		auto &indexData = modelData.buffers.at(indexReference.buffer);

		svh::Mesh mesh{};
		mesh.indexOffset = indices.size();
		mesh.vertexOffset = vertices.size();
		mesh.matrixIndex = matrices.size();
		mesh.textureOffset = textures.size() + primitive.material;
		mesh.indexLength = indexReference.byteLength / sizeof(uint16_t);

		indices.resize(mesh.indexOffset + mesh.indexLength);
		std::memcpy(indices.data() + mesh.indexOffset, indexData.data.data() + indexReference.byteOffset,
					indexReference.byteLength);

		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> texcoords;

		for (auto &attribute : primitive.attributes) {
			auto &accessor = modelData.accessors.at(attribute.second);
			auto &primitiveView = modelData.bufferViews.at(accessor.bufferView);
			auto &primitiveBuffer = modelData.buffers.at(primitiveView.buffer);

			if (attribute.first.compare("POSITION") == 0) {
				positions.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(positions.data(),
							primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			} else if (attribute.first.compare("NORMAL") == 0) {
				normals.resize(primitiveView.byteLength / sizeof(glm::vec3));
				std::memcpy(normals.data(),
							primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			} else if (attribute.first.compare("TEXCOORD_0") == 0) {
				texcoords.resize(primitiveView.byteLength / sizeof(glm::vec2));
				std::memcpy(texcoords.data(),
							primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
			}
		}

		mesh.vertexLength = texcoords.size();
		meshes.push_back(mesh);

		for (auto index = 0u; index < mesh.vertexLength; index++) {
			svh::Vertex vertex{};
			vertex.position = positions.at(index);
			vertex.normal = normals.at(index);
			vertex.texture = texcoords.at(index);
			vertices.push_back(vertex);
		}
	}
}

void loadNode(tinygltf::Model &modelData, tinygltf::Node &nodeData, glm::mat4 transform) {
	if (nodeData.mesh >= 0 && nodeData.mesh < modelData.meshes.size())
		loadMesh(modelData, modelData.meshes.at(nodeData.mesh));

	glm::mat4 scale{1.0f}, rotation{1.0f}, translation{1.0f};

	if (!nodeData.rotation.empty())
		rotation = glm::toMat4(glm::qua{
				nodeData.rotation.at(3), nodeData.rotation.at(0), nodeData.rotation.at(1), nodeData.rotation.at(2)});
	for (auto i = 0u; i < nodeData.scale.size(); i++)
		scale[i][i] = nodeData.scale.at(i);
	for (auto i = 0u; i < nodeData.translation.size(); i++)
		translation[3][i] = nodeData.translation.at(i);

	transform = transform * translation * rotation * scale;
	matrices.push_back(transform);

	for (auto &childIndex : nodeData.children)
		loadNode(modelData, modelData.nodes.at(childIndex), transform);
}

void loadModel(std::string filename, uint8_t portal = 0) {
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

	svh::Model model{};
	model.meshOffset = meshes.size();
	auto &scene = modelData.scenes.at(modelData.defaultScene);

	for (auto &nodeIndex : scene.nodes)
		loadNode(modelData, modelData.nodes.at(nodeIndex), glm::mat4{1.0f});

	model.meshCount = meshes.size() - model.meshOffset;
	models.push_back(model);

	for (auto i = 0u; i < modelData.materials.size(); i++) {
		auto &material = modelData.materials.at(i);

		for (auto &value : material.values) {
			if (!value.first.compare("baseColorTexture")) {
				auto &image = modelData.images.at(value.second.TextureIndex());

				if (!portal)
					loadTexture(image.width, image.height, details.mipLevels, details.imageFormat, image.image);
				else
					for (auto j = 0u; j < details.imageCount; j++)
						loadGateway(image.width, image.height, details.mipLevels, details.imageFormat, image.image);
			}
		}
	}
}

void addAsset(uint32_t model, glm::mat4 transform = glm::mat4{1.0f}) {
	svh::Asset asset{};
	asset.modelIndex = model;
	asset.matrixIndex = matrices.size();
	asset.transformOffset = details.renderTransformCount;
	assets.push_back(asset);

	details.renderTransformCount += models.at(model).meshCount;
	matrices.push_back(transform);
}

void addPortal(uint32_t blue, uint32_t orange) {
	svh::Portal portal{};
	portal.blueAssetIndex = blue;
	portal.orangeAssetIndex = orange;
	portals.push_back(portal);

	details.renderTransformCount -= 2;
	details.offscreenTransformCount += 2;
}

void createScene() {
	observerCamera.pos = glm::vec4{0.0f, 10.0f, 10.0f, 1.0f};
	observerCamera.dir = glm::vec4{0.0f, -std::sqrt(2.0f) / 2.0f, -std::sqrt(2.0f) / 2.0f, 0.0f};
	observerCamera.up = glm::vec4{0.0f, -std::sqrt(2.0f) / 2.0f, std::sqrt(2.0f) / 2.0f, 0.0f};

	playerCamera.pos = glm::vec4{0.0f, 0.0f, 1.0f, 1.0f};
	playerCamera.dir = glm::vec4{0.0f, -1.0f, 0.0f, 0.0f};
	playerCamera.up = glm::vec4{0.0f, 0.0f, 1.0f, 0.0f};

	loadModel("models/Room1.glb");
	loadModel("models/Room2.glb");
	loadModel("models/Portal1.glb", 1);
	loadModel("models/Portal2.glb", 1);

	addAsset(0);
	addAsset(1);
	addAsset(0, glm::translate(glm::vec3{0.0f, -30.0f, 0.0f}));
	addAsset(1, glm::translate(glm::vec3{0.0f, -30.0f, 0.0f}));
	addAsset(2);
	addAsset(3);

	addPortal(4, 5);
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

	for (auto &swapchainView : swapchainViews) {
		std::array<vk::ImageView, 3> attachments{colorImage.view, depthImage.view, swapchainView};

		vk::FramebufferCreateInfo framebufferInfo{};
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = details.swapchainExtent.width;
		framebufferInfo.height = details.swapchainExtent.height;
		framebufferInfo.layers = 1;

		swapchainFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
	}
}

void createOffscreenBuffers() {
	for (auto &portal : portals) {
		auto blueOffset =
				meshes.at(models.at(assets.at(portal.blueAssetIndex).modelIndex).meshOffset).textureOffset;
		auto orangeOffset =
				meshes.at(models.at(assets.at(portal.orangeAssetIndex).modelIndex).meshOffset).textureOffset;
		auto blueWidth = textures.at(blueOffset).width;
		auto blueHeight = textures.at(blueOffset).height;
		auto orangeWidth = textures.at(orangeOffset).width;
		auto orangeHeight = textures.at(orangeOffset).height;

		svh::Image blueColor, blueDepth, orangeColor, orangeDepth;

		createImage(blueWidth, blueHeight, 1, details.sampleCount, details.imageFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, blueColor.image, blueColor.memory);
		blueColor.view = createImageView(blueColor.image, details.imageFormat, vk::ImageAspectFlagBits::eColor, 1);

		createImage(blueWidth, blueHeight, 1, details.sampleCount, details.depthStencilFormat,
					vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
					blueDepth.image, blueDepth.memory);
		blueDepth.view = createImageView(blueDepth.image, details.depthStencilFormat, vk::ImageAspectFlagBits::eDepth,
										 1);

		createImage(orangeWidth, orangeHeight, 1, details.sampleCount, details.imageFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, orangeColor.image, orangeColor.memory);
		orangeColor.view = createImageView(orangeColor.image, details.imageFormat, vk::ImageAspectFlagBits::eColor, 1);

		createImage(orangeWidth, orangeHeight, 1, details.sampleCount, details.depthStencilFormat,
					vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
					orangeDepth.image, orangeDepth.memory);
		orangeDepth.view = createImageView(orangeDepth.image, details.depthStencilFormat,
										   vk::ImageAspectFlagBits::eDepth, 1);

		offscreenImages.push_back(blueColor);
		offscreenImages.push_back(blueDepth);
		offscreenImages.push_back(orangeColor);
		offscreenImages.push_back(orangeDepth);

		for (auto i = 0u; i < details.imageCount; i++) {
			std::array<vk::ImageView, 3> attachments{
					blueColor.view, blueDepth.view, textures.at(blueOffset + i).view};

			vk::FramebufferCreateInfo framebufferInfo{};
			framebufferInfo.renderPass = offscreenPass;
			framebufferInfo.attachmentCount = attachments.size();
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = blueWidth;
			framebufferInfo.height = blueHeight;
			framebufferInfo.layers = 1;

			offscreenFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
		}

		for (auto i = 0u; i < details.imageCount; i++) {
			std::array<vk::ImageView, 3> attachments{
					orangeColor.view, orangeDepth.view, textures.at(orangeOffset + i).view};

			vk::FramebufferCreateInfo framebufferInfo{};
			framebufferInfo.renderPass = offscreenPass;
			framebufferInfo.attachmentCount = attachments.size();
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = orangeWidth;
			framebufferInfo.height = orangeHeight;
			framebufferInfo.layers = 1;

			offscreenFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
		}
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

	auto transformSize = (details.renderTransformCount + details.offscreenTransformCount) * details.bufferAlignment;
	transformBuffers.resize(details.imageCount);

	for (auto &transformBuffer : transformBuffers)
		createBuffer(transformSize, vk::BufferUsageFlagBits::eUniformBuffer,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 transformBuffer.buffer, transformBuffer.memory);

	auto cameraSize = (2 * portals.size() + 1) * details.bufferAlignment;
	cameraBuffers.resize(details.imageCount);

	for (auto &cameraBuffer : cameraBuffers)
		createBuffer(cameraSize, vk::BufferUsageFlagBits::eUniformBuffer,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 cameraBuffer.buffer, cameraBuffer.memory);
}

//TODO: Combine model-view and projection matrices in the same buffer
void createDescriptorSets() {
	auto descriptorCount = details.imageCount * (details.renderTransformCount + details.offscreenTransformCount);

	vk::DescriptorPoolSize modelSize{};
	modelSize.type = vk::DescriptorType::eUniformBufferDynamic;
	modelSize.descriptorCount = descriptorCount;

	vk::DescriptorPoolSize cameraSize{};
	cameraSize.type = vk::DescriptorType::eUniformBufferDynamic;
	cameraSize.descriptorCount = descriptorCount;

	vk::DescriptorPoolSize samplerSize{};
	samplerSize.type = vk::DescriptorType::eCombinedImageSampler;
	samplerSize.descriptorCount = descriptorCount;

	std::array<vk::DescriptorPoolSize, 3> poolSizes{modelSize, cameraSize, samplerSize};

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.poolSizeCount = poolSizes.size();
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = descriptorCount;

	descriptorPool = device.createDescriptorPool(poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts{descriptorCount, descriptorSetLayout};

	vk::DescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = descriptorCount;
	allocateInfo.pSetLayouts = layouts.data();

	descriptorSets = device.allocateDescriptorSets(allocateInfo);

	for (auto i = 0u; i < details.imageCount; i++) {
		for (auto j = 0u; j < assets.size(); j++) {
			auto &asset = assets.at(j);
			auto &model = models.at(asset.modelIndex);
			auto imageIndex = j < assets.size() - 2 * portals.size() ? 0 : i;

			for (auto k = 0u; k < model.meshCount; k++) {
				auto &mesh = meshes.at(model.meshOffset + k);
				auto textureIndex = mesh.textureOffset + imageIndex;
				auto descriptorIndex = i * (details.renderTransformCount + details.offscreenTransformCount) +
									   asset.transformOffset + k;

				vk::DescriptorBufferInfo transformInfo{};
				transformInfo.buffer = transformBuffers.at(i).buffer;
				transformInfo.offset = 0;
				transformInfo.range = details.bufferAlignment;

				vk::DescriptorBufferInfo cameraInfo{};
				cameraInfo.buffer = cameraBuffers.at(i).buffer;
				cameraInfo.offset = 0;
				cameraInfo.range = details.bufferAlignment;

				vk::DescriptorImageInfo samplerInfo{};
				samplerInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
				samplerInfo.imageView = textures.at(textureIndex).view;
				samplerInfo.sampler = sampler;

				vk::WriteDescriptorSet modelWrite{};
				modelWrite.dstSet = descriptorSets.at(descriptorIndex);
				modelWrite.dstBinding = 0;
				modelWrite.dstArrayElement = 0;
				modelWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
				modelWrite.descriptorCount = 1;
				modelWrite.pBufferInfo = &transformInfo;
				modelWrite.pImageInfo = nullptr;
				modelWrite.pTexelBufferView = nullptr;

				vk::WriteDescriptorSet cameraWrite{};
				cameraWrite.dstSet = descriptorSets.at(descriptorIndex);
				cameraWrite.dstBinding = 1;
				cameraWrite.dstArrayElement = 0;
				cameraWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
				cameraWrite.descriptorCount = 1;
				cameraWrite.pBufferInfo = &cameraInfo;
				cameraWrite.pImageInfo = nullptr;
				cameraWrite.pTexelBufferView = nullptr;

				vk::WriteDescriptorSet samplerWrite{};
				samplerWrite.dstSet = descriptorSets.at(descriptorIndex);
				samplerWrite.dstBinding = 2;
				samplerWrite.dstArrayElement = 0;
				samplerWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
				samplerWrite.descriptorCount = 1;
				samplerWrite.pBufferInfo = nullptr;
				samplerWrite.pImageInfo = &samplerInfo;
				samplerWrite.pTexelBufferView = nullptr;

				std::array<vk::WriteDescriptorSet, 3> descriptorWrites{cameraWrite, modelWrite, samplerWrite};
				device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
				descriptorIndex++;
			}
		}
	}
}

//TODO: Implement an indexing function
void createCommandBuffers() {
	vk::CommandBufferAllocateInfo allocateInfo{};
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = vk::CommandBufferLevel::ePrimary;
	allocateInfo.commandBufferCount = details.imageCount;

	commandBuffers = device.allocateCommandBuffers(allocateInfo);

	for (auto i = 0u; i < details.imageCount; i++) {
		auto &commandBuffer = commandBuffers.at(i);

		vk::DeviceSize bufferOffset = 0;

		vk::Offset2D areaOffset{};
		areaOffset.x = 0;
		areaOffset.y = 0;

		vk::ClearValue colorClear{};
		colorClear.color.float32.at(0) = 0.0f;
		colorClear.color.float32.at(1) = 0.0f;
		colorClear.color.float32.at(2) = 0.0f;
		colorClear.color.float32.at(3) = 1.0f;

		vk::ClearValue depthClear{};
		depthClear.depthStencil.depth = 1.0f;
		depthClear.depthStencil.stencil = 0;

		std::array<vk::ClearValue, 2> clearValues{colorClear, depthClear};

		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.pInheritanceInfo = nullptr;

		commandBuffer.begin(beginInfo);
		commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
		commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer.buffer, &bufferOffset);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, offscreenPipeline);

		for (auto j = 0u; j < portals.size(); j++) {
			auto &portal = portals.at(j);
			auto &blueAsset = assets.at(portal.blueAssetIndex);
			auto &orangeAsset = assets.at(portal.orangeAssetIndex);
			auto &blueModel = models.at(blueAsset.modelIndex);
			auto &orangeModel = models.at(orangeAsset.modelIndex);
			auto &blueMesh = meshes.at(blueModel.meshOffset);
			auto &orangeMesh = meshes.at(orangeModel.meshOffset);
			auto &blueTexture = textures.at(blueMesh.textureOffset + i);
			auto &orangeTexture = textures.at(orangeMesh.textureOffset + i);
			auto &blueFramebuffer = offscreenFramebuffers.at(2 * j * details.imageCount + i);
			auto &orangeFramebuffer = offscreenFramebuffers.at(2 * j * details.imageCount + i + details.imageCount);

			vk::Extent2D blueExtent{};
			blueExtent.width = blueTexture.width;
			blueExtent.height = blueTexture.height;

			vk::Extent2D orangeExtent{};
			orangeExtent.width = orangeTexture.width;
			orangeExtent.height = orangeTexture.height;

			vk::Rect2D blueScissor{};
			blueScissor.offset = areaOffset;
			blueScissor.extent = blueExtent;

			vk::Rect2D orangeScissor{};
			orangeScissor.offset = areaOffset;
			orangeScissor.extent = orangeExtent;

			vk::Viewport blueViewport{};
			blueViewport.x = 0.0f;
			blueViewport.y = 0.0f;
			blueViewport.width = blueTexture.width;
			blueViewport.height = blueTexture.height;
			blueViewport.minDepth = 0.0f;
			blueViewport.maxDepth = 1.0f;

			vk::Viewport orangeViewport{};
			orangeViewport.x = 0.0f;
			orangeViewport.y = 0.0f;
			orangeViewport.width = orangeTexture.width;
			orangeViewport.height = orangeTexture.height;
			orangeViewport.minDepth = 0.0f;
			orangeViewport.maxDepth = 1.0f;

			vk::RenderPassBeginInfo bluePassInfo{};
			bluePassInfo.renderPass = offscreenPass;
			bluePassInfo.framebuffer = blueFramebuffer;
			bluePassInfo.renderArea.offset = areaOffset;
			bluePassInfo.renderArea.extent = blueExtent;
			bluePassInfo.clearValueCount = clearValues.size();
			bluePassInfo.pClearValues = clearValues.data();

			vk::RenderPassBeginInfo orangePassInfo{};
			orangePassInfo.renderPass = offscreenPass;
			orangePassInfo.framebuffer = orangeFramebuffer;
			orangePassInfo.renderArea.offset = areaOffset;
			orangePassInfo.renderArea.extent = orangeExtent;
			orangePassInfo.clearValueCount = clearValues.size();
			orangePassInfo.pClearValues = clearValues.data();

			commandBuffer.setScissor(0, 1, &blueScissor);
			commandBuffer.setViewport(0, 1, &blueViewport);
			commandBuffer.beginRenderPass(bluePassInfo, vk::SubpassContents::eInline);

			for (auto k = 0u; k < assets.size() - 2 * portals.size(); k++) {
				auto &asset = assets.at(k);
				auto &model = models.at(asset.modelIndex);

				for (auto l = 0u; l < model.meshCount; l++) {
					auto &mesh = meshes.at(model.meshOffset + l);
					auto descriptorIndex = i * (details.renderTransformCount + details.offscreenTransformCount) +
										   asset.transformOffset + l;
					auto transformOffset = (asset.transformOffset + l) * details.bufferAlignment;
					auto cameraOffset = 2 * j * details.bufferAlignment;

					std::array<uint32_t, 2> descriptorOffsets{transformOffset, cameraOffset};

					commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
													 &descriptorSets.at(descriptorIndex), descriptorOffsets.size(),
													 descriptorOffsets.data());
					commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
				}
			}

			commandBuffer.endRenderPass();

			commandBuffer.setScissor(0, 1, &orangeScissor);
			commandBuffer.setViewport(0, 1, &orangeViewport);
			commandBuffer.beginRenderPass(orangePassInfo, vk::SubpassContents::eInline);

			for (auto k = 0u; k < assets.size() - 2 * portals.size(); k++) {
				auto &asset = assets.at(k);
				auto &model = models.at(asset.modelIndex);

				for (auto l = 0u; l < model.meshCount; l++) {
					auto &mesh = meshes.at(model.meshOffset + l);
					auto descriptorIndex = i * (details.renderTransformCount + details.offscreenTransformCount) +
										   asset.transformOffset + l;
					auto transformOffset = (asset.transformOffset + l) * details.bufferAlignment;
					auto cameraOffset = (2 * j + 1) * details.bufferAlignment;

					std::array<uint32_t, 2> descriptorOffsets{transformOffset, cameraOffset};

					commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
													 &descriptorSets.at(descriptorIndex), descriptorOffsets.size(),
													 descriptorOffsets.data());
					commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
				}
			}

			commandBuffer.endRenderPass();
		}

		vk::RenderPassBeginInfo renderPassInfo{};
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapchainFramebuffers.at(i);
		renderPassInfo.renderArea.offset = areaOffset;
		renderPassInfo.renderArea.extent = details.swapchainExtent;
		renderPassInfo.clearValueCount = clearValues.size();
		renderPassInfo.pClearValues = clearValues.data();

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
		commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

		for (auto &asset : assets) {
			auto &model = models.at(asset.modelIndex);

			for (auto j = 0u; j < model.meshCount; j++) {
				auto &mesh = meshes.at(model.meshOffset + j);
				auto descriptorIndex = i * (details.renderTransformCount + details.offscreenTransformCount) +
									   asset.transformOffset + j;
				auto transformOffset = (asset.transformOffset + j) * details.bufferAlignment;
				auto cameraOffset = 2 * static_cast<uint32_t>(portals.size()) * details.bufferAlignment;

				std::array<uint32_t, 2> descriptorOffsets{transformOffset, cameraOffset};

				commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
												 &descriptorSets.at(descriptorIndex), descriptorOffsets.size(),
												 descriptorOffsets.data());
				commandBuffer.drawIndexed(mesh.indexLength, 1, mesh.indexOffset, mesh.vertexOffset, 0);
			}
		}

		commandBuffer.endRenderPass();
		commandBuffer.end();
	}
}

void createSyncObject() {
	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	vk::SemaphoreCreateInfo semaphoreInfo{};

	for (auto i = 0u; i < details.imageCount; i++) {
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
	createRenderPass();
	createOffscreenPass();
	createPipelineLayout();
	createGraphicsPipeline();
	createOffscreenPipeline();
	createScene();
	createFramebuffers();
	createOffscreenBuffers();
	createElementBuffers();
	createDescriptorSets();
	createCommandBuffers();
	createSyncObject();
}

//TODO: Remove constant memory map-unmaps
void updateScene(uint32_t imageIndex) {
	auto moveDelta = state.timeDelta * 10.0f, turnDelta = state.timeDelta * glm::radians(60.0f);
	auto vectorCount = std::abs(controls.keyW - controls.keyS) + std::abs(controls.keyA - controls.keyD);

	if (vectorCount > 0)
		moveDelta /= std::sqrt(vectorCount);

	//TODO: Refactor here like the player camera
	if (controls.observer) {
		auto left = glm::normalize(glm::cross(observerCamera.up, observerCamera.dir));

		if (controls.keyW || controls.keyS)
			observerCamera.pos += (controls.keyW - controls.keyS) * moveDelta *
								  (observerCamera.dir + observerCamera.up) / std::sqrt(2.0f);
		if (controls.keyA || controls.keyD)
			observerCamera.pos += (controls.keyA - controls.keyD) * moveDelta * left;
		if (controls.keyR || controls.keyF)
			observerCamera.pos += (controls.keyR - controls.keyF) * moveDelta * observerCamera.dir;
		if (controls.keyQ || controls.keyE) {
			glm::mat4 rotation = glm::rotate((controls.keyQ - controls.keyE) * turnDelta, glm::vec3{0.0f, 0.0f, 1.0f});

			auto direction = glm::normalize(glm::vec2{observerCamera.dir});
			observerCamera.pos += observerCamera.pos.z * glm::vec3{direction, 0.0f};
			observerCamera.dir = rotation * glm::vec4{observerCamera.dir, 0.0f};

			direction = glm::normalize(glm::vec2{observerCamera.dir});
			observerCamera.pos -= observerCamera.pos.z * glm::vec3{direction, 0.0f};
			observerCamera.up = rotation * glm::vec4{observerCamera.up, 0.0f};
		}
	} else {
		auto left = glm::normalize(glm::cross(playerCamera.up, playerCamera.dir));

		playerCamera.dir = glm::normalize(glm::vec3{glm::rotate(turnDelta * controls.deltaY, left) *
													glm::rotate(turnDelta * controls.deltaX, playerCamera.up) *
													glm::vec4{playerCamera.dir, 0.0f}});

		left = glm::normalize(glm::cross(playerCamera.up, playerCamera.dir));

		playerCamera.pos += moveDelta * (controls.keyW - controls.keyS) * playerCamera.dir +
							moveDelta * (controls.keyA - controls.keyD) * left;

		controls.deltaX = 0.0f;
		controls.deltaY = 0.0f;
	}

	auto &camera = controls.observer ? observerCamera : playerCamera;
	auto view = glm::lookAt(camera.pos, camera.pos + camera.dir, camera.up);
	auto proj = glm::perspective(glm::radians(45.0f),
								 static_cast<float_t>(details.swapchainExtent.width) /
								 static_cast<float_t>(details.swapchainExtent.height), 0.01f, 100.0f);
	proj[1][1] *= -1;

	auto memory = static_cast<uint8_t *>(device.mapMemory(transformBuffers.at(imageIndex).memory, 0,
														  (details.renderTransformCount +
														   details.offscreenTransformCount) *
														  details.bufferAlignment));

	for (auto i = 0u; i < assets.size(); i++) {
		auto &asset = assets.at(i);
		auto &model = models.at(asset.modelIndex);

		for (auto j = 0u; j < model.meshCount; j++) {
			auto &mesh = meshes.at(model.meshOffset + j);
			auto transform = matrices.at(asset.matrixIndex) * matrices.at(mesh.matrixIndex);
			auto offset = (asset.transformOffset + j) * details.bufferAlignment;

			std::memcpy(memory + offset, &transform, sizeof(glm::mat4));
		}
	}

	device.unmapMemory(transformBuffers.at(imageIndex).memory);
	memory = static_cast<uint8_t *>(device.mapMemory(cameraBuffers.at(imageIndex).memory, 0,
													 (2 * portals.size() + 1) * details.bufferAlignment));

	for (auto i = 0u; i < portals.size(); i++) {
		auto &portal = portals.at(i);
		auto &blueAsset = assets.at(portal.blueAssetIndex);
		auto &orangeAsset = assets.at(portal.orangeAssetIndex);
		auto &blueModel = models.at(blueAsset.modelIndex);
		auto &orangeModel = models.at(orangeAsset.modelIndex);
		auto &blueMesh = meshes.at(blueModel.meshOffset);
		auto &orangeMesh = meshes.at(orangeModel.meshOffset);
		auto &blueTexture = textures.at(blueMesh.textureOffset);
		auto &orangeTexture = textures.at(orangeMesh.textureOffset);

		auto blueMatrix = matrices.at(blueAsset.matrixIndex) * matrices.at(blueMesh.matrixIndex);
		auto orangeMatrix = matrices.at(orangeAsset.matrixIndex) * matrices.at(orangeMesh.matrixIndex);

		auto portalRotate = glm::rotate(glm::radians(180.0f), glm::vec3{0.0f, 0.0f, -1.0f});
		auto orangeToBlue = orangeMatrix * portalRotate * glm::inverse(blueMatrix);
		auto blueToOrange = blueMatrix * portalRotate * glm::inverse(orangeMatrix);

		svh::Camera blueCamera{}, orangeCamera{};

		/*blueCamera.dir = blueMatrix * portalRotate * glm::vec4{camera.dir, 0.0f};
		blueCamera.pos = blueMatrix * glm::vec4{0.0f, 0.0f, 0.0f, 1.0f};
		blueCamera.up = glm::vec3{0.0f, 0.0f,1.0f};

		orangeCamera.dir = orangeMatrix * portalRotate * glm::vec4{camera.dir, 0.0f};
		orangeCamera.pos = orangeMatrix * glm::vec4{0.0f, 0.0f, 0.0f, 1.0f};
		orangeCamera.up = glm::vec3{0.0f, 0.0f,1.0f};*/

		blueCamera.pos = blueToOrange * glm::vec4{camera.pos, 1.0f};
		blueCamera.dir = blueToOrange * glm::vec4{camera.dir, 0.0f};
		blueCamera.up = blueToOrange * glm::vec4{-camera.up, 0.0f};

		orangeCamera.pos = orangeToBlue * glm::vec4{camera.pos, 1.0f};
		orangeCamera.dir = orangeToBlue * glm::vec4{camera.dir, 0.0f};
		orangeCamera.up = orangeToBlue * glm::vec4{-camera.up, 0.0f};

		//blueCamera.dir = glm::normalize(blueCamera.dir + glm::normalize(blueCamera.pos - camera.pos));
		//orangeCamera.dir = glm::normalize(orangeCamera.dir + glm::normalize(orangeCamera.pos - camera.pos));

		auto blueView = glm::lookAt(blueCamera.pos, blueCamera.pos + blueCamera.dir, blueCamera.up);
		auto orangeView = glm::lookAt(orangeCamera.pos, orangeCamera.pos + orangeCamera.dir, orangeCamera.up);

		//auto blueView = view * blueMatrix * portalRotate * glm::inverse(orangeMatrix);
		//auto orangeView = view * orangeMatrix * portalRotate * glm::inverse(blueMatrix);

		auto blueProj = glm::perspective(glm::radians(45.0f),
										 static_cast<float_t>(blueTexture.width) /
										 static_cast<float_t>(blueTexture.height), 0.01f, 100.0f);
		auto orangeProj = glm::perspective(glm::radians(45.0f),
										   static_cast<float_t>(orangeTexture.width) /
										   static_cast<float_t>(orangeTexture.height), 0.01f, 100.0f);

		blueProj[1][1] *= -1;
		orangeProj[1][1] *= -1;

		auto offset = 2 * i * details.bufferAlignment;

		std::memcpy(memory + offset, &blueView, sizeof(glm::mat4));
		std::memcpy(memory + offset + sizeof(glm::mat4), &blueProj, sizeof(glm::mat4));
		std::memcpy(memory + offset + details.bufferAlignment, &orangeView, sizeof(glm::mat4));
		std::memcpy(memory + offset + details.bufferAlignment + sizeof(glm::mat4), &orangeProj, sizeof(glm::mat4));
	}

	std::memcpy(memory + 2 * portals.size() * details.bufferAlignment, &view, sizeof(glm::mat4));
	std::memcpy(memory + 2 * portals.size() * details.bufferAlignment + sizeof(glm::mat4), &proj, sizeof(glm::mat4));

	device.unmapMemory(cameraBuffers.at(imageIndex).memory);
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

		std::array<vk::Semaphore, 1> waitSemaphores{availableSemaphores[state.currentImage]};
		std::array<vk::Semaphore, 1> signalSemaphores{finishedSemaphores[state.currentImage]};
		std::array<vk::PipelineStageFlags, 1> waitStages{vk::PipelineStageFlagBits::eColorAttachmentOutput};

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

	device.waitIdle();
}

void clear() {
	for (auto &finishedSemaphore : finishedSemaphores)
		device.destroySemaphore(finishedSemaphore);
	for (auto &availableSemaphore : availableSemaphores)
		device.destroySemaphore(availableSemaphore);
	for (auto &frameFence : frameFences)
		device.destroyFence(frameFence);
	device.destroyDescriptorPool(descriptorPool);
	for (auto &modelBuffer : transformBuffers) {
		device.destroyBuffer(modelBuffer.buffer);
		device.freeMemory(modelBuffer.memory);
	}
	for (auto &cameraBuffer : cameraBuffers) {
		device.destroyBuffer(cameraBuffer.buffer);
		device.freeMemory(cameraBuffer.memory);
	}
	device.destroyBuffer(vertexBuffer.buffer);
	device.freeMemory(vertexBuffer.memory);
	device.destroyBuffer(indexBuffer.buffer);
	device.freeMemory(indexBuffer.memory);
	for (auto &offscreenFramebuffer : offscreenFramebuffers)
		device.destroyFramebuffer(offscreenFramebuffer);
	for (auto &offscreenImage : offscreenImages) {
		device.destroyImageView(offscreenImage.view);
		device.destroyImage(offscreenImage.image);
		device.freeMemory(offscreenImage.memory);
	}
	for (auto &swapchainFramebuffer : swapchainFramebuffers)
		device.destroyFramebuffer(swapchainFramebuffer);
	device.destroyImageView(colorImage.view);
	device.destroyImage(colorImage.image);
	device.freeMemory(colorImage.memory);
	device.destroyImageView(depthImage.view);
	device.destroyImage(depthImage.image);
	device.freeMemory(depthImage.memory);
	for (auto &texture : textures) {
		device.destroyImageView(texture.view);
		device.destroyImage(texture.image);
		device.freeMemory(texture.memory);
	}
	device.destroyPipeline(offscreenPipeline);
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroySampler(sampler);
	device.destroyShaderModule(fragmentShader);
	device.destroyShaderModule(vertexShader);
	device.destroyRenderPass(offscreenPass);
	device.destroyRenderPass(renderPass);
	for (auto &swapchainView : swapchainViews)
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
