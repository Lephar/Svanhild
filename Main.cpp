#include "Silibrand.hpp"

GLFWwindow *window;
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
vk::ShaderModule vertexShader, fragmentShader;
vk::DescriptorSetLayout descriptorSetLayout;
vk::PipelineLayout pipelineLayout;
vk::Pipeline graphicsPipeline;

#ifndef NDEBUG

vk::DispatchLoaderDynamic loader;
vk::DebugUtilsMessengerEXT messenger;

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

void initializeCore() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(1280, 720, "", nullptr, nullptr);
	//glfwGetCursorPos(window, &mouseX, &mouseY);
	//glfwSetKeyCallback(window, keyboardCallback);
	//glfwSetCursorPosCallback(window, mouseCallback);

	uint32_t extensionCount = 0;
	auto **extensionNames = glfwGetRequiredInstanceExtensions(&extensionCount);
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
	loader = vk::DispatchLoaderDynamic{instance, vkGetInstanceProcAddr};
	messenger = instance.createDebugUtilsMessengerEXT(messengerInfo, nullptr, loader);
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

	for (uint32_t index = 0; index < queueFamilies.size(); index++) {
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

	temporaryDetails.swapchainExtent = surfaceCapabilities.currentExtent;
	temporaryDetails.swapchainTransform = surfaceCapabilities.currentTransform;
	temporaryDetails.imageCount = std::min(surfaceCapabilities.minImageCount + 1, surfaceCapabilities.maxImageCount);

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

	return temporaryDetails;
}

VkImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags flags) {
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.viewType = vk::ImageViewType::e2D;
	viewInfo.image = image;
	viewInfo.format = format;
	viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
	viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
	viewInfo.subresourceRange.aspectMask = flags;
	viewInfo.subresourceRange.levelCount = 1;
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
				createImageView(swapchainImage, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor));
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

void createPipeline() {
	vertexShader = loadShader("shaders/vert.spv");
	fragmentShader = loadShader("shaders/frag.spv");

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
	positionDescription.offset = offsetof(svh::Vertex, pos);

	vk::VertexInputAttributeDescription normalDescription{};
	normalDescription.binding = 0;
	normalDescription.location = 1;
	normalDescription.format = vk::Format::eR32G32B32Sfloat;
	normalDescription.offset = offsetof(svh::Vertex, nor);

	vk::VertexInputAttributeDescription textureDescription{};
	textureDescription.binding = 0;
	textureDescription.location = 2;
	textureDescription.format = vk::Format::eR32G32Sfloat;
	textureDescription.offset = offsetof(svh::Vertex, tex);

	std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{positionDescription, normalDescription, textureDescription};

	vk::PipelineVertexInputStateCreateInfo inputInfo{};
	inputInfo.vertexBindingDescriptionCount = 1;
	inputInfo.pVertexBindingDescriptions = &bindingDescription;
	inputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
	inputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	vk::PipelineInputAssemblyStateCreateInfo assemblyInfo{};
	assemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;
	assemblyInfo.primitiveRestartEnable = false;

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

	std::array<float, 4> blendConstants{0.0f, 0.0f, 0.0f, 0.0f};

	vk::PipelineColorBlendStateCreateInfo blendInfo{};
	blendInfo.logicOpEnable = VK_FALSE;
	blendInfo.logicOp = vk::LogicOp::eCopy;
	blendInfo.attachmentCount = 1;
	blendInfo.pAttachments = &blendAttachment;
	blendInfo.blendConstants = blendConstants;

	vk::DescriptorSetLayoutBinding cameraLayoutBinding{};
	cameraLayoutBinding.binding = 0;
	cameraLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
	cameraLayoutBinding.descriptorCount = 1;
	cameraLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
	cameraLayoutBinding.pImmutableSamplers = nullptr;

	vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
	samplerLayoutBinding.pImmutableSamplers = nullptr;

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings{cameraLayoutBinding, samplerLayoutBinding};

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

void setup() {
	initializeCore();
	createDevice();
	createSwapchain();
	createRenderPass();
	createPipeline();
}

void draw() {
}

void clear() {
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroyShaderModule(fragmentShader);
	device.destroyShaderModule(vertexShader);
	device.destroyRenderPass(renderPass);
	for (auto &swapchainView : swapchainViews)
		device.destroyImageView(swapchainView);
	device.destroySwapchainKHR(swapchain);
	device.destroyCommandPool(commandPool);
	device.destroy();
	instance.destroySurfaceKHR(surface);
#ifndef NDEBUG
	instance.destroyDebugUtilsMessengerEXT(messenger, nullptr, loader);
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
