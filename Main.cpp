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
svh::Image depthImage, colorImage;
std::vector<vk::Framebuffer> framebuffers;
vk::Sampler sampler;


std::vector<uint16_t> indices;
std::vector<svh::Vertex> vertices;
std::vector<svh::Image> images;
std::vector<svh::Mesh> meshes;
std::vector<svh::Model> models;

#ifndef NDEBUG

vk::DispatchLoaderDynamic functionLoader;
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
	temporaryDetails.sampleCount = vk::SampleCountFlagBits::e2;

	temporaryDetails.mipLevels = 1;

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

uint32_t chooseMemoryType(uint32_t filter, vk::MemoryPropertyFlags flags) {
	auto memoryProperties = physicalDevice.getMemoryProperties();

	for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; index++)
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

void createFramebuffers() {
	createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
				details.surfaceFormat.format, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
				vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage.data, colorImage.memory);
	colorImage.view = createImageView(colorImage.data, details.surfaceFormat.format, vk::ImageAspectFlagBits::eColor,
									  1);

	createImage(details.swapchainExtent.width, details.swapchainExtent.height, 1, details.sampleCount,
				details.depthStencilFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage.data, depthImage.memory);
	depthImage.view = createImageView(depthImage.data, details.depthStencilFormat, vk::ImageAspectFlagBits::eDepth, 1);

	for (auto &swapchainView : swapchainViews) {
		std::array<vk::ImageView, 3> attachments{colorImage.view, depthImage.view, swapchainView};

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

void loadImage(uint32_t width, uint32_t height, uint32_t levels, vk::SampleCountFlagBits samples, vk::Format format,
			   std::vector<uint8_t> &data) {
	svh::Image image;
	svh::Buffer buffer;

	createBuffer(data.size(), vk::BufferUsageFlagBits::eTransferSrc,
				 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer.data,
				 buffer.memory);

	auto imageData = device.mapMemory(buffer.memory, 0, data.size());
	std::memcpy(imageData, data.data(), data.size());
	device.unmapMemory(buffer.memory);

	createImage(width, height, levels, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, image.data, image.memory);
	transitionImageLayout(image.data, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, levels);
	copyBufferToImage(buffer.data, image.data, width, height);
	transitionImageLayout(image.data, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
						  levels);
	image.view = createImageView(image.data, format, vk::ImageAspectFlagBits::eColor, levels);
	images.push_back(image);

	device.destroyBuffer(buffer.data);
	device.freeMemory(buffer.memory);
}

void loadMesh(tinygltf::Model &modelData, tinygltf::Mesh &meshData, glm::mat4 transform) {
	for (auto &primitive : meshData.primitives) {
		svh::Mesh mesh;
		mesh.indexOffset = indices.size();
		mesh.vertexOffset = vertices.size();

		auto &indexView = modelData.bufferViews.at(primitive.indices);
		auto &indexBuffer = modelData.buffers.at(indexView.buffer);

		mesh.indexLength = indexView.byteLength / sizeof(uint16_t);
		indices.insert(indices.end(), indexBuffer.data.begin() + indexView.byteOffset,
					   indexBuffer.data.begin() + indexView.byteOffset + indexView.byteLength);

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

		for (int index = 0; index < mesh.vertexLength; index++) {
			svh::Vertex vertex;
			vertex.pos = transform * glm::vec4{positions.at(index), 1.0f};
			vertex.nor = transform * glm::vec4{positions.at(index), 0.0f};
			vertex.tex = positions.at(index);
			vertices.push_back(vertex);
		}
	}
}

void loadNode(tinygltf::Model &modelData, tinygltf::Node &nodeData, glm::mat4 transform) {
	glm::mat4 scale{1.0f}, rotation{1.0f}, translation{1.0f};

	if (!nodeData.rotation.empty())
		rotation = glm::rotate(static_cast<float_t>(nodeData.rotation.at(3)),
							   glm::vec3{nodeData.rotation.at(0), nodeData.rotation.at(1), nodeData.rotation.at(2)});
	for (int i = 0; i < nodeData.scale.size(); i++)
		scale[i][i] = nodeData.scale.at(i);
	for (int i = 0; i < nodeData.translation.size(); i++)
		translation[3][i] = nodeData.translation.at(i);

	transform = transform * translation * rotation * scale;

	if (nodeData.mesh >= 0 && nodeData.mesh < modelData.meshes.size())
		loadMesh(modelData, modelData.meshes.at(nodeData.mesh), transform);

	for (auto &childIndex : nodeData.children)
		loadNode(modelData, modelData.nodes.at(childIndex), transform);

	for (auto &material : modelData.materials) {
		for (auto &value : material.values) {
			if (!value.first.compare("baseColorTexture")) {
				auto &image = modelData.images.at(modelData.textures.at(value.second.TextureIndex()).source);
				meshes.at(models.back().meshIndex + value.second.TextureTexCoord()).imageIndex = images.size();
				loadImage(image.width, image.height, details.mipLevels, details.sampleCount, details.imageFormat,
						  image.image);
			}
		}
	}
}

void loadModel(std::string filename) {
	std::string error, warning;
	tinygltf::TinyGLTF loader;
	tinygltf::Model modelData;

	bool result = loader.LoadBinaryFromFile(&modelData, &error, &warning, filename);

#ifndef NDEBUG
	if (!warning.empty())
		std::cout << "GLTF Warning: " << warning << std::endl;
	if (!error.empty())
		std::cout << "GLTF Error: " << error << std::endl;
	if (!result)
		std::cout << "GLTF modelData could not be loaded!" << std::endl;
#endif

	svh::Model model;
	model.meshIndex = meshes.size();
	model.meshCount = 0;
	models.push_back(model);
	auto &scene = modelData.scenes.at(modelData.defaultScene);

	for (auto &nodeIndex : scene.nodes)
		loadNode(modelData, modelData.nodes.at(nodeIndex), glm::mat4{1.0f});
}

void setup() {
	initializeCore();
	createDevice();
	createSwapchain();
	createRenderPass();
	createPipeline();
	createFramebuffers();
}

void draw() {
}

void clear() {
	device.destroySampler(sampler);

	for (auto &framebuffer : framebuffers)
		device.destroyFramebuffer(framebuffer);
	device.destroyImageView(colorImage.view);
	device.destroyImage(colorImage.data);
	device.freeMemory(colorImage.memory);
	device.destroyImageView(depthImage.view);
	device.destroyImage(depthImage.data);
	device.freeMemory(depthImage.memory);
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
