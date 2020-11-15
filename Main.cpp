#include "Silibrand.hpp"

std::string title, game, engine;
uint32_t width, height;
GLFWwindow *window;

vk::Instance instance;
vk::SurfaceKHR surface;

#ifndef NDEBUG

vk::DispatchLoaderDynamic loader;
vk::DebugUtilsMessengerEXT messenger;

VKAPI_ATTR VkBool32 VKAPI_CALL messageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
											   VkDebugUtilsMessageTypeFlagsEXT type,
											   const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
											   void *pUserData) {
	static_cast<void>(type);
	static_cast<void>(severity);
	static_cast<void>(pUserData);

	std::cout << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

#endif

void initializeCore() {
	width = 800;
	height = 600;
	title = "0";
	game = "Silibrand";
	engine = "Svanhild Engine";

	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
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
	applicationInfo.pApplicationName = game.c_str();
	applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
	applicationInfo.pEngineName = engine.c_str();
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

void setup() {
	initializeCore();
}

void draw() {
}

void clear() {
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
