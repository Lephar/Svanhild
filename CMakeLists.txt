cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)

project (Svanhild)

add_custom_command(OUTPUT shaders/vertex.spv COMMAND glslc shaders/vertex.vert -o shaders/vertex.spv -O)
add_custom_command(OUTPUT shaders/fragment.spv COMMAND glslc shaders/fragment.frag -o shaders/fragment.spv -O)
file(COPY models shaders DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(Shaders ALL DEPENDS ${PROJECT_BINARY_DIR}/shaders/vertex.spv ${PROJECT_BINARY_DIR}/shaders/fragment.spv)
add_custom_target(Models ALL DEPENDS ${PROJECT_BINARY_DIR}/models)

add_executable(Svanhild Svanhild.cpp Svanhild.hpp)
add_dependencies(Svanhild Shaders Models)

if(WIN32)
	find_package(Vulkan REQUIRED)
	target_include_directories(Svanhild PRIVATE Vulkan::Vulkan headers)
	target_link_libraries(Svanhild Vulkan::Vulkan ${PROJECT_SOURCE_DIR}/libraries/GLFW/glfw3.lib)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if(UNIX AND NOT APPLE)
	target_precompile_headers(Svanhild PRIVATE Svanhild.hpp)
	target_include_directories(Svanhild PRIVATE headers)
	target_link_libraries(Svanhild glfw vulkan)
endif()
