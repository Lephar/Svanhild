cmake_minimum_required(VERSION 3.17)
project(Svanhild)

set(CMAKE_CXX_STANDARD 20)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/shaders)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/shaders/vert.spv COMMAND glslc ${PROJECT_SOURCE_DIR}/shaders/vertex.vert -o ${PROJECT_BINARY_DIR}/shaders/vert.spv -O)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/shaders/frag.spv COMMAND glslc ${PROJECT_SOURCE_DIR}/shaders/fragment.frag -o ${PROJECT_BINARY_DIR}/shaders/frag.spv -O)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/shaders/sten.spv COMMAND glslc ${PROJECT_SOURCE_DIR}/shaders/stencil.frag -o ${PROJECT_BINARY_DIR}/shaders/sten.spv -O)
add_custom_target(Shaders ALL DEPENDS ${PROJECT_BINARY_DIR}/shaders/vert.spv ${PROJECT_BINARY_DIR}/shaders/frag.spv ${PROJECT_BINARY_DIR}/shaders/sten.spv)

file(COPY ${PROJECT_SOURCE_DIR}/models DESTINATION ${PROJECT_BINARY_DIR})
add_custom_target(Models ALL DEPENDS ${PROJECT_BINARY_DIR}/models)

add_executable(Svanhild Main.cpp)
add_dependencies(Svanhild Shaders)
target_precompile_headers(Svanhild PRIVATE Svanhild.hpp)
target_include_directories(Svanhild PRIVATE headers)
target_link_libraries(Svanhild glfw vulkan)
