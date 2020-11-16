cmake_minimum_required(VERSION 3.17)
project(Silibrand)

set(CMAKE_CXX_STANDARD 20)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/shaders)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/shaders/vert.spv COMMAND glslc ${PROJECT_SOURCE_DIR}/shaders/shader.vert -o ${PROJECT_BINARY_DIR}/shaders/vert.spv -O)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/shaders/frag.spv COMMAND glslc ${PROJECT_SOURCE_DIR}/shaders/shader.frag -o ${PROJECT_BINARY_DIR}/shaders/frag.spv -O)
add_custom_target(Shaders ALL DEPENDS ${PROJECT_BINARY_DIR}/shaders/vert.spv ${PROJECT_BINARY_DIR}/shaders/frag.spv)

add_executable(Silibrand Main.cpp)
add_dependencies(Silibrand Shaders)
target_precompile_headers(Silibrand PRIVATE Silibrand.hpp)
target_include_directories(Silibrand PRIVATE headers)
target_link_libraries(Silibrand glfw vulkan)
