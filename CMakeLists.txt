cmake_minimum_required(VERSION 3.17)
project(Silibrand)

set(CMAKE_CXX_STANDARD 20)

add_executable(Silibrand Main.cpp)
target_precompile_headers(Silibrand PRIVATE Silibrand.hpp)
target_include_directories(Silibrand PRIVATE headers)
target_link_libraries(Silibrand glfw vulkan)
