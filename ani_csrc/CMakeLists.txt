cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CXX_STANDARD 14)
project(cuaev LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})

# cuaev lib
add_library(cuaev SHARED cuaev.cpp aev.cu aev.h cuaev_cub.cuh)
target_include_directories(cuaev PRIVATE ./)
target_link_libraries(cuaev ${TORCH_LIBRARIES})
target_link_libraries(cuaev ${Python_LIBRARIES})

# ani interface lib
project(ani LANGUAGES CXX)
find_package(OpenMP REQUIRED)
add_library(ani SHARED mnp.cpp ani.cpp)
target_include_directories(ani PRIVATE ./)
target_link_libraries(ani ${TORCH_LIBRARIES})
target_link_libraries(ani ${Python_LIBRARIES})
target_link_libraries(ani OpenMP::OpenMP_CXX)

# test_model
project(test_model LANGUAGES CXX)
add_executable(test_model test_model.cpp)
target_include_directories(test_model PRIVATE ./)
target_link_libraries(test_model ${TORCH_LIBRARIES})
target_link_libraries(test_model ${Python_LIBRARIES})
target_link_libraries(test_model ani)
target_link_libraries(test_model cuaev)
