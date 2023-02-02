cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

# use cub in a safe manner, see:
# https://github.com/pytorch/pytorch/pull/55292
# https://github.com/pytorch/pytorch/pull/66219
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_QUALIFIER=::cuaev::cub")
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_PREFIX='namespace cuaev {'")
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_POSTFIX=}")

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
target_link_libraries(ani -Wl,--no-as-needed cuaev)

# test_model
project(test_model LANGUAGES CXX)
add_executable(test_model test_model.cpp)
target_include_directories(test_model PRIVATE ./)
target_link_libraries(test_model PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(test_model PRIVATE ${Python_LIBRARIES})
# -Wl,--no-as-needed flag forces the link to happen
# https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
target_link_libraries(test_model PRIVATE -Wl,--no-as-needed cuaev)
target_link_libraries(test_model PRIVATE -Wl,--no-as-needed ani)
