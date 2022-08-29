##########################################
# CMake build system for plugin examples.
# The is meant to be used as a template for plugins that are
# distributed independent from the LAMMPS package.
##########################################

cmake_minimum_required(VERSION 3.10)

# enforce out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed. You must create and use a build directory. "
    "Please remove CMakeCache.txt and CMakeFiles first.")
endif()

project(plugins VERSION 1.0 LANGUAGES CXX)

# when this file is included as subdirectory in the LAMMPS build, many settings are directly imported
if(LAMMPS_DIR)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
else()
  # NOTE: the next line should be commented out when used outside of the LAMMPS package
  get_filename_component(LAMMPS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../src ABSOLUTE)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR} CACHE PATH "Location of LAMMPS headers")
  if(NOT LAMMPS_HEADER_DIR)
    message(FATAL_ERROR "Must set LAMMPS_HEADER_DIR")
  endif()
  # by default, install into $HOME/.local (not /usr/local),
  # so that no root access (and sudo) is needed
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
endif()

# C++14 is required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Need -restrict with Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(CheckIncludeFileCXX)
if(NOT LAMMPS_DIR)
  include(LAMMPSInterfaceCXX)
endif()


##########################

# find/fetch libtorch
set(libtorch_VERSION 1.12.1)
# find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
find_package(Torch)
if(NOT Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - not found")
  message(STATUS "Fetching libtorch")
  include(FetchContent)
  FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-${libtorch_VERSION}%2Bcu116.zip
    SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

# packages
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(ani_csrc)

# building the plugins
add_library(ani_plugin MODULE ani_plugin.cpp pair_ani.cpp)
target_include_directories(ani_plugin PRIVATE "${LAMMPS_HEADER_DIR}/OPENMP")
target_include_directories(ani_plugin PRIVATE "ani_csrc")
target_link_libraries(ani_plugin PRIVATE lammps)

# link to ani
target_link_libraries(ani_plugin PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES})
# -Wl,--no-as-needed flag forces the link to happen
# https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
target_link_libraries(ani_plugin PRIVATE -Wl,--no-as-needed cuaev)
target_link_libraries(ani_plugin PRIVATE -Wl,--no-as-needed ani)

set_target_properties(ani_plugin PROPERTIES PREFIX "" SUFFIX ".so" LINK_FLAGS "-rdynamic")

install(TARGETS ani_plugin LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_custom_target(plugins ALL ${CMAKE_COMMAND} -E echo "Building Plugins" DEPENDS ani_plugin)
