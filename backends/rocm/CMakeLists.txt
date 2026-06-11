# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI ROCm (HIP) backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

find_package(hip REQUIRED)

# Common C++ compile options for ROCm backend targets.
set(_rocm_cxx_compile_options -fexceptions -frtti -fPIC)

# Platform-specific linker option for exporting symbols from shared libs.
set(_rocm_export_dynamic_option "")
if(NOT WIN32)
  if(APPLE)
    set(_rocm_export_dynamic_option -Wl,-export_dynamic)
  else()
    set(_rocm_export_dynamic_option -Wl,--export-dynamic)
  endif()
endif()

# Use ExecuTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

# Platform utilities (load_library, close_library, etc.)
set(_rocm_platform_sources runtime/platform/platform.cpp)
add_library(rocm_platform STATIC ${_rocm_platform_sources})

target_include_directories(
  rocm_platform
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_compile_options(
  rocm_platform
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_rocm_cxx_compile_options}>"
)

target_link_libraries(rocm_platform PRIVATE executorch_core ${CMAKE_DL_LIBS})

install(
  TARGETS rocm_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# HIP-specific AOTI shim symbols (dynamically linked)
set(_aoti_rocm_shim_sources
    runtime/rocm_allocator.cpp
    runtime/shims/memory.cpp
    runtime/shims/hip_guard.cpp
)

# Only build HIP shims when HIP compiler is available.
enable_language(HIP)

add_library(aoti_rocm_shims SHARED ${_aoti_rocm_shim_sources})

# Define HIP_AVAILABLE to use SlimTensor on GPU
target_compile_definitions(aoti_rocm_shims PRIVATE HIP_AVAILABLE=1)

target_include_directories(
  aoti_rocm_shims
  PUBLIC ${HIP_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)

target_compile_options(
  aoti_rocm_shims
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_rocm_cxx_compile_options}>"
)

# Ensure symbols are exported properly
if(_rocm_export_dynamic_option)
  target_link_options(aoti_rocm_shims PUBLIC ${_rocm_export_dynamic_option})
endif()

target_link_libraries(
  aoti_rocm_shims
  PRIVATE rocm_platform
  PUBLIC -Wl,--whole-archive aoti_common_shims_slim -Wl,--no-whole-archive
         hip::host ${CMAKE_DL_LIBS}
)

executorch_target_link_options_shared_lib(aoti_rocm_shims)

install(
  TARGETS aoti_rocm_shims
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# ROCm backend implementation
set(_aoti_rocm_backend_sources runtime/rocm_backend.cpp)

add_library(aoti_rocm_backend STATIC ${_aoti_rocm_backend_sources})

target_include_directories(
  aoti_rocm_backend
  PUBLIC ${HIP_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)
target_compile_options(
  aoti_rocm_backend
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_rocm_cxx_compile_options}>"
)
# Ensure symbols are exported properly
if(_rocm_export_dynamic_option)
  target_link_options(aoti_rocm_backend PUBLIC ${_rocm_export_dynamic_option})
endif()

target_link_libraries(
  aoti_rocm_backend
  PUBLIC rocm_platform extension_tensor hip::host extension_hip ${CMAKE_DL_LIBS}
)

target_link_libraries(aoti_rocm_backend PUBLIC aoti_rocm_shims)

executorch_target_link_options_shared_lib(aoti_rocm_backend)

install(
  TARGETS aoti_rocm_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
