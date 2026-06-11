# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

find_package(hip REQUIRED)

add_library(extension_hip SHARED caller_stream.cpp)
target_link_libraries(extension_hip PUBLIC hip::host)
target_include_directories(
  extension_hip PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
                       $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
                       $<INSTALL_INTERFACE:include>
)
target_compile_options(
  extension_hip PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_common_compile_options}>"
)
target_compile_definitions(
  extension_hip PRIVATE EXECUTORCH_EXTENSION_HIP_BUILDING
)

install(
  TARGETS extension_hip
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
