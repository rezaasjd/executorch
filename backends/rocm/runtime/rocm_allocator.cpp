/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NOTE: This is a stub for Phase 1. Full implementation in Phase 2.
 */

#include <executorch/backends/aoti/slim/c10/hip/Exception.h>
#include <executorch/backends/aoti/slim/hip/guard.h>
#include <executorch/backends/rocm/runtime/rocm_allocator.h>
#include <executorch/extension/hip/caller_stream.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::rocm {

RocmAllocator& RocmAllocator::instance() {
  static RocmAllocator allocator;
  return allocator;
}

executorch::runtime::Result<void*> RocmAllocator::allocate(
    size_t nbytes,
    executorch::runtime::etensor::DeviceIndex index,
    size_t alignment) {
  (void)nbytes;
  (void)index;
  (void)alignment;
  ET_CHECK_MSG(false, "RocmAllocator not yet implemented (Phase 2)");
}

void RocmAllocator::deallocate(
    void* ptr,
    executorch::runtime::etensor::DeviceIndex index) {
  (void)ptr;
  (void)index;
  ET_LOG(Error, "RocmAllocator not yet implemented (Phase 2)");
}

executorch::runtime::Error RocmAllocator::copy_host_to_device(
    void* dst,
    const void* src,
    size_t nbytes,
    executorch::runtime::etensor::DeviceIndex index) {
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)index;
  ET_CHECK_MSG(false, "RocmAllocator not yet implemented (Phase 2)");
}

executorch::runtime::Error RocmAllocator::copy_device_to_host(
    void* dst,
    const void* src,
    size_t nbytes,
    executorch::runtime::etensor::DeviceIndex index) {
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)index;
  ET_CHECK_MSG(false, "RocmAllocator not yet implemented (Phase 2)");
}

executorch::runtime::etensor::DeviceType RocmAllocator::device_type() const {
  ET_CHECK_MSG(false, "RocmAllocator not yet implemented (Phase 2)");
}

} // namespace executorch::backends::rocm
