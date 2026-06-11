/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NOTE: This is a stub for Phase 1. Full implementation in Phase 2.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <executorch/runtime/core/device_allocator.h>

namespace executorch::backends::rocm {

class RocmAllocator final : public executorch::runtime::DeviceAllocator {
 public:
  executorch::runtime::Result<void*> allocate(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      size_t alignment = kDefaultAlignment) override;

  void deallocate(void* ptr, executorch::runtime::etensor::DeviceIndex index)
      override;

  executorch::runtime::Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::etensor::DeviceType device_type() const override;

  static RocmAllocator& instance();

  static executorch::runtime::Result<void*> allocate_async(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      hipStream_t stream);

  static void deallocate_async(
      void* ptr,
      executorch::runtime::etensor::DeviceIndex index,
      hipStream_t stream);

  static executorch::runtime::Error memcpy_async(
      void* dst,
      const void* src,
      size_t nbytes,
      hipMemcpyKind direction,
      hipStream_t stream);
};

} // namespace executorch::backends::rocm
