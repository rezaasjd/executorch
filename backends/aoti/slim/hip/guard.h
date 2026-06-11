/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <optional>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/hip/Exception.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::hip {

using executorch::runtime::Error;
using executorch::runtime::Result;

using executorch::backends::aoti::slim::c10::DeviceIndex;

Error setCurrentHIPStream(hipStream_t stream, DeviceIndex device_index = -1);

Result<hipStream_t> getCurrentHIPStream(DeviceIndex device_index = -1);

std::optional<hipStream_t> peekCurrentHIPStream(DeviceIndex device_index = -1);

void clearCurrentHIPStream(DeviceIndex device_index = -1);

class HIPGuard {
 private:
  explicit HIPGuard()
      : original_device_index_(-1), current_device_index_(-1) {}

 public:
  static Result<HIPGuard> create(DeviceIndex device_index);

  HIPGuard(const HIPGuard&) = delete;
  HIPGuard& operator=(const HIPGuard&) = delete;

  HIPGuard(HIPGuard&& other) noexcept;
  HIPGuard& operator=(HIPGuard&& other) = delete;

  ~HIPGuard();

  Error set_index(DeviceIndex device_index);

  DeviceIndex original_device() const {
    return original_device_index_;
  }

  DeviceIndex current_device() const {
    return current_device_index_;
  }

 private:
  DeviceIndex original_device_index_;
  DeviceIndex current_device_index_;
};

class HIPStreamGuard {
 private:
  explicit HIPStreamGuard(HIPGuard&& guard)
      : device_guard_(std::move(guard)),
        original_stream_(nullptr),
        current_stream_(nullptr),
        device_index_(-1) {}

 public:
  static Result<HIPStreamGuard> create(
      hipStream_t stream,
      DeviceIndex device_index);

  HIPStreamGuard(const HIPStreamGuard&) = delete;
  HIPStreamGuard& operator=(const HIPStreamGuard&) = delete;

  HIPStreamGuard(HIPStreamGuard&& other) noexcept;
  HIPStreamGuard& operator=(HIPStreamGuard&& other) noexcept = delete;

  ~HIPStreamGuard();

  Error set_stream(hipStream_t stream, DeviceIndex device_index);

  hipStream_t stream() const {
    return current_stream_;
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

 private:
  HIPGuard device_guard_;
  hipStream_t original_stream_ = nullptr;
  hipStream_t current_stream_ = nullptr;
  DeviceIndex device_index_;
};

} // namespace executorch::backends::hip
