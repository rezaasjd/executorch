/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/hip/guard.h>
#include <executorch/runtime/platform/log.h>
#include <limits>
#include <optional>
#include <unordered_map>

namespace executorch::backends::hip {

namespace {
thread_local std::unordered_map<DeviceIndex, hipStream_t> current_streams_;
} // namespace

Error setCurrentHIPStream(hipStream_t stream, DeviceIndex device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    ET_HIP_CHECK_OR_RETURN_ERROR(hipGetDevice(&tmp_device));
    device_index = static_cast<DeviceIndex>(tmp_device);
  }

  current_streams_[device_index] = stream;
  return Error::Ok;
}

Result<hipStream_t> getCurrentHIPStream(DeviceIndex device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    ET_HIP_CHECK_OR_RETURN_ERROR(hipGetDevice(&tmp_device));
    device_index = static_cast<DeviceIndex>(tmp_device);
  }

  auto it = current_streams_.find(device_index);
  if (it != current_streams_.end()) {
    return it->second;
  }

  hipStream_t stream;
  ET_HIP_CHECK_OR_RETURN_ERROR(hipStreamCreate(&stream));
  setCurrentHIPStream(stream, device_index);
  return stream;
}

std::optional<hipStream_t> peekCurrentHIPStream(DeviceIndex device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    if (hipGetDevice(&tmp_device) != hipSuccess) {
      return std::nullopt;
    }
    device_index = static_cast<DeviceIndex>(tmp_device);
  }

  auto it = current_streams_.find(device_index);
  if (it == current_streams_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void clearCurrentHIPStream(DeviceIndex device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    if (hipGetDevice(&tmp_device) != hipSuccess) {
      return;
    }
    device_index = static_cast<DeviceIndex>(tmp_device);
  }
  current_streams_.erase(device_index);
}

HIPGuard::HIPGuard(HIPGuard&& other) noexcept
    : original_device_index_(other.original_device_index_),
      current_device_index_(other.current_device_index_) {
  other.original_device_index_ = other.current_device_index_;
}

HIPGuard::~HIPGuard() {
  if (original_device_index_ != current_device_index_) {
    hipError_t err = hipSetDevice(original_device_index_);
    if (err != hipSuccess) {
      ET_LOG(
          Error,
          "~HIPGuard: Failed to restore device to %d: %s",
          static_cast<int>(original_device_index_),
          hipGetErrorString(err));
    }
  }
}

Error HIPGuard::set_index(DeviceIndex device_index) {
  int tmp_device = -1;
  ET_HIP_CHECK_OR_RETURN_ERROR(hipGetDevice(&tmp_device));

  original_device_index_ = static_cast<DeviceIndex>(tmp_device);
  current_device_index_ = device_index;

  if (current_device_index_ != original_device_index_) {
    ET_HIP_CHECK_OR_RETURN_ERROR(hipSetDevice(current_device_index_));
  }

  return Error::Ok;
}

Result<HIPGuard> HIPGuard::create(DeviceIndex device_index) {
  HIPGuard guard;
  ET_CHECK_OK_OR_RETURN_ERROR(guard.set_index(device_index));
  return guard;
}

HIPStreamGuard::HIPStreamGuard(HIPStreamGuard&& other) noexcept
    : device_guard_(std::move(other.device_guard_)),
      original_stream_(other.original_stream_),
      current_stream_(other.current_stream_),
      device_index_(other.device_index_) {
  other.original_stream_ = other.current_stream_;
}

HIPStreamGuard::~HIPStreamGuard() {
  if (original_stream_ != current_stream_) {
    Error err = setCurrentHIPStream(original_stream_, device_index_);
    if (err != Error::Ok) {
      ET_LOG(
          Error,
          "~HIPStreamGuard: Failed to restore stream for device %d",
          static_cast<int>(device_index_));
    }
  }
}

Error HIPStreamGuard::set_stream(
    hipStream_t stream,
    DeviceIndex device_index) {
  auto result = getCurrentHIPStream(device_index);
  if (!result.ok()) {
    ET_LOG(
        Error,
        "Failed to get current stream for device %d",
        static_cast<int>(device_index));
    return result.error();
  }

  original_stream_ = result.get();
  current_stream_ = stream;
  device_index_ = device_index;

  ET_CHECK_OK_OR_RETURN_ERROR(setCurrentHIPStream(stream, device_index));

  return Error::Ok;
}

Result<HIPStreamGuard> HIPStreamGuard::create(
    hipStream_t stream,
    DeviceIndex device_index) {
  auto guard_result = HIPGuard::create(device_index);
  ET_CHECK_OK_OR_RETURN_ERROR(guard_result.error());

  HIPStreamGuard stream_guard(std::move(guard_result.get()));
  ET_CHECK_OK_OR_RETURN_ERROR(
      stream_guard.set_stream(stream, device_index));

  return stream_guard;
}

} // namespace executorch::backends::hip
