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

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>

namespace executorch::backends::rocm {

using slim::SlimTensor;

inline SlimTensor from_etensor_to_hip(
    const executorch::runtime::etensor::Tensor& etensor,
    const aoti::slim::c10::Device& src_device,
    const aoti::slim::c10::Device& dst_device) {
  (void)etensor;
  (void)src_device;
  (void)dst_device;
  ET_CHECK_MSG(false, "from_etensor_to_hip not yet implemented (Phase 2)");
}

} // namespace executorch::backends::rocm
