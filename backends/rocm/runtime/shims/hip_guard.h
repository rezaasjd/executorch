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
#include <optional>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::rocm {

using executorch::runtime::Error;
using executorch::runtime::Result;

Error setCurrentHIPDevice(int device_index);

Result<int> getCurrentHIPDevice();

} // namespace executorch::backends::rocm
