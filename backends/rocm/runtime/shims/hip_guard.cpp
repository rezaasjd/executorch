/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NOTE: This is a stub for Phase 1. Full implementation in Phase 2.
 */

#include <executorch/backends/rocm/runtime/shims/hip_guard.h>

namespace executorch::backends::rocm {

Error setCurrentHIPDevice(int device_index) {
  (void)device_index;
  ET_CHECK_MSG(false, "HIP guard not yet implemented (Phase 2)");
}

Result<int> getCurrentHIPDevice() {
  ET_CHECK_MSG(false, "HIP guard not yet implemented (Phase 2)");
}

} // namespace executorch::backends::rocm
