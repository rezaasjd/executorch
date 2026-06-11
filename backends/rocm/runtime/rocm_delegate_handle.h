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

#include <memory>

#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/slim/c10/hip/Exception.h>

namespace executorch::backends::rocm {

enum class HipGraphPhase {
  Disabled,
  Warmup,
  Replay,
  Capture,
};

struct HipGraphState {
  HipGraphPhase phase = HipGraphPhase::Disabled;
  int warmup_remaining = 0;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  std::vector<void*> static_input_ptrs;
  std::vector<size_t> static_input_nbytes;
  std::vector<void*> static_output_ptrs;
  std::vector<size_t> static_output_nbytes;
};

struct RocmDelegateHandle : public AOTIDelegateHandle {
  std::shared_ptr<hipStream_t> hip_stream;
  HipGraphState hip_graph_state;
};

} // namespace executorch::backends::rocm
