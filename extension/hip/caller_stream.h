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
#include <type_traits>

#include <executorch/extension/hip/export.h>

namespace executorch::extension::hip {

/**
 * The HIP stream selected by the innermost CallerStreamGuard active on this
 * thread, or std::nullopt if none is active.
 *
 * This reports only a stream the caller explicitly selected, so a backend can
 * honor that choice or fall back to its own default. It is backend-neutral: any
 * HIP backend (e.g. the ROCm/AOTI delegate) can consult it, so a single
 * caller-provided stream can drive several delegates in one program.
 */
EXECUTORCH_EXTENSION_HIP_API std::optional<hipStream_t> getCallerStream();

/**
 * Scopes, for the calling thread, the HIP stream a backend should run on, and
 * restores the previous selection on destruction. Scope it on the thread that
 * runs the call; the selection is one value per thread.
 */
class EXECUTORCH_EXTENSION_HIP_API CallerStreamGuard {
 public:
  explicit CallerStreamGuard(hipStream_t stream);
  ~CallerStreamGuard();
  CallerStreamGuard(const CallerStreamGuard&) = delete;
  CallerStreamGuard& operator=(const CallerStreamGuard&) = delete;
  CallerStreamGuard(CallerStreamGuard&&) = delete;
  CallerStreamGuard& operator=(CallerStreamGuard&&) = delete;

 private:
  std::optional<hipStream_t> previous_;
};

static_assert(std::is_trivially_copyable_v<std::optional<hipStream_t>>);

} // namespace executorch::extension::hip
