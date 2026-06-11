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

#include <executorch/runtime/core/result.h>

namespace executorch::backends::rocm {

using executorch::runtime::Error;
using executorch::runtime::Result;

void* rocm_malloc(size_t nbytes, hipStream_t stream);

void rocm_free(void* ptr, hipStream_t stream);

Error rocm_memcpy_async(
    void* dst,
    const void* src,
    size_t nbytes,
    hipMemcpyKind direction,
    hipStream_t stream);

} // namespace executorch::backends::rocm
