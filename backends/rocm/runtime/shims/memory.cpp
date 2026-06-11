/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NOTE: This is a stub for Phase 1. Full implementation in Phase 2.
 */

#include <executorch/backends/rocm/runtime/shims/memory.h>

namespace executorch::backends::rocm {

void* rocm_malloc(size_t nbytes, hipStream_t stream) {
  (void)nbytes;
  (void)stream;
  ET_CHECK_MSG(false, "rocm_malloc not yet implemented (Phase 2)");
}

void rocm_free(void* ptr, hipStream_t stream) {
  (void)ptr;
  (void)stream;
  ET_LOG(Error, "rocm_free not yet implemented (Phase 2)");
}

Error rocm_memcpy_async(
    void* dst,
    const void* src,
    size_t nbytes,
    hipMemcpyKind direction,
    hipStream_t stream) {
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)direction;
  (void)stream;
  ET_CHECK_MSG(false, "rocm_memcpy not yet implemented (Phase 2)");
}

} // namespace executorch::backends::rocm
