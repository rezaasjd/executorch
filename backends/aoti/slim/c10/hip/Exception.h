/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

/// Checks a HIP expression and aborts on error.
#ifndef ET_HIP_CHECK
#define ET_HIP_CHECK(EXPR)                                            \
  do {                                                                \
    const hipError_t __err = EXPR;                                    \
    if (__err == hipSuccess) {                                        \
      break;                                                          \
    }                                                                 \
    ET_LOG(                                                           \
        Error,                                                        \
        "%s:%d HIP error: %s",                                        \
        __FILE__,                                                     \
        __LINE__,                                                     \
        hipGetErrorString(__err));                                    \
    ET_CHECK_MSG(false, "HIP error: %s", hipGetErrorString(__err));   \
  } while (0)
#endif

/// Checks a HIP expression and returns Error::Internal on failure.
#ifndef ET_HIP_CHECK_OR_RETURN_ERROR
#define ET_HIP_CHECK_OR_RETURN_ERROR(EXPR)           \
  do {                                               \
    const hipError_t __err = EXPR;                   \
    if (__err == hipSuccess) {                       \
      break;                                         \
    }                                                \
    ET_LOG(                                          \
        Error,                                       \
        "%s:%d HIP error: %s",                       \
        __FILE__,                                    \
        __LINE__,                                    \
        hipGetErrorString(__err));                   \
    return ::executorch::runtime::Error::Internal;   \
  } while (0)
#endif

/// Checks a HIP expression and logs a warning on error (non-fatal).
#ifndef ET_HIP_LOG_WARN
#define ET_HIP_LOG_WARN(EXPR)                                      \
  do {                                                             \
    const hipError_t __err = EXPR;                                 \
    if (SLIMTENSOR_UNLIKELY(__err != hipSuccess)) {                \
      [[maybe_unused]] auto error_unused = hipGetLastError();      \
      ET_LOG(Error, "HIP warning: %s", hipGetErrorString(__err));  \
    }                                                              \
  } while (0)
#endif

/// Kernel launch check macro (with return).
#ifndef ET_HIP_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR
#define ET_HIP_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR() \
  ET_HIP_CHECK_OR_RETURN_ERROR(hipGetLastError())
#endif

/// Kernel launch check macro (without return).
#ifndef ET_HIP_KERNEL_LAUNCH_CHECK
#define ET_HIP_KERNEL_LAUNCH_CHECK() ET_HIP_CHECK(hipGetLastError())
#endif
