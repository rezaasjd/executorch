/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NOTE: This is a stub for Phase 1. Full implementation in Phase 2.
 */

#include <executorch/backends/rocm/runtime/rocm_backend.h>
#include <executorch/backends/rocm/runtime/rocm_allocator.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/device_allocator.h>

namespace executorch::backends::rocm {

bool RocmBackend::is_available() const {
  return 0;
}

executorch::runtime::Error RocmBackend::set_option(
    executorch::runtime::BackendOptionContext& context,
    const executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  (void)context;
  (void)backend_options;
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error RocmBackend::get_option(
    executorch::runtime::BackendOptionContext& context,
    executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  (void)context;
  (void)backend_options;
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Result<executorch::runtime::DelegateHandle*>
RocmBackend::init(
    executorch::runtime::BackendInitContext& context,
    executorch::runtime::FreeableBuffer* processed,
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
        compile_specs) const {
  (void)context;
  (void)processed;
  (void)compile_specs;
  ET_CHECK_MSG(false, "RocmBackend::init not yet implemented (Phase 2)");
}

executorch::runtime::Error RocmBackend::execute(
    executorch::runtime::BackendExecutionContext& context,
    executorch::runtime::DelegateHandle* handle,
    executorch::runtime::Span<executorch::runtime::EValue*> args) const {
  (void)context;
  (void)handle;
  (void)args;
  ET_CHECK_MSG(false, "RocmBackend::execute not yet implemented (Phase 2)");
}

void RocmBackend::destroy(
    executorch::runtime::DelegateHandle* handle) const {
  (void)handle;
  ET_LOG(Error, "RocmBackend::destroy not yet implemented (Phase 2)");
}

} // namespace executorch::backends::rocm

namespace executorch::backends {
namespace {
auto cls = rocm::RocmBackend();
executorch::runtime::Backend backend{"RocmBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);

static bool rocm_allocator_registered = [] {
  executorch::runtime::register_device_allocator(
      &rocm::RocmAllocator::instance());
  return true;
}();
} // namespace
} // namespace executorch::backends
