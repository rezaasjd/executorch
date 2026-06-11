# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTE: This is a stub for Phase 1. Full implementation in Phase 3.

from typing import Any, Dict, List, Optional

from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec


@experimental(
    "This API and all of ROCm backend related functionality are experimental."
)
class RocmBackend(AotiBackend, BackendDetails):
    @classmethod
    def get_device_name(cls) -> str:
        return "hip"

    @staticmethod
    def _setup_rocm_environment() -> bool:
        return False

    @classmethod
    def save_data_externally(cls) -> bool:
        return True

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {}

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[Any]:
        return []

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "loop_ordering_after_fusion": False,
            "emulate_precision_casts": True,
            "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.link_libtorch": False,
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.package_constants_on_disk_format": "binary_blob",
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_conv_backends": "TRITON",
        }
        return options

    @classmethod
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: Optional[List[CompileSpec]] = None
    ):
        import contextlib

        return contextlib.nullcontext()

    @classmethod
    def release_moved_tensors(
        cls,
        device_edge_program,
        compile_specs: List[CompileSpec],
    ) -> None:
        pass
