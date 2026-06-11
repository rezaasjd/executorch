# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTE: This is a stub for Phase 1. Full implementation in Phase 3.

from typing import List

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir._warnings import experimental


@experimental(
    "This API and all of ROCm backend related functionality are experimental."
)
class RocmPartitioner(AotiPartitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        super().__init__("RocmBackend", compile_spec)
