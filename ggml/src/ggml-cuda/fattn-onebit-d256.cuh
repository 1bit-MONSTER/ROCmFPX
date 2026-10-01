// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "common.cuh"

// Prompt-processing FlashAttention for 256-wide heads on RDNA3.5 (gfx1151),
// F32 queries against an F16 K/V cache, using gfx11 WMMA for both products.
// GGML_ONEBIT_FA256=0 turns it off; GGML_ONEBIT_FLASH_PREFILL=<alpha> enables the
// opt-in sparse prefill (see the .cu file).
bool ggml_cuda_fattn_onebit_d256_eligible(const ggml_tensor * dst);
void ggml_cuda_fattn_onebit_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
