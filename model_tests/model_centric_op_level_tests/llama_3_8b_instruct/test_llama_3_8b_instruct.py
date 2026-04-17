import os
import sys
import math
import unittest
import pytest
import torch
import torch.nn.functional as F

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from utils_llama_3_8b_instruct import (
    # Infrastructure
    ParameterizedTestMeta,
    DEVICE,
    TOLERANCES,
    make_strided_tensor,
    # Architecture constants
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GQA_GROUPS,
    SCALE,
    SLIDING_WINDOW,
    NUM_LAYERS,
    DEFAULT_DTYPE,
    VOCAB_SIZE,
    ROPE_THETA,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
 
    # Shorthand dtypes
    BF16,
    F16,
    F32,
    I64,

 
    # SDPA pre-built param dicts
    PREFILL_PARAMS,
    DECODE_PARAMS,
    DTYPE_PARAMS,
    NUMERIC_COVERAGE_PARAMS,
    GROWING_KV_PARAMS,
 
    # Tensor factories
    make_qkv,
    make_tensor,
    _t,
    _W,
    cached_randn,
    expand_kv,
 
    # Mask builders
    causal_mask,
    sdpa_fn,
 
    # Comparison helpers
    compare_with_cpu,
)

S = slice
_ = slice(None)


# ─────────────────────────────────────────────────────────────────────────────
# TestAdd
# ─────────────────────────────────────────────────────────────────────────────

class TestAdd(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.add patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    Four call signatures appear in the model:

      scalar      torch.add(tensor, scalar)   — arange offset, epsilon
      binary      torch.add(tensor, tensor)   — same shape, no broadcast
      broadcast   torch.add(tensor, tensor)   — inputs differ in seq dim,
                                                output is larger (RoPE path)

    pytestmark stamps every generated method with @pytest.mark.torch_add.
    """

    pytestmark = pytest.mark.torch_add

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # Scalar patterns  torch.add(tensor, scalar)
        # ------------------------------------------------------------------

        # arange base: [64] int64 + scalar 0
        ("test_torch_add_pattern_000", "_run_add_scalar"): {
            "param_sets": {
                "scalar_64_int64_0_eager": (
                    make_strided_tensor((64,), (1,), torch.int64),
                    0,
                    False,
                ),
                "scalar_64_int64_0_compiled": (
                    make_strided_tensor((64,), (1,), torch.int64),
                    0,
                    True,
                ),
            }
        },

        # arange offset: [1] int64 + scalar 64
        ("test_torch_add_pattern_001", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1_int64_64_eager": (
                    make_strided_tensor((1,), (1,), torch.int64),
                    64,
                    False,
                ),
                "scalar_1_int64_64_compiled": (
                    make_strided_tensor((1,), (1,), torch.int64),
                    64,
                    True,
                ),
            }
        },

        # RMSNorm variance epsilon: [1,64,1] float32 + 1e-5
        ("test_torch_add_pattern_002", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1x64x1_f32_eps_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    1e-5,
                    False,
                ),
                "scalar_1x64x1_f32_eps_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    1e-5,
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Binary tensor + tensor  (same shape, no broadcast)
        # ------------------------------------------------------------------

        # q_embed rotary prefill: [1,32,64,128] + [1,32,64,128]
        ("test_torch_add_pattern_003", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x64x128_bf16_eager": (
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    False,
                ),
                "binary_1x32x64x128_bf16_compiled": (
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    True,
                ),
            }
        },

        # q_embed rotary decode: [1,32,1,128] + [1,32,1,128]
        ("test_torch_add_pattern_004", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x1x128_bf16_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    False,
                ),
                "binary_1x32x1x128_bf16_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    True,
                ),
            }
        },

        # k_embed rotary decode: [1,8,1,128] + [1,8,1,128]
        ("test_torch_add_pattern_005", "_run_add_binary"): {
            "param_sets": {
                "binary_1x8x1x128_bf16_eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    False,
                ),
                "binary_1x8x1x128_bf16_compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    True,
                ),
            }
        },

        # residual + hidden decode: [1,1,4096] + [1,1,4096]
        ("test_torch_add_pattern_006", "_run_add_binary"): {
            "param_sets": {
                "binary_1x1x4096_bf16_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    False,
                ),
                "binary_1x1x4096_bf16_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_add_scalar(self, a, scalar, compiled):
        """torch.add(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.add(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_add_binary(self, a, b, compiled):
        """torch.add(tensor, tensor) — identical shapes, no broadcast."""
        compare_with_cpu(torch.add, a, b, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestSub
# ─────────────────────────────────────────────────────────────────────────────

class TestSub(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.sub patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    One call signature appears in the model:

      scalar      torch.sub(tensor, scalar)  — position index offset,
                                               non-contiguous slice of a
                                               [64, N] position-ids buffer.

    pytestmark stamps every generated method with @pytest.mark.torch_sub.
    """

    pytestmark = pytest.mark.torch_sub

    torch.manual_seed(0)

    PARAMS = {
        # position-ids slice: [1,1] int64 non-contiguous − scalar 1
        ("test_torch_sub_pattern_000", "_run_sub_scalar"): {
            "param_sets": {
                "scalar_1x1_int64_noncontig_sub1_eager": (
                    make_strided_tensor((1, 1), (64, 1), torch.int64),
                    1,
                    False,
                ),
                "scalar_1x1_int64_noncontig_sub1_compiled": (
                    make_strided_tensor((1, 1), (64, 1), torch.int64),
                    1,
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_sub_scalar(self, a, scalar, compiled):
        """torch.sub(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.sub(x, scalar),
            a,
            compiled=compiled,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TestCat
# ─────────────────────────────────────────────────────────────────────────────

class TestCat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cat patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    The scalar value is the concatenation dim (always -1 except the final
    KV-cache append which uses -2).

    Two structural variants appear:
      same_dim    both inputs have the same shape, cat along last dim (-1)
      mixed_seq   inputs differ in seq dim (KV-cache append), cat along -2
    """

    pytestmark = pytest.mark.torch_cat

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # same_dim — both inputs identical shape, concat on last dim (-1)
        # ------------------------------------------------------------------

        # RoPE sin/cos prefill float32: [1,64,64] + [1,64,64] → [1,64,128]
        ("test_torch_cat_pattern_000", "_run_cat"): {
            "param_sets": {
                "cat_1x64x64_f32_dim-1_eager": (
                    [
                        make_strided_tensor((1, 64, 64), (4096, 1, 64), torch.float32),
                        make_strided_tensor((1, 64, 64), (4096, 1, 64), torch.float32),
                    ],
                    -1,
                    False,
                ),
                "cat_1x64x64_f32_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 64, 64), (4096, 1, 64), torch.float32),
                        make_strided_tensor((1, 64, 64), (4096, 1, 64), torch.float32),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # RoPE sin/cos decode float32: [1,1,64] + [1,1,64] → [1,1,128]
        ("test_torch_cat_pattern_001", "_run_cat"): {
            "param_sets": {
                "cat_1x1x64_f32_dim-1_eager": (
                    [
                        make_strided_tensor((1, 1, 64), (64, 1, 1), torch.float32),
                        make_strided_tensor((1, 1, 64), (64, 1, 1), torch.float32),
                    ],
                    -1,
                    False,
                ),
                "cat_1x1x64_f32_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 1, 64), (64, 1, 1), torch.float32),
                        make_strided_tensor((1, 1, 64), (64, 1, 1), torch.float32),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # q head RoPE prefill float16: [1,32,64,64] + [1,32,64,64] → [1,32,64,128]
        ("test_torch_cat_pattern_002", "_run_cat"): {
            "param_sets": {
                "cat_1x32x64x64_bf16_dim-1_eager": (
                    [
                        make_strided_tensor((1, 32, 64, 64), (131072,  64, 2048, 1), torch.float16),
                        make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    ],
                    -1,
                    False,
                ),
                "cat_1x32x64x64_bf16_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 32, 64, 64), (131072,  64, 2048, 1), torch.float16),
                        make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # k/v head RoPE prefill float16: [1,8,64,64] + [1,8,64,64] → [1,8,64,128]
        ("test_torch_cat_pattern_003", "_run_cat"): {
            "param_sets": {
                "cat_1x8x64x64_bf16_dim-1_eager": (
                    [
                        make_strided_tensor((1, 8, 64, 64), (32768,  64,  512, 1), torch.float16),
                        make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    ],
                    -1,
                    False,
                ),
                "cat_1x8x64x64_bf16_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 8, 64, 64), (32768,  64,  512, 1), torch.float16),
                        make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # q head RoPE decode float16: [1,32,1,64] + [1,32,1,64] → [1,32,1,128]
        ("test_torch_cat_pattern_004", "_run_cat"): {
            "param_sets": {
                "cat_1x32x1x64_bf16_dim-1_eager": (
                    [
                        make_strided_tensor((1, 32, 1, 64), (2048,  64, 2048, 1), torch.float16),
                        make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16),
                    ],
                    -1,
                    False,
                ),
                "cat_1x32x1x64_bf16_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 32, 1, 64), (2048,  64, 2048, 1), torch.float16),
                        make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # k/v head RoPE decode float16: [1,8,1,64] + [1,8,1,64] → [1,8,1,128]
        ("test_torch_cat_pattern_005", "_run_cat"): {
            "param_sets": {
                "cat_1x8x1x64_bf16_dim-1_eager": (
                    [
                        make_strided_tensor((1, 8, 1, 64), (512,  64,  512, 1), torch.float16),
                        make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16),
                    ],
                    -1,
                    False,
                ),
                "cat_1x8x1x64_bf16_dim-1_compiled": (
                    [
                        make_strided_tensor((1, 8, 1, 64), (512,  64,  512, 1), torch.float16),
                        make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16),
                    ],
                    -1,
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # mixed_seq — inputs differ in seq dim, concat on dim -2
        # KV-cache append: [1,8,64,128] + [1,8,1,128] → [1,8,65,128]
        # ------------------------------------------------------------------

        ("test_torch_cat_pattern_006", "_run_cat_with_shape_check"): {
            "param_sets": {
                "cat_kvcache_1x8x64x128_plus_1x8x1x128_eager": (
                    [
                        make_strided_tensor((1, 8, 64, 128), (65536, 8192, 128, 1), torch.float16),
                        make_strided_tensor((1, 8,  1, 128), (1024,   128, 1024, 1), torch.float16),
                    ],
                    -2,
                    (1, 8, 65, 128),
                    False,
                ),
                "cat_kvcache_1x8x64x128_plus_1x8x1x128_compiled": (
                    [
                        make_strided_tensor((1, 8, 64, 128), (65536, 8192, 128, 1), torch.float16),
                        make_strided_tensor((1, 8,  1, 128), (1024,   128, 1024, 1), torch.float16),
                    ],
                    -2,
                    (1, 8, 65, 128),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_cat(self, tensors, dim, compiled):
        """torch.cat(tensor_list, dim) — standard concatenation."""
        compare_with_cpu(
            lambda *xs: torch.cat(list(xs), dim=dim),
            *tensors,
            compiled=compiled,
        )

    def _run_cat_with_shape_check(self, tensors, dim, expected_shape, compiled):
        """torch.cat where output shape must be verified (mixed-seq KV append)."""
        def fn(*xs):
            out = torch.cat(list(xs), dim=dim)
            assert tuple(out.shape) == expected_shape, (
                f"cat shape mismatch: expected {expected_shape}, got {tuple(out.shape)}"
            )
            return out

        compare_with_cpu(fn, *tensors, compiled=compiled)

# ─────────────────────────────────────────────────────────────────────────────
# TestRsqrt
# ─────────────────────────────────────────────────────────────────────────────

class TestRsqrt(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.rsqrt patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    torch.rsqrt appears in the RMSNorm computation: rsqrt(variance + eps).

      prefill    input shape [1, 64, 1]  — one variance value per token
      decode     input shape [1,  1, 1]  — single token
    """

    pytestmark = pytest.mark.torch_rsqrt

    torch.manual_seed(0)

    PARAMS = {
        # RMSNorm prefill: rsqrt([1,64,1])
        ("test_torch_rsqrt_pattern_000", "_run_rsqrt"): {
            "param_sets": {
                "rsqrt_1x64x1_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    False,
                ),
                "rsqrt_1x64x1_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    True,
                ),
            }
        },

        # RMSNorm decode: rsqrt([1,1,1])
        ("test_torch_rsqrt_pattern_001", "_run_rsqrt"): {
            "param_sets": {
                "rsqrt_1x1x1_f32_eager": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float32),
                    False,
                ),
                "rsqrt_1x1x1_f32_compiled": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float32),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_rsqrt(self, a, compiled):
        """torch.rsqrt(tensor) — elementwise reciprocal square root."""
        compare_with_cpu(torch.rsqrt, a, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestUnsqueeze
# ─────────────────────────────────────────────────────────────────────────────

class TestUnsqueeze(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.unsqueeze patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    The dim is inferred from input → output shape change:

      (64,)        → (1, 64)          dim=0   position-ids prefill
      (1,)         → (1, 1)           dim=0   position-ids decode
      (1, 64, 128) → (1, 1, 64, 128) dim=1   hidden state expand for attention
    """

    pytestmark = pytest.mark.torch_unsqueeze

    torch.manual_seed(0)

    PARAMS = {
        # position-ids prefill: [64] → [1,64]
        ("test_torch_unsqueeze_pattern_000", "_run_unsqueeze"): {
            "param_sets": {
                "unsqueeze_64_int64_dim0_eager": (
                    make_strided_tensor((64,), (1,), torch.int64),
                    0,
                    False,
                ),
                "unsqueeze_64_int64_dim0_compiled": (
                    make_strided_tensor((64,), (1,), torch.int64),
                    0,
                    True,
                ),
            }
        },

        # position-ids decode: [1] → [1,1]
        ("test_torch_unsqueeze_pattern_001", "_run_unsqueeze"): {
            "param_sets": {
                "unsqueeze_1_int64_dim0_eager": (
                    make_strided_tensor((1,), (1,), torch.int64),
                    0,
                    False,
                ),
                "unsqueeze_1_int64_dim0_compiled": (
                    make_strided_tensor((1,), (1,), torch.int64),
                    0,
                    True,
                ),
            }
        },

        # hidden state expand: [1,64,128] → [1,1,64,128]
        ("test_torch_unsqueeze_pattern_002", "_run_unsqueeze"): {
            "param_sets": {
                "unsqueeze_1x64x128_bf16_dim1_eager": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float16),
                    1,
                    False,
                ),
                "unsqueeze_1x64x128_bf16_dim1_compiled": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float16),
                    1,
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_unsqueeze(self, a, dim, compiled):
        """torch.unsqueeze(tensor, dim) — insert a size-1 dimension."""
        compare_with_cpu(
            lambda x: torch.unsqueeze(x, dim),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestPow
# ─────────────────────────────────────────────────────────────────────────────

class TestPow(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.pow patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    torch.pow appears in RMSNorm: x ** 2 before mean → rsqrt.
    Exponent is always the scalar 2.

      prefill    [1, 64, 4096]  float32
      decode     [1,  1, 4096]  float32
    """

    pytestmark = pytest.mark.torch_pow

    torch.manual_seed(0)

    PARAMS = {
        # RMSNorm prefill: [1,64,4096] ** 2
        ("test_torch_pow_pattern_000", "_run_pow_scalar"): {
            "param_sets": {
                "pow_1x64x4096_f32_exp2_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    2,
                    False,
                ),
                "pow_1x64x4096_f32_exp2_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    2,
                    True,
                ),
            }
        },

        # RMSNorm decode: [1,1,4096] ** 2
        ("test_torch_pow_pattern_001", "_run_pow_scalar"): {
            "param_sets": {
                "pow_1x1x4096_f32_exp2_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    2,
                    False,
                ),
                "pow_1x1x4096_f32_exp2_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    2,
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_pow_scalar(self, a, exponent, compiled):
        """torch.pow(tensor, scalar_exponent)."""
        compare_with_cpu(
            lambda x: torch.pow(x, exponent),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestNe
# ─────────────────────────────────────────────────────────────────────────────

class TestNe(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.ne patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    torch.ne appears in attention mask generation: input_ids != pad_token_id.

    One pattern observed:
      [1, 64] int64 non-contiguous (stride (64,1) — row slice of position
      buffer) compared against a scalar, producing a bool mask.
    """

    pytestmark = pytest.mark.torch_ne

    torch.manual_seed(0)

    PARAMS = {
        # attention mask: [1,64] int64 non-contiguous != scalar
        ("test_torch_ne_pattern_000", "_run_ne_scalar"): {
            "param_sets": {
                "ne_1x64_int64_noncontig_eager": (
                    make_strided_tensor((1, 64), (64, 1), torch.int64),
                    0,
                    False,
                ),
                "ne_1x64_int64_noncontig_compiled": (
                    make_strided_tensor((1, 64), (64, 1), torch.int64),
                    0,
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_ne_scalar(self, a, scalar, compiled):
        """torch.ne(tensor, scalar) — elementwise not-equal, returns bool."""
        compare_with_cpu(
            lambda x: torch.ne(x, scalar),
            a,
            compiled=compiled,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TestMatmul
# ─────────────────────────────────────────────────────────────────────────────

class TestMatmul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.matmul patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    Two unique call-sites, both float32 with non-contiguous inputs:

      outer    [1,64,1] × [1,1,64] → [1,64,64]   attention score outer-product (prefill)
      inner    [1,64,1] × [1,1,1]  → [1,64,1]    scalar scaling / projection   (decode)

    Three call-site variants are covered per shape pair:
      torch.matmul(a, b)   — function form
      a.matmul(b)          — method alias
      a @ b                — operator alias

    Plus: zeros/ones special inputs, matmul→add fusion, and CPU-only
    inf/NaN propagation checks.
    """

    pytestmark = pytest.mark.torch_matmul

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic matmul  [1,64,1] @ [1,1,64] → [1,64,64]  prefill
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_000", "_run_matmul_test"): {
            "param_sets": {
                "1x64x1_1x1x64_eager": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1),  torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1),  torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "1x64x1_1x1x64_compiled": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1),  torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1),  torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_001  basic matmul  [1,64,1] @ [1,1,1] → [1,64,1]   decode
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_001", "_run_matmul_test"): {
            "param_sets": {
                "1x64x1_1x1x1_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "1x64x1_1x1x1_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_002  method alias  [1,64,1] @ [1,1,64] → [1,64,64]
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_002", "_run_matmul_test"): {
            "param_sets": {
                "method_1x64x1_1x1x64_eager": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: a.matmul(b),
                    False,
                ),
                "method_1x64x1_1x1x64_compiled": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: a.matmul(b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_003  method alias  [1,64,1] @ [1,1,1] → [1,64,1]
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_003", "_run_matmul_test"): {
            "param_sets": {
                "method_1x64x1_1x1x1_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: a.matmul(b),
                    False,
                ),
                "method_1x64x1_1x1x1_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: a.matmul(b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_004  @ operator  [1,64,1] @ [1,1,64] → [1,64,64]
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_004", "_run_matmul_test"): {
            "param_sets": {
                "bmm_op_1x64x1_1x1x64_eager": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: a @ b,
                    False,
                ),
                "bmm_op_1x64x1_1x1x64_compiled": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: a @ b,
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_005  @ operator  [1,64,1] @ [1,1,1] → [1,64,1]
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_005", "_run_matmul_test"): {
            "param_sets": {
                "bmm_op_1x64x1_1x1x1_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: a @ b,
                    False,
                ),
                "bmm_op_1x64x1_1x1x1_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: a @ b,
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_006  all-zeros A  [1,64,1] @ [1,1,64] → all-zeros out
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_006", "_run_matmul_test"): {
            "param_sets": {
                "zeros_a_1x64x1_1x1x64_eager": (
                    torch.zeros(1, 64,  1, dtype=torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "zeros_a_1x64x1_1x1x64_compiled": (
                    torch.zeros(1, 64,  1, dtype=torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_007  all-zeros B  [1,64,1] @ [1,1,1] → all-zeros out
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_007", "_run_matmul_test"): {
            "param_sets": {
                "zeros_b_1x64x1_1x1x1_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    torch.zeros(1,  1, 1, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "zeros_b_1x64x1_1x1x1_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    torch.zeros(1,  1, 1, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_008  all-ones  [1,64,1] @ [1,1,64] → all-ones (inner dim=1)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_008", "_run_matmul_test"): {
            "param_sets": {
                "ones_1x64x1_1x1x64_eager": (
                    torch.ones(1, 64,  1, dtype=torch.float32),
                    torch.ones(1,  1, 64, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "ones_1x64x1_1x1x64_compiled": (
                    torch.ones(1, 64,  1, dtype=torch.float32),
                    torch.ones(1,  1, 64, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_009  all-ones  [1,64,1] @ [1,1,1] → 1.0
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_009", "_run_matmul_test"): {
            "param_sets": {
                "ones_1x64x1_1x1x1_eager": (
                    torch.ones(1, 64, 1, dtype=torch.float32),
                    torch.ones(1,  1, 1, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    False,
                ),
                "ones_1x64x1_1x1x1_compiled": (
                    torch.ones(1, 64, 1, dtype=torch.float32),
                    torch.ones(1,  1, 1, dtype=torch.float32),
                    lambda a, b: torch.matmul(a, b),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_010  matmul → add  [1,64,1] @ [1,1,64] + bias  (prefill)
        # Simulates attention score computation with bias addition.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_010", "_run_matmul_test"): {
            "param_sets": {
                "add_bias_1x64x1_1x1x64_eager": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b) + torch.ones(
                        1, 64, 64, dtype=torch.float32, device=a.device),
                    False,
                ),
                "add_bias_1x64x1_1x1x64_compiled": (
                    make_strided_tensor((1, 64,  1), (64,  1, 1), torch.float32),
                    make_strided_tensor((1,  1, 64), (64, 64, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b) + torch.ones(
                        1, 64, 64, dtype=torch.float32, device=a.device),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_011  matmul → add  [1,64,1] @ [1,1,1] + bias  (decode)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_011", "_run_matmul_test"): {
            "param_sets": {
                "add_bias_1x64x1_1x1x1_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b) + torch.ones(
                        1, 64, 1, dtype=torch.float32, device=a.device),
                    False,
                ),
                "add_bias_1x64x1_1x1x1_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    make_strided_tensor((1,  1, 1), (1,  1, 1), torch.float32),
                    lambda a, b: torch.matmul(a, b) + torch.ones(
                        1, 64, 1, dtype=torch.float32, device=a.device),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_012  special values  +inf in A → inf/nan in output  (prefill)
        # CPU-only structural check.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_012", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_inf_a_1x64x1_1x1x64": (
                    torch.full((1, 64,  1), float("inf"), dtype=torch.float32),
                    torch.ones((1,  1, 64),               dtype=torch.float32),
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_013  special values  NaN in A poisons all output  (decode)
        # CPU-only structural check.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_013", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_nan_a_1x64x1_1x1x1": (
                    torch.full((1, 64, 1), float("nan"), dtype=torch.float32),
                    torch.ones((1,  1, 1),               dtype=torch.float32),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_matmul_test(self, a, b, op, compiled):
        """torch.matmul / a.matmul(b) / a @ b — batched matrix multiply."""
        compare_with_cpu(op, a, b, compiled=compiled)

    def _run_matmul_special_values_test(self, a, b):
        """CPU-only: inf/NaN in input must propagate to output."""
        result = torch.matmul(a.cpu(), b.cpu())
        if torch.isinf(a).any() or torch.isinf(b).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in output when input contains inf, got: {result}"
            )
        if torch.isnan(a).any() or torch.isnan(b).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in output when input contains NaN, got: {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestMean
# ─────────────────────────────────────────────────────────────────────────────

class TestMean(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mean patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    torch.mean appears in RMSNorm: mean(x ** 2, dim=-1, keepdim=True).

      prefill    [1, 64, 4096] → [1, 64, 1]   float32
      decode     [1,  1, 4096] → [1,  1, 1]   float32

    Variants covered per shape:
      global mean (no dim), dim=-1, dim=-1 keepdim, dim=0, dim=1,
      method alias t.mean(), cast to float16, zeros/ones inputs,
      CPU-only NaN/inf special value checks.
    """

    pytestmark = pytest.mark.torch_mean

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  global mean (no dim)  [1,64,4096]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_000", "_run_mean_test"): {
            "param_sets": {
                "1x64x4096_global_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t),
                    False,
                ),
                "1x64x4096_global_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_001  global mean (no dim)  [1,1,4096]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_001", "_run_mean_test"): {
            "param_sets": {
                "1x1x4096_global_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t),
                    False,
                ),
                "1x1x4096_global_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_002  dim=-1  [1,64,4096] → [1,64]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_002", "_run_mean_test"): {
            "param_sets": {
                "1x64x4096_dim-1_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1),
                    False,
                ),
                "1x64x4096_dim-1_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_003  dim=-1  [1,1,4096] → [1,1]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_003", "_run_mean_test"): {
            "param_sets": {
                "1x1x4096_dim-1_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1),
                    False,
                ),
                "1x1x4096_dim-1_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_004  dim=-1 keepdim=True  [1,64,4096] → [1,64,1]
        # This is the exact RMSNorm call-site shape.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_004", "_run_mean_test"): {
            "param_sets": {
                "1x64x4096_keepdim_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "1x64x4096_keepdim_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_005  dim=-1 keepdim=True  [1,1,4096] → [1,1,1]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_005", "_run_mean_test"): {
            "param_sets": {
                "1x1x4096_keepdim_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "1x1x4096_keepdim_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_006  dim=1 (seq reduction)  [1,64,4096] → [1,4096]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_006", "_run_mean_test"): {
            "param_sets": {
                "1x64x4096_dim1_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=1),
                    False,
                ),
                "1x64x4096_dim1_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=1),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_007  dim=0 (batch reduction)  [1,1,4096] → [1,4096]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_007", "_run_mean_test"): {
            "param_sets": {
                "1x1x4096_dim0_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=0),
                    False,
                ),
                "1x1x4096_dim0_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=0),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_008  method alias  [1,64,4096]  t.mean(dim=-1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_008", "_run_mean_test"): {
            "param_sets": {
                "method_1x64x4096_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: t.mean(dim=-1),
                    False,
                ),
                "method_1x64x4096_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: t.mean(dim=-1),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_009  method alias  [1,1,4096]  t.mean(dim=-1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_009", "_run_mean_test"): {
            "param_sets": {
                "method_1x1x4096_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: t.mean(dim=-1),
                    False,
                ),
                "method_1x1x4096_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: t.mean(dim=-1),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_010  all-zeros  [1,64,4096]  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_010", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x64x4096_eager": (
                    torch.zeros(1, 64, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "zeros_1x64x4096_compiled": (
                    torch.zeros(1, 64, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_011  all-zeros  [1,1,4096]  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_011", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x1x4096_eager": (
                    torch.zeros(1, 1, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "zeros_1x1x4096_compiled": (
                    torch.zeros(1, 1, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_012  all-ones  [1,64,4096]  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_012", "_run_mean_test"): {
            "param_sets": {
                "ones_1x64x4096_eager": (
                    torch.ones(1, 64, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "ones_1x64x4096_compiled": (
                    torch.ones(1, 64, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_013  all-ones  [1,1,4096]  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_013", "_run_mean_test"): {
            "param_sets": {
                "ones_1x1x4096_eager": (
                    torch.ones(1, 1, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    False,
                ),
                "ones_1x1x4096_compiled": (
                    torch.ones(1, 1, 4096, dtype=torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_014  mean → cast to float16  [1,64,4096]
        # Llama uses float16 throughout; simulates mixed-precision downcast.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_014", "_run_mean_test"): {
            "param_sets": {
                "cast_bf16_1x64x4096_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True).to(torch.float16),
                    False,
                ),
                "cast_bf16_1x64x4096_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True).to(torch.float16),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_015  mean → cast to float16  [1,1,4096]
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_015", "_run_mean_test"): {
            "param_sets": {
                "cast_bf16_1x1x4096_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True).to(torch.float16),
                    False,
                ),
                "cast_bf16_1x1x4096_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    lambda t: torch.mean(t, dim=-1, keepdim=True).to(torch.float16),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_016  special values  NaN in [1,64,4096] poisons mean
        # CPU-only structural check.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_016", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_nan_1x64x4096": (
                    torch.cat([
                        torch.tensor([float("nan")], dtype=torch.float32),
                        torch.ones(64 * 4096 - 1,    dtype=torch.float32),
                    ]).reshape(1, 64, 4096),
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_017  special values  +inf in [1,1,4096] → inf/nan mean
        # CPU-only structural check.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_017", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_inf_1x1x4096": (
                    torch.cat([
                        torch.tensor([float("inf")], dtype=torch.float32),
                        torch.ones(4096 - 1,          dtype=torch.float32),
                    ]).reshape(1, 1, 4096),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_mean_test(self, tensor, op, compiled):
        """torch.mean with various dim/keepdim/cast variants."""
        compare_with_cpu(op, tensor, compiled=compiled)

    def _run_mean_special_values_test(self, tensor):
        """CPU-only: NaN/inf in input must propagate correctly to mean output."""
        result = torch.mean(tensor.cpu(), dim=-1)
        if torch.isnan(tensor).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in mean output when input contains NaN, got: {result}"
            )
        if torch.isinf(tensor).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in mean output when input contains inf, got: {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestNeg
# ─────────────────────────────────────────────────────────────────────────────

class TestNeg(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.neg patterns observed in Llama-3.1-8B-Instruct.

    Shapes, strides, and dtypes are sourced from llama_ops_strides.txt.
    torch.neg appears in RoPE: negating the second half of the rotary
    embedding before recombining with cos/sin. All inputs are float16
    and non-contiguous (interleaved real/imag RoPE layout).

    Four model shapes:
      q-head prefill    [1, 32, 64, 64]
      k/v-head prefill  [1,  8, 64, 64]
      q-head decode     [1, 32,  1, 64]
      k/v-head decode   [1,  8,  1, 64]

    Variants covered per shape:
      torch.neg(x), x.neg(), -x operator,
      zeros/ones inputs, CPU-only sign-flip structural check.
    """

    pytestmark = pytest.mark.torch_neg

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  torch.neg  q-head prefill  [1,32,64,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_000", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x64x64_bf16_prefill_eager": (
                    make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "neg_1x32x64x64_bf16_prefill_compiled": (
                    make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_001  torch.neg  k/v-head prefill  [1,8,64,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_001", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x64x64_bf16_prefill_eager": (
                    make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "neg_1x8x64x64_bf16_prefill_compiled": (
                    make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_002  torch.neg  q-head decode  [1,32,1,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_002", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x1x64_bf16_decode_eager": (
                    make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "neg_1x32x1x64_bf16_decode_compiled": (
                    make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_003  torch.neg  k/v-head decode  [1,8,1,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_003", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x1x64_bf16_decode_eager": (
                    make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "neg_1x8x1x64_bf16_decode_compiled": (
                    make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_004  method alias  x.neg()  q-head prefill  [1,32,64,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_004", "_run_neg_test"): {
            "param_sets": {
                "method_1x32x64x64_bf16_eager": (
                    make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    lambda x: x.neg(),
                    False,
                ),
                "method_1x32x64x64_bf16_compiled": (
                    make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                    lambda x: x.neg(),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_005  - operator  -x  k/v-head prefill  [1,8,64,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_005", "_run_neg_test"): {
            "param_sets": {
                "unary_minus_1x8x64x64_bf16_eager": (
                    make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    lambda x: -x,
                    False,
                ),
                "unary_minus_1x8x64x64_bf16_compiled": (
                    make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                    lambda x: -x,
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_006  all-zeros  neg(zeros) must be zeros  [1,32,64,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_006", "_run_neg_test"): {
            "param_sets": {
                "zeros_1x32x64x64_bf16_eager": (
                    torch.zeros(1, 32, 64, 64, dtype=torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "zeros_1x32x64x64_bf16_compiled": (
                    torch.zeros(1, 32, 64, 64, dtype=torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_007  all-ones  neg(ones) must be all -1  [1,8,1,64]
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_007", "_run_neg_test"): {
            "param_sets": {
                "ones_1x8x1x64_bf16_eager": (
                    torch.ones(1, 8, 1, 64, dtype=torch.float16),
                    lambda x: torch.neg(x),
                    False,
                ),
                "ones_1x8x1x64_bf16_compiled": (
                    torch.ones(1, 8, 1, 64, dtype=torch.float16),
                    lambda x: torch.neg(x),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # pattern_008  CPU-only sign-flip check  neg(neg(x)) == x
        # Structural: double negation must be identity for all four shapes.
        # ------------------------------------------------------------------
        ("test_torch_neg_pattern_008", "_run_neg_double_negation_test"): {
            "param_sets": {
                "double_neg_1x32x64x64": (
                    make_strided_tensor((1, 32, 64, 64), (262144, 128, 4096, 1), torch.float16),
                ),
                "double_neg_1x8x64x64": (
                    make_strided_tensor((1, 8, 64, 64), (65536, 128, 1024, 1), torch.float16),
                ),
                "double_neg_1x32x1x64": (
                    make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), torch.float16),
                ),
                "double_neg_1x8x1x64": (
                    make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), torch.float16),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_neg_test(self, a, op, compiled):
        """torch.neg / x.neg() / -x — elementwise negation."""
        compare_with_cpu(op, a, compiled=compiled)

    def _run_neg_double_negation_test(self, a):
        """CPU-only: neg(neg(x)) must equal x (identity under double negation)."""
        result = torch.neg(torch.neg(a.cpu()))
        torch.testing.assert_close(result, a.cpu(), msg="double negation is not identity")


# ─────────────────────────────────────────────────────────────────────────────
# TestGetitem
# ─────────────────────────────────────────────────────────────────────────────

class TestGetitem(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.__getitem__ across Llama-3.1-8B-Instruct shapes.


    Index shapes  (int64) : [64], [1, 64], [1, 1]
    Data  shapes  (float16): [1, 32, 64, 128], [1, 8, 64, 128], [1, 8, 2048, 128]
                             [1, 64, 4096],     [1, 32, 1, 128], [1, 8, 1, 128]
                             [1, 1, 4096]
    Data  shapes  (float32): [64]


    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_getitem_shape_test  → @pytest.mark.torch_getitem_shape
        _run_getitem_values_test → @pytest.mark.torch_getitem_values
        _run_getitem_dtype_test  → @pytest.mark.torch_getitem_dtype


    Param tuple layout
    ------------------
    _run_getitem_shape_test  : (x, idx, expected_shape, compiled)
    _run_getitem_values_test : (x, idx, compiled)
    _run_getitem_dtype_test  : (x, idx, compiled)


    idx is a plain Python int, slice, or tuple of slices — not a Tensor.
    compare_with_cpu passes it through unchanged (only Tensor args are moved
    to the target device).
    """


    pytestmark = pytest.mark.torch_getitem


    torch.manual_seed(0)


    PARAMS = {


        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — output shape is correct after indexing
        # ══════════════════════════════════════════════════════════════════


        # [64] float32  →  [:32]  →  [32]
        ("test_torch_getitem_shape_pattern_000", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_64_sl32_eager":    ( make_strided_tensor((64,), (1,), torch.float32), S(None, 32), [32], False),
                "s_64_sl32_compiled": ( make_strided_tensor((64,), (1,), torch.float32), S(None, 32), [32], True),
            },
        },
        # [1, 64] int64  →  [0]  →  [64]
        ("test_torch_getitem_shape_pattern_001", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x64_idx0_eager":    ( make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, [64], False),
                "s_1x64_idx0_compiled": ( make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, [64], True),
            },
        },
        # [1, 32, 64, 128] float16  →  [:, :, :1, :]  →  [1, 32, 1, 128]  (decode slice)
        ("test_torch_getitem_shape_pattern_002", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x64x128_d2sl1_eager":    (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), [1, 32, 1, 128], False),
                "s_1x32x64x128_d2sl1_compiled": (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), [1, 32, 1, 128], True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :1, :]  →  [1, 8, 1, 128]
        ("test_torch_getitem_shape_pattern_003", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x64x128_d2sl1_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), [1, 8, 1, 128], False),
                "s_1x8x64x128_d2sl1_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), [1, 8, 1, 128], True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :64, :]  →  [1, 8, 64, 128]  (KV-cache prefill)
        ("test_torch_getitem_shape_pattern_004", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x64x128_slfull_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), [1, 8, 64, 128], False),
                "s_1x8x64x128_slfull_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), [1, 8, 64, 128], True),
            },
        },
        # [1, 64, 4096] float16  →  [:, 0, :]  →  [1, 4096]  (first token hidden state)
        ("test_torch_getitem_shape_pattern_005", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x64x4096_d1idx0_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), [1, 4096], False),
                "s_1x64x4096_d1idx0_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), [1, 4096], True),
            },
        },
        # [1, 1] int64  →  [0]  →  [1]
        ("test_torch_getitem_shape_pattern_006", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1_idx0_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, [1], False),
                "s_1x1_idx0_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, [1], True),
            },
        },
        # [1, 32, 1, 128] float16  →  [0]  →  [32, 1, 128]
        ("test_torch_getitem_shape_pattern_007", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_idx0_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, [32, 1, 128], False),
                "s_1x32x1x128_idx0_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, [32, 1, 128], True),
            },
        },
        # [1, 8, 1, 128] float16  →  [0]  →  [8, 1, 128]
        ("test_torch_getitem_shape_pattern_008", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x1x128_idx0_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, [8, 1, 128], False),
                "s_1x8x1x128_idx0_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, [8, 1, 128], True),
            },
        },
        # [1, 1, 4096] float16  →  [0]  →  [1, 4096]
        ("test_torch_getitem_shape_pattern_009", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1x4096_idx0_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, [1, 4096], False),
                "s_1x1x4096_idx0_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, [1, 4096], True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — indexed values must match CPU reference
        # ══════════════════════════════════════════════════════════════════


        # [64] float32  →  [:32]  first half
        ("test_torch_getitem_values_pattern_000", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32_eager":    ( make_strided_tensor((64,), (1,), torch.float32), S(None, 32), False),
                "v_64_sl32_compiled": ( make_strided_tensor((64,), (1,), torch.float32), S(None, 32), True),
            },
        },
        # [64] float32  →  [32:]  second half
        ("test_torch_getitem_values_pattern_001", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32end_eager":    (make_strided_tensor((64,), (1,), torch.float32), S(32, None), False),
                "v_64_sl32end_compiled": (make_strided_tensor((64,), (1,), torch.float32), S(32, None), True),
            },
        },
        # [1, 64] int64  →  [0]
        ("test_torch_getitem_values_pattern_002", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x64_idx0_eager":    ( make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "v_1x64_idx0_compiled": ( make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 64, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_values_pattern_003", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x64x128_d2sl1_eager":    (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), False),
                "v_1x32x64x128_d2sl1_compiled": (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :64, :]  (KV-cache prefill slice)
        ("test_torch_getitem_values_pattern_004", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x64x128_slfull_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), False),
                "v_1x8x64x128_slfull_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :1, :]  (KV-cache decode slice)
        ("test_torch_getitem_values_pattern_005", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x64x128_sl1_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), False),
                "v_1x8x64x128_sl1_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 64, 4096] float16  →  [:, 0, :]  (first token)
        ("test_torch_getitem_values_pattern_006", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x64x4096_d1idx0_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), False),
                "v_1x64x4096_d1idx0_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_values_pattern_007", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1_idx0_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "v_1x1_idx0_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 1, 128] float16  →  [:, :, 0, :]  →  [1, 32, 128]
        ("test_torch_getitem_values_pattern_008", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d2idx0_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), (_, _, 0, _), False),
                "v_1x32x1x128_d2idx0_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), (_, _, 0, _), True),
            },
        },
        # [1, 8, 1, 128] float16  →  [:, :, 0, :]  →  [1, 8, 128]
        ("test_torch_getitem_values_pattern_009", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x1x128_d2idx0_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), (_, _, 0, _), False),
                "v_1x8x1x128_d2idx0_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), (_, _, 0, _), True),
            },
        },
        # [1, 1, 4096] float16  →  [0]  →  [1, 4096]
        ("test_torch_getitem_values_pattern_010", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1x4096_idx0_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, False),
                "v_1x1x4096_idx0_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — dtype must not change after indexing
        # ══════════════════════════════════════════════════════════════════


        # [64] float32  →  [:32]
        ("test_torch_getitem_dtype_pattern_000", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_64_eager":    (make_strided_tensor((64,), (1,), torch.float32), S(None, 32), False),
                "dtype_64_compiled": (make_strided_tensor((64,), (1,), torch.float32), S(None, 32), True),
            },
        },
        # [1, 64] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_001", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x64_eager":    (make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "dtype_1x64_compiled": (make_strided_tensor((1, 64), (64, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 64, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_002", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x64x128_eager":    (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), False),
                "dtype_1x32x64x128_compiled": (make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_003", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x64x128_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), False),
                "dtype_1x8x64x128_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 64, 128] float16  →  [:, :, :64, :]
        ("test_torch_getitem_dtype_pattern_004", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x64x128_slfull_eager":    (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), False),
                "dtype_1x8x64x128_slfull_compiled": (make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16), (_, _, S(None, 64), _), True),
            },
        },
        # [1, 64, 4096] float16  →  [:, 0, :]
        ("test_torch_getitem_dtype_pattern_005", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x64x4096_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), False),
                "dtype_1x64x4096_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), (_, 0, _), True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_006", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1_eager":    (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, False),
                "dtype_1x1_compiled": (make_strided_tensor((1, 1), (1, 1), torch.int64, min_val=0, max_val=1000), 0, True),
            },
        },
        # [1, 32, 1, 128] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_007", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_eager":    (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, False),
                "dtype_1x32x1x128_compiled": (make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16), 0, True),
            },
        },
        # [1, 8, 1, 128] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_008", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x1x128_eager":    (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, False),
                "dtype_1x8x1x128_compiled": (make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16), 0, True),
            },
        },
        # [1, 1, 4096] float16  →  [0]
        ("test_torch_getitem_dtype_pattern_009", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, False),
                "dtype_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), 0, True),
            },
        },
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # ── Base test methods ──────────────────────────────────────────────────


    def _run_getitem_shape_test(self, x, idx, expected_shape, compiled):
        def fn(t):
            out = t[idx]
            assert list(out.shape) == expected_shape, (
                f"Shape mismatch: expected {expected_shape}, got {list(out.shape)}"
            )
            return out


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_getitem_values_test(self, x, idx, compiled):
        compare_with_cpu(lambda t: t[idx], x, compiled=compiled)


    def _run_getitem_dtype_test(self, x, idx, compiled):
        def fn(t):
            result = t[idx]
            assert result.dtype == t.dtype, (
                f"dtype changed after getitem: expected {t.dtype}, got {result.dtype}"
            )
            return result


        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
#  TestCumsum  —  torch.cumsum
# ─────────────────────────────────────────────────────────────────────────────

class TestCumsum(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_cumsum
    torch.manual_seed(0)

    PARAMS = {
        ("test_torch_cumsum_bool", "_run_cumsum_test"): {
            "param_sets": {
                "prefill_zeros_eager":    (make_strided_tensor((1, 64), (64, 1), torch.bool, fill="zeros"), -1, False),
                "prefill_zeros_compiled": (make_strided_tensor((1, 64), (64, 1), torch.bool, fill="zeros"), -1, True),
                "prefill_ones_eager":     (make_strided_tensor((1, 64), (64, 1), torch.bool, fill="ones"), -1, False),
                "prefill_ones_compiled":  (make_strided_tensor((1, 64), (64, 1), torch.bool, fill="ones"), -1, True),
                "prefill_mixed_eager": (
                    torch.tensor([[False]*10 + [True] + [False]*20 + [True] + [False]*32], dtype=torch.bool),
                    -1, False,
                ),
            }
        },

        ("test_torch_cumsum_int64", "_run_cumsum_test"): {
            "param_sets": {
                "int64_prefill_eager":    (make_strided_tensor((1, 64), (64, 1), I64, fill="arange"), -1, False),
                "int64_prefill_compiled": (make_strided_tensor((1, 64), (64, 1), I64, fill="arange"), -1, True),
            }
        },

    }

    def _run_cumsum_test(self, x, dim, compiled):
        compare_with_cpu(lambda t: torch.cumsum(t, dim), x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestTranspose
# ─────────────────────────────────────────────────────────────────────────────

class TestTranspose(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.transpose across all Llama-3.1-8B-Instruct shapes.


    Input shapes:
        3-D : [1, 64, 1]  (float32),  [1, 64, 64]  (float32)
        4-D : [1,  1, 32, 128] (float16), [1,  1,  8, 128] (float16)
              [1, 32,  1, 128] (float16), [1, 64, 32, 128] (float16)
              [1, 64,  8, 128] (float16), [1, 32, 64, 128] (float16)


    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_transpose_shape_test       → @pytest.mark.torch_transpose_shape
        _run_transpose_values_test      → @pytest.mark.torch_transpose_values
        _run_transpose_neg_dims_test    → @pytest.mark.torch_transpose_neg_dims
        _run_transpose_contiguity_test  → @pytest.mark.torch_transpose_contiguity
        _run_transpose_contig_copy_test → @pytest.mark.torch_transpose_contig_copy
        _run_transpose_dtype_test       → @pytest.mark.torch_transpose_dtype
    """


    pytestmark = pytest.mark.torch_transpose


    torch.manual_seed(0)


    PARAMS = {


        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════


        # 3-D [1, 64, 1] float32 ──────────────────────────────────────────
        # (0,1): [1,64,1] → [64,1,1]
        ("test_torch_transpose_shape_pattern_000", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d01_eager":    (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), False),
                "s_1x64x1_d01_compiled": (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), True),
            },
        },
        # (1,2): [1,64,1] → [1,1,64]
        ("test_torch_transpose_shape_pattern_001", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d12_eager":    (1, 2, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), False),
                "s_1x64x1_d12_compiled": (1, 2, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), True),
            },
        },


        # 3-D [1, 64, 64] float32 ─────────────────────────────────────────
        # (0,1): [1,64,64] → [64,1,64]
        ("test_torch_transpose_shape_pattern_002", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x64_d01_eager":    (0, 1, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), False),
                "s_1x64x64_d01_compiled": (0, 1, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), True),
            },
        },
        # (1,2): [1,64,64] → [1,64,64]
        ("test_torch_transpose_shape_pattern_003", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x64_d12_eager":    (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), False),
                "s_1x64x64_d12_compiled": (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), True),
            },
        },


        # 4-D [1, 1, 32, 128] float16 ─────────────────────────────────────
        # (1,2): [1,1,32,128] → [1,32,1,128]
        ("test_torch_transpose_shape_pattern_004", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d12_eager":    (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "s_1x1x32x128_d12_compiled": (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,1,32,128] → [1,1,128,32]
        ("test_torch_transpose_shape_pattern_005", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "s_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },


        # 4-D [1, 1, 8, 128] float16 ──────────────────────────────────────
        # (1,3): [1,1,8,128] → [1,128,8,1]
        ("test_torch_transpose_shape_pattern_006", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "s_1x1x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,1,8,128] → [1,1,128,8]
        ("test_torch_transpose_shape_pattern_007", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "s_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },


        # 4-D [1, 32, 1, 128] float16 ─────────────────────────────────────
        # (1,3): [1,32,1,128] → [1,128,1,32]
        ("test_torch_transpose_shape_pattern_008", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,32,1,128] → [1,32,128,1]
        ("test_torch_transpose_shape_pattern_009", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x32x1x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },


        # 4-D [1, 64, 32, 128] float16 ────────────────────────────────────
        # (1,2): [1,64,32,128] → [1,32,64,128]
        ("test_torch_transpose_shape_pattern_010", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x32x128_d12_eager":    (1, 2, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), False),
                "s_1x64x32x128_d12_compiled": (1, 2, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,64,32,128] → [1,64,128,32]
        ("test_torch_transpose_shape_pattern_011", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), False),
                "s_1x64x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), True),
            },
        },


        # 4-D [1, 64, 8, 128] float16 ─────────────────────────────────────
        # (1,2): [1,64,8,128] → [1,8,64,128]
        ("test_torch_transpose_shape_pattern_012", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x8x128_d12_eager":    (1, 2, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), False),
                "s_1x64x8x128_d12_compiled": (1, 2, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,64,8,128] → [1,64,128,8]
        ("test_torch_transpose_shape_pattern_013", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), False),
                "s_1x64x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), True),
            },
        },


        # 4-D [1, 32, 64, 128] float16 ────────────────────────────────────
        # (1,2): [1,32,64,128] → [1,64,32,128]
        ("test_torch_transpose_shape_pattern_014", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x64x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "s_1x32x64x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },
        # (2,3): [1,32,64,128] → [1,32,128,64]
        ("test_torch_transpose_shape_pattern_015", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x64x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "s_1x32x64x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════


        # [1,64,64] float32  (1,2): t[b,r,c] == result[b,c,r]
        ("test_torch_transpose_values_pattern_000", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x64x64_d12_eager":    (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), False),
                "v_1x64x64_d12_compiled": (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), True),
            },
        },
        # [1,1,32,128] float16  (2,3): t[b,h,s,d] == result[b,h,d,s]
        ("test_torch_transpose_values_pattern_001", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "v_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128] float16  (2,3)
        ("test_torch_transpose_values_pattern_002", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "v_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128] float16  (1,3): t[b,h,s,d] == result[b,d,s,h]
        ("test_torch_transpose_values_pattern_003", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "v_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,64,32,128] float16  (2,3)
        ("test_torch_transpose_values_pattern_004", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x64x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), False),
                "v_1x64x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,64,8,128] float16  (1,3)
        ("test_torch_transpose_values_pattern_005", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x64x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), False),
                "v_1x64x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,64,128] float16  (1,2)
        ("test_torch_transpose_values_pattern_006", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x64x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "v_1x32x64x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },
        # [1,32,64,128] float16  (2,3)
        ("test_torch_transpose_values_pattern_007", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x64x128_d23_eager":    (2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "v_1x32x64x128_d23_compiled": (2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # NEGATIVE DIMENSION INDEXING
        # ══════════════════════════════════════════════════════════════════


        # [1,64,64] float32  (-2,-1) == (1,2)
        ("test_torch_transpose_neg_dims_pattern_000", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x64_m2m1_vs_12_eager":    (-2, -1, 1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), False),
                "neg_1x64x64_m2m1_vs_12_compiled": (-2, -1, 1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), True),
            },
        },
        # [1,1,32,128] float16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_001", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "neg_1x1x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128] float16  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_002", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x8x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "neg_1x1x8x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128] float16  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_003", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x1x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "neg_1x32x1x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,64,32,128] float16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_004", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), False),
                "neg_1x64x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,64,8,128] float16  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_005", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x8x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), False),
                "neg_1x64x8x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,64,128] float16  (-4,-1) == (0,3)
        ("test_torch_transpose_neg_dims_pattern_006", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x64x128_m4m1_vs_03_eager":    (-4, -1, 0, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "neg_1x32x64x128_m4m1_vs_03_compiled": (-4, -1, 0, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },
        # [1,32,64,128] float16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_007", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x64x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "neg_1x32x64x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION
        # ══════════════════════════════════════════════════════════════════


        # [1,64,1] float32  (0,1)
        ("test_torch_transpose_dtype_pattern_000", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x1_d01_eager":    (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), False),
                "dtype_1x64x1_d01_compiled": (0, 1, make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32), True),
            },
        },
        # [1,1,32,128] float16  (2,3)
        ("test_torch_transpose_dtype_pattern_001", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), False),
                "dtype_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,1,8,128] float16  (2,3)
        ("test_torch_transpose_dtype_pattern_002", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x8x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), False),
                "dtype_1x1x8x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,1,128] float16  (1,3)
        ("test_torch_transpose_dtype_pattern_003", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_d13_eager":    (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), False),
                "dtype_1x32x1x128_d13_compiled": (1, 3, make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1,64,64] float32  (1,2)
        ("test_torch_transpose_dtype_pattern_004", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x64_d12_eager":    (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), False),
                "dtype_1x64x64_d12_compiled": (1, 2, make_strided_tensor((1, 64, 64), (4096, 64, 1), torch.float32), True),
            },
        },
        # [1,64,32,128] float16  (2,3)
        ("test_torch_transpose_dtype_pattern_005", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), False),
                "dtype_1x64x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 4096, 128, 1), torch.float16), True),
            },
        },
        # [1,64,8,128] float16  (1,3)
        ("test_torch_transpose_dtype_pattern_006", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x8x128_d13_eager":    (1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), False),
                "dtype_1x64x8x128_d13_compiled": (1, 3, make_strided_tensor((1, 64, 8, 128), (65536, 1024, 128, 1), torch.float16), True),
            },
        },
        # [1,32,64,128] float16  (1,2)
        ("test_torch_transpose_dtype_pattern_007", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x64x128_d12_eager":    (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), False),
                "dtype_1x32x64x128_d12_compiled": (1, 2, make_strided_tensor((1, 32, 64, 128), (262144, 8192, 128, 1), torch.float16), True),
            },
        },
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # ── Base test methods ──────────────────────────────────────────────────


    def _run_transpose_shape_test(self, dim0, dim1, x, compiled):
        expected = list(x.shape)
        d0, d1 = dim0 % x.ndim, dim1 % x.ndim
        expected[d0], expected[d1] = expected[d1], expected[d0]


        def fn(t):
            out = torch.transpose(t, dim0, dim1).contiguous()
            assert list(out.shape) == expected, (
                f"Shape mismatch: expected {expected}, got {list(out.shape)}"
            )
            return out


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_transpose_values_test(self, dim0, dim1, x, compiled):
        compare_with_cpu(
            lambda t: torch.transpose(t, dim0, dim1).contiguous(),
            x,
            compiled=compiled,
        )


    def _run_transpose_neg_dims_test(self, neg0, neg1, pos0, pos1, x, compiled):
        def fn(t):
            neg_result = torch.transpose(t, neg0, neg1).contiguous()
            pos_result = torch.transpose(t, pos0, pos1).contiguous()
            torch.testing.assert_close(
                neg_result, pos_result,
                msg=f"transpose({neg0},{neg1}) differs from transpose({pos0},{pos1})",
            )
            return neg_result


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_transpose_dtype_test(self, dim0, dim1, x, compiled):
        def fn(t):
            result = torch.transpose(t, dim0, dim1).contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after transpose({dim0},{dim1}): "
                f"expected {t.dtype}, got {result.dtype}"
            )
            return result


        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestCos
# ─────────────────────────────────────────────────────────────────────────────

class TestCos(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cos across Llama-3.1-8B-Instruct shapes.


    Op specification:
        name  : torch.cos.2_spyre
        op    : torch.cos
        dtype : torch.float32


    Input shapes:
        [1, 64, 128]
        [1,  1, 128]


    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_cos_shape_test  → @pytest.mark.torch_cos_shape
        _run_cos_values_test → @pytest.mark.torch_cos_values
        _run_cos_dtype_test  → @pytest.mark.torch_cos_dtype
    """


    pytestmark = pytest.mark.torch_cos


    torch.manual_seed(0)


    PARAMS = {


        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — cos is pointwise: output shape == input shape
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 128]
        ("test_torch_cos_shape_pattern_000", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x64x128_eager":    (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), False),
                "s_1x64x128_compiled": (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_shape_pattern_001", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x1x128_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), False),
                "s_1x1x128_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — cos(x) CPU vs Spyre element-wise
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 128]  rand input
        ("test_torch_cos_values_pattern_000", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x64x128_rand_eager":    (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), False),
                "v_1x64x128_rand_compiled": (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), True),
            },
        },
        # [1, 64, 128]  zeros input: cos(0) == 1.0 for all elements
        ("test_torch_cos_values_pattern_001", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x64x128_zeros_eager":    (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32, fill="zeros"), False),
                "v_1x64x128_zeros_compiled": (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32, fill="zeros"), True),
            },
        },
        # [1, 1, 128]  rand input
        ("test_torch_cos_values_pattern_002", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_rand_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), False),
                "v_1x1x128_rand_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), True),
            },
        },
        # [1, 1, 128]  zeros input
        ("test_torch_cos_values_pattern_003", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_zeros_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32, fill="zeros"), False),
                "v_1x1x128_zeros_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32, fill="zeros"), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — float32 in must give float32 out
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 128]
        ("test_torch_cos_dtype_pattern_000", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x64x128_eager":    (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), False),
                "dtype_1x64x128_compiled": (make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_dtype_pattern_001", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x1x128_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), False),
                "dtype_1x1x128_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), True),
            },
        },
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # ── Base test methods ──────────────────────────────────────────────────


    def _run_cos_shape_test(self, x, compiled):
        def fn(t):
            out = torch.cos(t)
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_cos_values_test(self, x, compiled):
        compare_with_cpu(torch.cos, x, compiled=compiled)


    def _run_cos_dtype_test(self, x, compiled):
        def fn(t):
            result = torch.cos(t)
            assert result.dtype == t.dtype, (
                f"dtype changed after cos: expected {t.dtype}, got {result.dtype}"
            )
            return result


        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestMul
# ─────────────────────────────────────────────────────────────────────────────

class TestMul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mul patterns observed in Llama-3.1-8B-Instruct.


    Shapes sourced from Llama-3.1-8B-Instruct_spyre.yaml.
    Three call signatures appear in the model:


      binary       torch.mul(tensor, tensor)   — same or broadcast shapes
      scalar_right torch.mul(tensor, scalar)
      scalar_left  torch.mul(scalar, tensor)


    yaml entries covered:
      mul.1   [1,64,128]    * 1.0           scalar_right  attention_scaling
      mul.2   [1,64,4096]   * [1,64,1]      binary        rsqrt normalisation
      mul.3   [4096]        * [1,64,4096]   binary        weight * hidden (broadcast)
      mul.4   [1,32,64,128] * [1,1,64,128]  binary        q * cos (broadcast)
      mul.5   [1,8,64,128]  * [1,1,64,128]  binary        k * cos (broadcast)
      mul.6   [1,64,14336]  * [1,64,14336]  binary        gate * up prefill (elementwise)
      mul.7   [1,1,128]     * 1.0           scalar_right  attention_scaling decode
      mul.8   [1,1,4096]    * [1,1,1]       binary        rsqrt normalisation decode
      mul.9   [4096]        * [1,1,4096]    binary        weight * hidden decode (broadcast)
      mul.10  [1,32,1,128]  * [1,1,1,128]   binary        q * cos decode (broadcast)
      mul.11  [1,8,1,128]   * [1,1,1,128]   binary        k * cos decode (broadcast)
      mul.12  [1,1,14336]   * [1,1,14336]   binary        gate * up decode (elementwise)
    """


    pytestmark = pytest.mark.torch_mul


    torch.manual_seed(0)


    PARAMS = {
        # ------------------------------------------------------------------
        # mul.1  [1,64,128] * scalar 1.0  — attention_scaling prefill
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_000", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x64x128_eager": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), 1.0, False,
                ),
                "scalar_right_1x64x128_compiled": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), torch.float32), 1.0, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.2  [1,64,4096] * [1,64,1]  — rsqrt normalisation prefill
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_001", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x64x4096_1x64x1_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    False,
                ),
                "binary_1x64x4096_1x64x1_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float32),
                    make_strided_tensor((1, 64, 1), (64, 1, 1), torch.float32),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.3  [4096] * [1,64,4096]  — weight * hidden prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_002", "_run_mul_binary"): {
            "param_sets": {
                "binary_4096_1x64x4096_eager": (
                    make_strided_tensor((4096,), (1,), torch.float16),
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16),
                    False,
                ),
                "binary_4096_1x64x4096_compiled": (
                    make_strided_tensor((4096,), (1,), torch.float16),
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.4  [1,32,64,128] * [1,1,64,128]  — q * cos prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_003", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x64x128_1x1x64x128_eager": (
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 64, 128), (8192, 8192, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x64x128_1x1x64x128_compiled": (
                    make_strided_tensor((1, 32, 64, 128), (262144, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 64, 128), (8192, 8192, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.5  [1,8,64,128] * [1,1,64,128]  — k * cos prefill (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_004", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x64x128_1x1x64x128_eager": (
                    make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 64, 128), (8192, 8192, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x64x128_1x1x64x128_compiled": (
                    make_strided_tensor((1, 8, 64, 128), (65536, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 64, 128), (8192, 8192, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.6  [1,64,14336] * [1,64,14336]  — gate * up prefill (elementwise)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_005", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x64x14336_eager": (
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), torch.float16),
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), torch.float16),
                    False,
                ),
                "binary_1x64x14336_compiled": (
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), torch.float16),
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.7  [1,1,128] * scalar 1.0  — attention_scaling decode
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_006", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x1x128_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), 1.0, False,
                ),
                "scalar_right_1x1x128_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), torch.float32), 1.0, True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.8  [1,1,4096] * [1,1,1]  — rsqrt normalisation decode
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_007", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x4096_1x1x1_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float32),
                    False,
                ),
                "binary_1x1x4096_1x1x1_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float32),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), torch.float32),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.9  [4096] * [1,1,4096]  — weight * hidden decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_008", "_run_mul_binary"): {
            "param_sets": {
                "binary_4096_1x1x4096_eager": (
                    make_strided_tensor((4096,), (1,), torch.float16),
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    False,
                ),
                "binary_4096_1x1x4096_compiled": (
                    make_strided_tensor((4096,), (1,), torch.float16),
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.10  [1,32,1,128] * [1,1,1,128]  — q * cos decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_009", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x1x128_1x1x1x128_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x32x1x128_1x1x1x128_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.11  [1,8,1,128] * [1,1,1,128]  — k * cos decode (broadcast)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_010", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x1x128_1x1x1x128_eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    False,
                ),
                "binary_1x8x1x128_1x1x1x128_compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), torch.float16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), torch.float16),
                    True,
                ),
            }
        },
        # ------------------------------------------------------------------
        # mul.12  [1,1,14336] * [1,1,14336]  — gate * up decode (elementwise)
        # ------------------------------------------------------------------
        ("test_torch_mul_pattern_011", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x14336_eager": (
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), torch.float16),
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), torch.float16),
                    False,
                ),
                "binary_1x1x14336_compiled": (
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), torch.float16),
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), torch.float16),
                    True,
                ),
            }
        },
    }


    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------


    def _run_mul_binary(self, a, b, compiled):
        """torch.mul(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.mul, a, b, compiled=compiled)


    def _run_mul_scalar_right(self, a, scalar, compiled):
        """torch.mul(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.mul(x, scalar),
            a,
            compiled=compiled,
        )


    def _run_mul_scalar_left(self, scalar, b, compiled):
        """torch.mul(scalar, tensor) — scalar on left, tensor on right."""
        compare_with_cpu(
            lambda x: torch.mul(scalar, x),
            b,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestView
# ─────────────────────────────────────────────────────────────────────────────

class TestView(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.view patterns observed in Llama-3.1-8B-Instruct.


    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_view_test``
    into one concrete test method per (pattern, param_set) combination.
    The metaclass also stamps each generated method with @pytest.mark.torch_view
    so the entire op can be selected with  pytest -m torch_view.


    ``Tensor.view(*shape)`` returns a new tensor with the same data but a
    different shape.  The -1 dimension is inferred from the total element
    count and the other dimensions.


    Each param_set contains:
      - input_tensor: tensor to reshape
      - target_shape: the desired shape passed to .view()
      - compiled:     whether to run under torch.compile


    Shapes are sourced from the llama 3 model:
      [1, 64, 4096]  -> (1, 64, -1, 128)  — prefill, multi-head reshape
      [1, 64, 1024]  -> (1, 64, -1, 128)  — prefill, grouped-query reshape
      [1, 1, 4096]   -> (1, 1, -1, 128)   — decode, multi-head reshape
      [1, 1, 1024]   -> (1, 1, -1, 128)   — decode, grouped-query reshape
    """


    pytestmark = pytest.mark.torch_view


    torch.manual_seed(0)


    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1, 64, 4096] -> (1, 64, -1, 128)  i.e. (1, 64, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_000", "_run_view_test"): {
            "param_sets": {
                "1x64x4096_to_1x64xNx128_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16),
                    (1, 64, -1, 128),
                    False,
                ),
                "1x64x4096_to_1x64xNx128_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16),
                    (1, 64, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1, 64, 1024] -> (1, 64, -1, 128)  i.e. (1, 64, 8, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_001", "_run_view_test"): {
            "param_sets": {
                "1x64x1024_to_1x64xNx128_eager": (
                    make_strided_tensor((1, 64, 1024), (65536, 1024, 1), torch.float16),
                    (1, 64, -1, 128),
                    False,
                ),
                "1x64x1024_to_1x64xNx128_compiled": (
                    make_strided_tensor((1, 64, 1024), (65536, 1024, 1), torch.float16),
                    (1, 64, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  [1, 1, 4096] -> (1, 1, -1, 128)  i.e. (1, 1, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_002", "_run_view_test"): {
            "param_sets": {
                "1x1x4096_to_1x1xNx128_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    (1, 1, -1, 128),
                    False,
                ),
                "1x1x4096_to_1x1xNx128_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16),
                    (1, 1, -1, 128),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  [1, 1, 1024] -> (1, 1, -1, 128)  i.e. (1, 1, 8, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_003", "_run_view_test"): {
            "param_sets": {
                "1x1x1024_to_1x1xNx128_eager": (
                    make_strided_tensor((1, 1, 1024), (1024, 1024, 1), torch.float16),
                    (1, 1, -1, 128),
                    False,
                ),
                "1x1x1024_to_1x1xNx128_compiled": (
                    make_strided_tensor((1, 1, 1024), (1024, 1024, 1), torch.float16),
                    (1, 1, -1, 128),
                    True,
                ),
            },
        },
    }


    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # Never called directly; the metaclass replaces them with concrete
    # tests, each stamped with @pytest.mark.torch_view.
    # ------------------------------------------------------------------


    def _run_view_test(self, input_tensor, target_shape, compiled):
        """
        Tensor.view(*shape) — returns a new tensor with the same data
        but a different shape.


        Wraps the method call in a function so compare_with_cpu can
        handle device placement.
        """
        def view_fn(x):
            return x.view(*target_shape)


        compare_with_cpu(view_fn, input_tensor, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestContiguous
# ─────────────────────────────────────────────────────────────────────────────

class TestContiguous(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.contiguous across Llama-3.1-8B-Instruct shapes.


    Input shapes (dtype=float16):
        4-D : [1, 64, 32, 128], [1, 1, 32, 128]
        3-D : [1, 64, 4096],    [1, 1, 4096]


    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_contiguous_shape_test     → @pytest.mark.torch_contiguous_shape
        _run_contiguous_values_test    → @pytest.mark.torch_contiguous_values
        _run_contiguous_noncontig_test → @pytest.mark.torch_contiguous_noncontig
        _run_contiguous_dtype_test     → @pytest.mark.torch_contiguous_dtype
    """


    pytestmark = pytest.mark.torch_contiguous


    torch.manual_seed(0)


    PARAMS = {


        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — .contiguous() must not change shape
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 32, 128]
        ("test_torch_contiguous_shape_pattern_000", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x64x32x128_eager":    (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), False),
                "s_1x64x32x128_compiled": (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), True),
            },
        },
        # [1, 64, 4096]
        ("test_torch_contiguous_shape_pattern_001", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x64x4096_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), False),
                "s_1x64x4096_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_shape_pattern_002", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "s_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_shape_pattern_003", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), False),
                "s_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — values are preserved after .contiguous()
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_000", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x64x32x128_eager":    (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), False),
                "v_1x64x32x128_compiled": (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), True),
            },
        },
        # [1, 64, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_001", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x64x4096_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), False),
                "v_1x64x4096_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_002", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "v_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_003", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), False),
                "v_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # NON-CONTIGUOUS INPUT — transpose first to get a non-contig view,
        # then .contiguous() must produce is_contiguous()==True with same values.
        # Only dim pairs where BOTH swapped sizes > 1 guarantee non-contiguity.
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 32, 128]  (1,3) sizes 64↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_001", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x64x32x128_d13_eager":    (1, 3, make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), False),
                "nc_1x64x32x128_d13_compiled": (1, 3, make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), True),
            },
        },
        # [1, 64, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_002", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x64x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), False),
                "nc_1x64x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), True),
            },
        },
        # [1, 64, 4096]  (1,2) sizes 64↔4096 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_003", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x64x4096_d12_eager":    (1, 2, make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), False),
                "nc_1x64x4096_d12_compiled": (1, 2, make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_004", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x1x32x128_d23_eager":    (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "nc_1x1x32x128_d23_compiled": (2, 3, make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096] has no pair with both dims > 1 — copy test only
        ("test_torch_contiguous_noncontig_pattern_005", "_run_contiguous_values_test"): {
            "param_sets": {
                "nc_copy_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), False),
                "nc_copy_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), True),
            },
        },



        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — float16 in must give float16 out
        # ══════════════════════════════════════════════════════════════════


        # [1, 64, 32, 128]
        ("test_torch_contiguous_dtype_pattern_000", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x64x32x128_eager":    (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), False),
                "dtype_1x64x32x128_compiled": (make_strided_tensor((1, 64, 32, 128), (262144, 128, 8192, 1), torch.float16), True),
            },
        },
        # [1, 64, 4096]
        ("test_torch_contiguous_dtype_pattern_001", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x64x4096_eager":    (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), False),
                "dtype_1x64x4096_compiled": (make_strided_tensor((1, 64, 4096), (262144, 4096, 1), torch.float16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_dtype_pattern_002", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_eager":    (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), False),
                "dtype_1x1x32x128_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), torch.float16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_dtype_pattern_003", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x4096_eager":    (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), False),
                "dtype_1x1x4096_compiled": (make_strided_tensor((1, 1, 4096), (4096, 4096, 1), torch.float16), True),
            },
        },
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # ── Base test methods ──────────────────────────────────────────────────


    def _run_contiguous_shape_test(self, x, compiled):
        def fn(t):
            out = t.contiguous()
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_contiguous_values_test(self, x, compiled):
        compare_with_cpu(lambda t: t.contiguous(), x, compiled=compiled)


    def _run_contiguous_noncontig_test(self, dim0, dim1, x, compiled):
        # Verify the transposed view is non-contiguous on CPU before device dispatch
        raw = torch.transpose(x, dim0, dim1)
        assert not raw.is_contiguous(), (
            f"transpose({dim0},{dim1}) on shape {list(x.shape)} should be non-contiguous"
        )


        def fn(t):
            view = torch.transpose(t, dim0, dim1)
            out = view.contiguous()
            assert out.is_contiguous(), (
                f"contiguous() result should be contiguous for shape {list(t.shape)}"
            )
            return out


        compare_with_cpu(fn, x, compiled=compiled)


    def _run_contiguous_dtype_test(self, x, compiled):
        def fn(t):
            result = t.contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after contiguous(): expected {t.dtype}, got {result.dtype}"
            )
            return result


        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
#  TestTorchReshape
# ─────────────────────────────────────────────────────────────────────────────

class TestReshape(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.reshape — Llama-3.1-8B-Instruct.


    Source ops from model:
        - attn_output.reshape(*input_shape, -1).contiguous()  # Line 173
        - Grouped KV reshape: (1, 8, 4, seq_kv, 128) → (1, 32, seq_kv, 128)
        - FFN intermediate reshape: (1, 64, 14336) → (64, -1)
    """


    pytestmark = pytest.mark.torch_reshape
    torch.manual_seed(0)


    PARAMS = {

        # ── GROUP A: model output shapes [B,S,32,128] → [B,S,4096] ──────────
        ("test_torch_reshape_A000", "_run_reshape_test"): {
            "param_sets": {
                "decode_1x1x32x128_eager":    (_t((1,  1, 32, 128), torch.float16, (4096, 128, 128, 1)), (1,  1, -1), False),
                "decode_1x1x32x128_compiled": (_t((1,  1, 32, 128), torch.float16, (4096, 128, 128, 1)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_A001", "_run_reshape_test"): {
            "param_sets": {
                "prefill_1x64x32x128_eager":    (_t((1, 64, 32, 128), torch.float16, (262144, 4096, 128, 1)), (1, 64, -1), False),
                "prefill_1x64x32x128_compiled": (_t((1, 64, 32, 128), torch.float16, (262144, 4096, 128, 1)), (1, 64, -1), True),
            }
        },



        # ── GROUP L: non-contiguous (transpose → reshape, exact model path) ──
        ("test_torch_reshape_L000", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_decode_eager":    (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), False),
                "noncontig_q_decode_compiled": (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_L001", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_prefill64_eager":    (_t((1, 32, 64, 128)), 1, 2, (1, 64, -1), False),
                "noncontig_q_prefill64_compiled": (_t((1, 32, 64, 128)), 1, 2, (1, 64, -1), True),
            }
        },


        # ── GROUP M: .reshape().contiguous() — full model op chain ───────────
        ("test_torch_reshape_M000", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_decode_eager":    (_t((1,  1, 32, 128), torch.float16, (4096, 128, 128, 1)), (1,  1, -1), False),
                "chain_decode_compiled": (_t((1,  1, 32, 128), torch.float16, (4096, 128, 128, 1)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_M001", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_prefill_eager":    (_t((1, 64, 32, 128), torch.float16, (262144, 4096, 128, 1)), (1, 64, -1), False),
                "chain_prefill_compiled": (_t((1, 64, 32, 128), torch.float16, (262144, 4096, 128, 1)), (1, 64, -1), True),
            }
        },


        # ── GROUP S: CPU-only contiguity structural assertion ─────────────────
        ("test_torch_reshape_S000", "_run_reshape_contiguity_test"): {
            "param_sets": {
                "contiguity_decode":  (_t((1,  1, 32, 128), torch.float16, (4096, 128, 128, 1)), (1,  1, -1)),
                "contiguity_prefill": (_t((1, 64, 32, 128), torch.float16, (262144, 4096, 128, 1)), (1, 64, -1)),
            }
        },
    }


    def _run_reshape_test(self, tensor, target_shape, compiled):
        """tensor.reshape(*target_shape) — eager or compiled."""
        compare_with_cpu(lambda t: t.reshape(*target_shape), tensor, compiled=compiled)


    def _run_reshape_contiguous_test(self, tensor, target_shape, compiled):
        """.reshape().contiguous() — full model op chain."""
        compare_with_cpu(
            lambda t: t.reshape(*target_shape).contiguous(),
            tensor, compiled=compiled,
        )


    def _run_reshape_after_transpose_test(self, tensor, d0, d1, target_shape, compiled):
        """tensor.transpose(d0,d1).reshape(*target_shape) — non-contiguous input."""
        compare_with_cpu(
            lambda t: t.transpose(d0, d1).reshape(*target_shape),
            tensor, compiled=compiled,
        )


    def _run_reshape_contiguity_test(self, tensor, target_shape):
        """CPU-only: contiguous input → contiguous output, same numel."""
        t = tensor.cpu()
        assert t.is_contiguous()
        result = t.reshape(*target_shape)
        assert result.is_contiguous(), (
            f"reshape {tuple(t.shape)} → {target_shape} is not contiguous"
        )
        assert result.numel() == t.numel()


# ─────────────────────────────────────────────────────────────────────────────
#  TestFunctionalSilu
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionalSilu(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.nn.functional.silu —
    Llama-3.1-8B-Instruct.
    SiLU is used in every FFN block as the SwiGLU gate:
        F.silu(gate_proj(x)) * up_proj(x)
    intermediate_size: 14336 (24B)
    """


    pytestmark = pytest.mark.torch_nn_functional_silu
    torch.manual_seed(0xCAFE)


    _INTERMEDIATE = INTERMEDIATE_SIZE  # 14336
    _HIDDEN       = HIDDEN_SIZE


    PARAMS = {


        # ── GROUP A: FFN gate decode [B,1,intermediate] ───────────────────────
        ("test_silu_A000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_1x1x14336_eager":    (_t((1, 1, 14336), torch.float16, (14336, 14336, 1)), False),
                "decode_1x1x14336_compiled": (_t((1, 1, 14336), torch.float16, (14336, 14336, 1)), True),
            }
        },



        # ── GROUP B: FFN gate prefill [B,S,intermediate] ─────────────────────
        ("test_silu_B000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x64x14336_eager":    (_t((1, 64, 14336), torch.float16, (917504, 14336, 1)), False),
                "prefill_1x64x14336_compiled": (_t((1, 64, 14336), torch.float16, (917504, 14336, 1)), True),
            }
        },



        # ── GROUP D: SwiGLU product F.silu(gate) * up ────────────────────────
        ("test_silu_D000", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_decode_1x1x14336_eager":    (_t((1, 1, 14336), torch.float16, (14336, 14336, 1)), _t((1, 1, 14336), torch.float16, (14336, 14336, 1)), False),
                "swiglu_decode_1x1x14336_compiled": (_t((1, 1, 14336), torch.float16, (14336, 14336, 1)), _t((1, 1, 14336), torch.float16, (14336, 14336, 1)), True),
            }
        },
        ("test_silu_D001", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_prefill_1x64x14336_eager":    (_t((1, 64, 14336), torch.float16, (917504, 14336, 1)), _t((1, 64, 14336), torch.float16, (917504, 14336, 1)), False),
                "swiglu_prefill_1x64x14336_compiled": (_t((1, 64, 14336), torch.float16, (917504, 14336, 1)), _t((1, 64, 14336), torch.float16, (917504, 14336, 1)), True),
            }
        },


        # ── GROUP F: non-contiguous input (transpose → silu) ─────────────────
        ("test_silu_F000", "_run_silu_noncontig_test"): {
            "param_sets": {
                "noncontig_gate_decode_eager":    (_t((1, 14336, 1)), 0, 1, False),
                "noncontig_gate_decode_compiled": (_t((1, 14336, 1)), 0, 1, True),
            }
        },



        # ── GROUP H: CPU-only numerical identity F.silu(x) == x*sigmoid(x) ───
        ("test_silu_H000", "_run_silu_identity_check_test"): {
            "param_sets": {
                "identity_gate_decode_fp32":  (_t((1, 1, 14336), torch.float16, (14336, 14336, 1)),),
                "identity_gate_prefill_fp32": (_t((1, 64, 14336), torch.float16, (917504, 14336, 1)),),
            }
        },
    }


    def _run_silu_ffn_gate_test(self, tensor, compiled):
        """F.silu(gate) — FFN gate activation, eager or compiled."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t),
            tensor, compiled=compiled,
        )


    def _run_silu_swiglu_test(self, gate, up, compiled):
        """F.silu(gate) * up — full SwiGLU product, eager or compiled."""
        compare_with_cpu(
            lambda g, u: torch.nn.functional.silu(g) * u,
            gate, up, compiled=compiled,
        )


    def _run_silu_noncontig_test(self, tensor, d0, d1, compiled):
        """F.silu on a non-contiguous view produced by transpose — made contiguous for Spyre."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t.transpose(d0, d1).contiguous()),
            tensor, compiled=compiled,
        )


    def _run_silu_special_values_test(self, tensor):
        """CPU-only: IEEE 754 special-value behaviour of F.silu."""
        t      = tensor.cpu().float()
        result = torch.nn.functional.silu(t).float()
        for idx in range(t.numel()):
            raw = t.view(-1)[idx].item()
            got = result.view(-1)[idx].item()
            if math.isnan(raw):
                assert math.isnan(got), f"silu(NaN) should be NaN, got {got}"
            elif raw == float("inf"):
                assert got == float("inf"), f"silu(+inf) should be +inf, got {got}"
            elif raw == float("-inf"):
                assert math.isnan(got), f"silu(-inf) should be NaN (IEEE 754), got {got}"
            elif raw == 0.0:
                assert got == 0.0, f"silu(0) should be 0.0, got {got}"
            else:
                if got == 0.0:
                    pass  # underflow to signed-zero is valid IEEE 754 behaviour
                else:
                    assert (raw >= 0) == (got >= 0), \
                        f"silu({raw}) sign wrong: got {got}"


    def _run_silu_identity_check_test(self, tensor):
        """CPU-only: F.silu(x) == x * sigmoid(x) element-wise (fp32)."""
        t = tensor.cpu().float()
        torch.testing.assert_close(
            torch.nn.functional.silu(t),
            t * torch.sigmoid(t),
            atol=1e-5, rtol=1e-5,
            msg=lambda msg: f"F.silu(x) != x*sigmoid(x) on {tuple(t.shape)}\n\n{msg}\n",
        )


    def test_silu_zero_fixed_point(self):
        """silu(0) == 0 for fp16, bf16, fp32."""
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            assert torch.nn.functional.silu(torch.zeros(1, dtype=dtype)).item() == 0.0


    def test_silu_shape_preserved_model_shapes(self):
        """F.silu must not alter shape for canonical model shapes."""
        for shape in [
            (1, 1, 14336), (1, 64, 14336), (1, 128, 14336),
            (1, 1,  4096), (1,  1,  1024), (1, 64,  4096),
        ]:
            t = _t(shape)
            assert torch.nn.functional.silu(t).shape == t.shape


    def test_silu_swiglu_matches_decomposition(self):
        """F.silu(gate)*up == (gate*sigmoid(gate))*up in fp32."""
        gate = _t((1, 64, 14336), torch.float32)
        up   = _t((1, 64, 14336), torch.float32)
        torch.testing.assert_close(
            torch.nn.functional.silu(gate) * up,
            (gate * torch.sigmoid(gate)) * up,
            atol=1e-5, rtol=1e-5,
        )



# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestEmbedding  — torch.nn.functional.embedding
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbedding(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.nn.functional.embedding.

    Traced shapes:
      prefill: indices [1, 64] int64 stride (64, 1),
               weight  [128256, 4096] float16 stride (4096, 1),
               output  [1, 64, 4096] float16
      decode : indices [1, 1] int64 stride (1, 1),
               weight  [128256, 4096] float16 stride (4096, 1),
               output  [1, 1, 4096] float16

    Source: modeling_llama.py:395 — self.embed_tokens(input_ids)
    """

    pytestmark = pytest.mark.torch_embedding

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_embedding_test"): {
            "param_sets": {
                "bs1_seq64_i64_f16_eager": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, False,
                ),
                "bs1_seq64_i64_f16_compiled": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, True,
                ),
            },
        },

        # ── 2. Exact-trace decode ──────────────────────────────────────────
        ("test_exact_decode", "_run_embedding_test"): {
            "param_sets": {
                "bs1_seq1_i64_f16_eager": (
                    make_strided_tensor((1, 1), (1, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, False,
                ),
                "bs1_seq1_i64_f16_compiled": (
                    make_strided_tensor((1, 1), (1, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, True,
                ),
            },
        },

        # ── 3. Fill sweep ──────────────────────────────────────────────────
        ("test_fill_sweep", "_run_embedding_test"): {
            "param_sets": {
                "bs1_seq64_zeros_eager": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="zeros"),
                    _W, False,
                ),
                "bs1_seq64_zeros_compiled": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="zeros"),
                    _W, True,
                ),
                "bs1_seq64_ones_eager": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="ones"),
                    _W, False,
                ),
                "bs1_seq64_ones_compiled": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64, fill="ones"),
                    _W, True,
                ),
            },
        },

        # ── 4. Index boundary ──────────────────────────────────────────────
        ("test_index_boundary", "_run_embedding_test"): {
            "param_sets": {
                "idx_0_and_vocab_minus1_eager": (
                    torch.cat([torch.zeros(1, 1, dtype=I64), torch.full((1, 1), 128255, dtype=I64)], dim=1),
                    _W, False,
                ),
                "idx_0_and_vocab_minus1_compiled": (
                    torch.cat([torch.zeros(1, 1, dtype=I64), torch.full((1, 1), 128255, dtype=I64)], dim=1),
                    _W, True,
                ),
            },
        },

        # ── 5. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_embedding_strided_vs_contig_test"): {
            "param_sets": {
                "bs1_seq64_ncontig_eager": (
                    make_strided_tensor((1, 64), (128, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, False,
                ),
                "bs1_seq64_ncontig_compiled": (
                    make_strided_tensor((1, 64), (128, 1), dtype=I64, fill="randn", min_val=0, max_val=128256),
                    _W, True,
                ),
            },
        },
    }

    def _run_embedding_test(self, indices, weight, compiled):
        """Embedding: CPU vs Spyre via compare_with_cpu."""
        fn = lambda idx, w: F.embedding(idx, w)
        compare_with_cpu(fn, indices, weight, compiled=compiled)

    def _run_embedding_strided_vs_contig_test(self, indices, weight, compiled):
        """Non-contiguous index tensor must produce the same output as its
        contiguous counterpart — verified on CPU and Spyre."""
        fn = lambda idx, w: F.embedding(idx, w)
        compare_with_cpu(fn, indices, weight, compiled=compiled)
        out_strided = F.embedding(indices, weight)
        out_contig  = F.embedding(indices.contiguous(), weight)
        torch.testing.assert_close(
            out_strided, out_contig, atol=0.0, rtol=0.0,
            msg="embedding: strided vs contiguous index input differ",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestArange  — torch.arange
# ═══════════════════════════════════════════════════════════════════════════════

class TestArange(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.arange.

    Traced shapes:
      prefill: arange(64) → (64,) int64
      decode : arange(1)  → (1,)  int64

    Source: modeling_llama.py:403 — torch.arange(inputs_embeds.shape[1], device=...)
    """

    pytestmark = pytest.mark.torch_arange

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_arange_test"): {
            "param_sets": {
                "end64_i64_eager":    (64, False),
                "end64_i64_compiled": (64, True),
            },
        },

        # ── 2. Exact-trace decode ──────────────────────────────────────────
        ("test_exact_decode", "_run_arange_test"): {
            "param_sets": {
                "end1_i64_eager":    (1, False),
                "end1_i64_compiled": (1, True),
            },
        },
    }

    def _run_arange_test(self, end, compiled):
        """arange factory: CPU vs Spyre via compare_with_cpu (needs_device=True)."""
        def fn(end, *, device=None):
            return torch.arange(end, device=device)

        compare_with_cpu(fn, end, compiled=compiled, needs_device=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestDiff  — torch.diff
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiff(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.diff.

    Present in prefill only — the causal-mask creation path is skipped in decode.

    Traced shape:
      input  [1, 64] int64 stride (64, 1)
      append [1,  1] int64 stride (1, 1)
      dim=-1
      output [1, 64] int64

    Source: modeling_llama.py:409 — create_causal_mask
    """

    pytestmark = pytest.mark.torch_diff

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_diff_test"): {
            "param_sets": {
                "bs1_seq64_dim_neg1_i64_eager": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64),
                    make_strided_tensor((1, 1), (1, 1), dtype=I64),
                    False,
                ),
                "bs1_seq64_dim_neg1_i64_compiled": (
                    make_strided_tensor((1, 64), (64, 1), dtype=I64),
                    make_strided_tensor((1, 1), (1, 1), dtype=I64),
                    True,
                ),
            },
        },

        # ── 2. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_diff_strided_vs_contig_test"): {
            "param_sets": {
                "bs1_seq64_ncontig_input_eager": (
                    make_strided_tensor((1, 64), (128, 1), dtype=I64),
                    make_strided_tensor((1, 1), (1, 1), dtype=I64),
                    False,
                ),
                "bs1_seq64_ncontig_input_compiled": (
                    make_strided_tensor((1, 64), (128, 1), dtype=I64),
                    make_strided_tensor((1, 1), (1, 1), dtype=I64),
                    True,
                ),
            },
        },
    }

    def _run_diff_test(self, x, append, compiled):
        """diff with append tensor, dim=-1: CPU vs Spyre."""
        fn = lambda x, app: torch.diff(x, append=app, dim=-1)
        compare_with_cpu(fn, x, append, compiled=compiled)

    def _run_diff_strided_vs_contig_test(self, x, append, compiled):
        """Non-contiguous input must produce the same diff as its contiguous copy."""
        fn = lambda x, app: torch.diff(x, append=app, dim=-1)
        compare_with_cpu(fn, x, append, compiled=compiled)
        out_strided = fn(x, append)
        out_contig  = fn(x.contiguous(), append)
        torch.testing.assert_close(
            out_strided, out_contig, atol=0, rtol=0,
            msg="diff: strided vs contiguous input differ",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestEq  — torch.__eq__
# ═══════════════════════════════════════════════════════════════════════════════

class TestEq(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.__eq__ (tensor == scalar).

    Present in prefill only.

    Traced shape:
      input  (1,) int64  stride (64,)  ← non-unit stride; this is a strided view
      scalar 0            int64
      output (1,) bool

    Note: the stride (64,) on a 1-element tensor reproduces the exact
    non-contiguous layout recorded in the trace.

    Source: modeling_llama.py:409
    """

    pytestmark = pytest.mark.torch_eq

    PARAMS = {
        # ── 1. Exact-trace prefill (non-unit stride) ───────────────────────
        ("test_exact_prefill", "_run_eq_test"): {
            "param_sets": {
                "shape1_stride64_vs0_eager": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="randn"),
                    0, False,
                ),
                "shape1_stride64_vs0_compiled": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="randn"),
                    0, True,
                ),
            },
        },

        # ── 2. Non-trivial stride — assert stride is exactly (64,) ────────
        ("test_non_trivial_stride", "_run_eq_non_trivial_stride_test"): {
            "param_sets": {
                "shape1_stride64_i64_eager": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="randn"),
                    0, False,
                ),
                "shape1_stride64_i64_compiled": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="randn"),
                    0, True,
                ),
            },
        },

        # ── 3. Fill sweep — deterministic True / False outcomes ────────────
        ("test_fill_sweep", "_run_eq_test"): {
            "param_sets": {
                "zeros_eq0_true_eager": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="zeros"),
                    0, False,
                ),
                "zeros_eq0_true_compiled": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="zeros"),
                    0, True,
                ),
                "ones_eq0_false_eager": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="ones"),
                    0, False,
                ),
                "ones_eq0_false_compiled": (
                    make_strided_tensor((1,), (64,), dtype=I64, fill="ones"),
                    0, True,
                ),
            },
        },
    }

    def _run_eq_test(self, x, scalar, compiled):
        """tensor == scalar: CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: torch.eq(t, scalar)
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_eq_non_trivial_stride_test(self, x, scalar, compiled):
        """Verify stride (64,) is faithfully reproduced before running __eq__."""
        assert x.stride() == (64,), (
            f"Expected non-unit stride (64,), got {x.stride()}"
        )
        fn = lambda t: torch.eq(t, scalar)
        compare_with_cpu(fn, x, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestAll  — torch.all
# ═══════════════════════════════════════════════════════════════════════════════

class TestAll(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.all.

    Present in prefill only.

    Traced shape:
      input  (1,) bool  stride (1,)
      output ()   bool  (scalar)

    Source: modeling_llama.py:409
    """

    pytestmark = pytest.mark.torch_all

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_all_test"): {
            "param_sets": {
                "shape1_stride1_bool_eager": (
                    torch.as_strided(torch.randint(0, 2, (1,), dtype=torch.bool), size=(1,), stride=(1,)),
                    False,
                ),
                "shape1_stride1_bool_compiled": (
                    torch.as_strided(torch.randint(0, 2, (1,), dtype=torch.bool), size=(1,), stride=(1,)),
                    True,
                ),
            },
        },

        # ── 2. True / False coverage ───────────────────────────────────────
        ("test_true_false_coverage", "_run_all_test"): {
            "param_sets": {
                "all_true_eager": (
                    make_strided_tensor((1,), (1,), dtype=torch.bool, fill="ones"),
                    False,
                ),
                "all_true_compiled": (
                    make_strided_tensor((1,), (1,), dtype=torch.bool, fill="ones"),
                    True,
                ),
                "all_false_eager": (
                    make_strided_tensor((1,), (1,), dtype=torch.bool, fill="zeros"),
                    False,
                ),
                "all_false_compiled": (
                    make_strided_tensor((1,), (1,), dtype=torch.bool, fill="zeros"),
                    True,
                ),
            },
        },

        # ── 3. Scalar output check ─────────────────────────────────────────
        ("test_scalar_output", "_run_all_scalar_output_test"): {
            "param_sets": {
                "shape1_bool_eager": (
                    torch.as_strided(torch.randint(0, 2, (1,), dtype=torch.bool), size=(1,), stride=(1,)),
                    False,
                ),
                "shape1_bool_compiled": (
                    torch.as_strided(torch.randint(0, 2, (1,), dtype=torch.bool), size=(1,), stride=(1,)),
                    True,
                ),
            },
        },
    }

    def _run_all_test(self, x, compiled):
        """torch.all → scalar bool: CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: torch.all(t)
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_all_scalar_output_test(self, x, compiled):
        """Verify that torch.all produces a scalar () tensor on both CPU and Spyre."""
        cpu_out = torch.all(x)
        assert cpu_out.shape == torch.Size([]), (
            f"Expected scalar output shape (), got {cpu_out.shape}"
        )
        fn = lambda t: torch.all(t)
        compare_with_cpu(fn, x, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestTensorFloat  — torch.Tensor.float()
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorFloat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.Tensor.float().

    Traced shapes (both prefill and decode):
      prefill occ1 : (1, 64, 1)  stride (64, 1, 1)   float32 → float32
      prefill occ2 : (1, 1, 64)  stride (64, 64, 1)  int64   → float32
      decode  occ1 : (1, 64, 1)  stride (64, 1, 1)   float32 → float32
      decode  occ2 : (1, 1, 1)   stride (1, 1, 1)    int64   → float32

    Source: modeling_llama.py:125–126 — RoPE inv_freq and position_ids promotion
    """

    pytestmark = pytest.mark.torch_tensor_float

    PARAMS = {
        # ── 1–4. Exact trace (all four occurrences) ────────────────────────
        ("test_exact_trace", "_run_float_test"): {
            "param_sets": {
                "prefill_occ1_1x64x1_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), False,
                ),
                "prefill_occ1_1x64x1_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), True,
                ),
                "prefill_occ2_1x1x64_i64_eager": (
                    make_strided_tensor((1, 1, 64), (64, 64, 1), dtype=I64), False,
                ),
                "prefill_occ2_1x1x64_i64_compiled": (
                    make_strided_tensor((1, 1, 64), (64, 64, 1), dtype=I64), True,
                ),
                "decode_occ1_1x64x1_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), False,
                ),
                "decode_occ1_1x64x1_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), True,
                ),
                "decode_occ2_1x1x1_i64_eager": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), dtype=I64), False,
                ),
                "decode_occ2_1x1x1_i64_compiled": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), dtype=I64), True,
                ),
            },
        },

        # ── 5. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_float_strided_vs_contig_test"): {
            "param_sets": {
                "ncontig_i64_eager": (
                    make_strided_tensor((1, 1, 64), (256, 128, 1), dtype=I64), False,
                ),
                "ncontig_i64_compiled": (
                    make_strided_tensor((1, 1, 64), (256, 128, 1), dtype=I64), True,
                ),
            },
        },
    }

    def _run_float_test(self, x, compiled):
        """t.float(): CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: t.float()
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_float_strided_vs_contig_test(self, x, compiled):
        """Non-contiguous tensor .float() must equal its contiguous copy's .float()."""
        fn = lambda t: t.float()
        compare_with_cpu(fn, x, compiled=compiled)
        out_strided = x.float()
        out_contig  = x.contiguous().float()
        torch.testing.assert_close(
            out_strided, out_contig, atol=0, rtol=0,
            msg="float(): strided vs contiguous input differ",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestTensorExpand  — torch.Tensor.expand
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorExpand(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.Tensor.expand.

    Traced shapes (prefill and decode share identical input):
      input  (1, 64, 1) stride (64, 1, 1) float32
      sizes  [1, -1, 1]   (-1 keeps dimension unchanged)
      output (1, 64, 1) float32

    Note: expand is effectively a no-op here since the input is already
    (1, 64, 1) and expand(1, -1, 1) preserves all dimensions.

    Source: modeling_llama.py:125 — inv_freq_expanded.expand(bs, -1, 1)
    """

    pytestmark = pytest.mark.torch_tensor_expand

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_expand_test"): {
            "param_sets": {
                "bs1_seq64_1_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), False,
                ),
                "bs1_seq64_1_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), True,
                ),
            },
        },

        # ── 2. Exact-trace decode (same shape as prefill) ──────────────────
        # ("test_exact_decode", "_run_expand_test"): {
        #     "param_sets": {
        #         "bs1_seq64_1_f32_eager": (
        #             make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), False,
        #         ),
        #         "bs1_seq64_1_f32_compiled": (
        #             make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), True,
        #         ),
        #     },
        # },

        # ── 3. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_expand_strided_vs_contig_test"): {
            "param_sets": {
                "ncontig_f32_eager": (
                    make_strided_tensor((1, 64, 1), (128, 1, 1), dtype=F32), False,
                ),
                "ncontig_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (128, 1, 1), dtype=F32), True,
                ),
            },
        },

        # ── 4. Output shape check ──────────────────────────────────────────
        ("test_output_stride", "_run_expand_output_stride_test"): {
            "param_sets": {
                "bs1_seq64_1_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), False,
                ),
                "bs1_seq64_1_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), True,
                ),
            },
        },
    }

    def _run_expand_test(self, x, compiled):
        """expand(1, -1, 1): CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: t.expand(1, -1, 1)
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_expand_strided_vs_contig_test(self, x, compiled):
        """Non-contiguous input must produce the same expand output as its
        contiguous copy."""
        fn = lambda t: t.expand(1, -1, 1)
        compare_with_cpu(fn, x, compiled=compiled)
        out_strided = fn(x)
        out_contig  = fn(x.contiguous())
        torch.testing.assert_close(
            out_strided, out_contig, atol=0.0, rtol=0.0,
            msg="expand: strided vs contiguous input differ",
        )

    def _run_expand_output_stride_test(self, x, compiled):
        """Verify output shape is (1, 64, 1) after expand."""
        out = x.expand(1, -1, 1)
        assert out.shape == torch.Size([1, 64, 1]), (
            f"Expected shape (1, 64, 1), got {tuple(out.shape)}"
        )
        fn = lambda t: t.expand(1, -1, 1)
        compare_with_cpu(fn, x, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestTensorTo  — torch.Tensor.to
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorTo(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.Tensor.to(dtype).

    Traced shapes:
      prefill occ1 [0016]: (1, 64, 1)    stride (64, 1, 1)          f32 → f32
      prefill occ2 [0025]: (1, 64, 128)  stride (8192, 128, 1)       f32 → f16
      prefill occ3 [0026]: (1, 64, 4096) stride (262144, 4096, 1)    f16 → f32
      decode  occ1 [0008]: (1, 64, 1)    stride (64, 1, 1)           f32 → f32
      decode  occ2 [0017]: (1, 1, 128)   stride (128, 128, 1)        f32 → f16
      decode  occ3 [0018]: (1, 1, 4096)  stride (4096, 4096, 1)      f16 → f32

    Sources:
      modeling_llama.py:125 — .to(x.device) after inv_freq expand
      modeling_llama.py:135 — cos/sin.to(dtype=x.dtype)
      modeling_llama.py:64  — hidden_states.to(torch.float32) for RMSNorm
    """

    pytestmark = pytest.mark.torch_tensor_to

    PARAMS = {
        # ── 1. Exact trace (all six occurrences) ───────────────────────────
        ("test_exact_trace", "_run_to_test"): {
            "param_sets": {
                "prefill_occ1_1x64x1_f32_to_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), F32, False,
                ),
                "prefill_occ1_1x64x1_f32_to_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), F32, True,
                ),
                "prefill_occ2_1x64x128_f32_to_f16_eager": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), F16, False,
                ),
                "prefill_occ2_1x64x128_f32_to_f16_compiled": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), F16, True,
                ),
                "prefill_occ3_1x64x4096_f16_to_f32_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16), F32, False,
                ),
                "prefill_occ3_1x64x4096_f16_to_f32_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16), F32, True,
                ),
                "decode_occ1_1x64x1_f32_to_f32_eager": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), F32, False,
                ),
                "decode_occ1_1x64x1_f32_to_f32_compiled": (
                    make_strided_tensor((1, 64, 1), (64, 1, 1), dtype=F32), F32, True,
                ),
                "decode_occ2_1x1x128_f32_to_f16_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), dtype=F32), F16, False,
                ),
                "decode_occ2_1x1x128_f32_to_f16_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), dtype=F32), F16, True,
                ),
                "decode_occ3_1x1x4096_f16_to_f32_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16), F32, False,
                ),
                "decode_occ3_1x1x4096_f16_to_f32_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16), F32, True,
                ),
            },
        },

        # ── 2. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_to_strided_vs_contig_test"): {
            "param_sets": {
                "ncontig_f32_to_f16_eager": (
                    make_strided_tensor((1, 1, 128), (512, 256, 1), dtype=F32), F16, False,
                ),
                "ncontig_f32_to_f16_compiled": (
                    make_strided_tensor((1, 1, 128), (512, 256, 1), dtype=F32), F16, True,
                ),
            },
        },

        # ── 3. Output dtype check ──────────────────────────────────────────
        ("test_output_dtype", "_run_to_output_dtype_test"): {
            "param_sets": {
                "f32_to_f16_eager": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), F16, False,
                ),
                "f32_to_f16_compiled": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), F16, True,
                ),
                "f16_to_f32_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16), F32, False,
                ),
                "f16_to_f32_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16), F32, True,
                ),
            },
        },
    }

    def _run_to_test(self, x, dtype, compiled):
        """t.to(dtype): CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: t.to(dtype)
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_to_strided_vs_contig_test(self, x, dtype, compiled):
        """Non-contiguous .to(dtype) must match contiguous copy .to(dtype)."""
        fn = lambda t: t.to(dtype)
        compare_with_cpu(fn, x, compiled=compiled)
        out_strided = x.to(dtype)
        out_contig  = x.contiguous().to(dtype)
        torch.testing.assert_close(
            out_strided, out_contig, atol=0.0, rtol=0.0,
            msg=f"to({dtype}): strided vs contiguous input differ",
        )

    def _run_to_output_dtype_test(self, x, target_dtype, compiled):
        """Output tensor must carry the requested dtype on both CPU and Spyre."""
        out = x.to(target_dtype)
        assert out.dtype == target_dtype, (
            f"Expected dtype {target_dtype}, got {out.dtype}"
        )
        fn = lambda t: t.to(target_dtype)
        compare_with_cpu(fn, x, compiled=compiled)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestSin  — torch.sin
# ═══════════════════════════════════════════════════════════════════════════════

class TestSin(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.sin.

    Traced shapes:
      prefill: (1, 64, 128) stride (8192, 128, 1) float32
      decode : (1,  1, 128) stride (128,  128, 1) float32

    Source: modeling_llama.py:133 — sin = emb.sin() * self.attention_scaling
    """

    pytestmark = pytest.mark.torch_sin

    PARAMS = {
        # ── 1. Exact-trace prefill ─────────────────────────────────────────
        ("test_exact_prefill", "_run_sin_test"): {
            "param_sets": {
                "bs1_seq64_head128_f32_eager": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), False,
                ),
                "bs1_seq64_head128_f32_compiled": (
                    make_strided_tensor((1, 64, 128), (8192, 128, 1), dtype=F32), True,
                ),
            },
        },

        # ── 2. Exact-trace decode ──────────────────────────────────────────
        ("test_exact_decode", "_run_sin_test"): {
            "param_sets": {
                "bs1_seq1_head128_f32_eager": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), dtype=F32), False,
                ),
                "bs1_seq1_head128_f32_compiled": (
                    make_strided_tensor((1, 1, 128), (128, 128, 1), dtype=F32), True,
                ),
            },
        },

        # ── 3. Strided vs contiguous ───────────────────────────────────────
        ("test_strided_vs_contiguous", "_run_sin_strided_vs_contig_test"): {
            "param_sets": {
                "ncontig_f32_eager": (
                    make_strided_tensor((1, 64, 128), (16384, 128, 1), dtype=F32), False,
                ),
                "ncontig_f32_compiled": (
                    make_strided_tensor((1, 64, 128), (16384, 128, 1), dtype=F32), True,
                ),
            },
        },

        # ── 4. Numerical stability ─────────────────────────────────────────
        ("test_numerical_stability", "_run_sin_test"): {
            "param_sets": {
                "extreme_large_small_eager": (
                    torch.cat([
                        torch.full((1, 32, 128),  1e30, dtype=F32),
                        torch.full((1, 32, 128), -1e30, dtype=F32),
                    ], dim=1),
                    False,
                ),
                "extreme_large_small_compiled": (
                    torch.cat([
                        torch.full((1, 32, 128),  1e30, dtype=F32),
                        torch.full((1, 32, 128), -1e30, dtype=F32),
                    ], dim=1),
                    True,
                ),
            },
        },
    }

    def _run_sin_test(self, x, compiled):
        """torch.sin: CPU vs Spyre via compare_with_cpu."""
        fn = lambda t: torch.sin(t)
        compare_with_cpu(fn, x, compiled=compiled)

    def _run_sin_strided_vs_contig_test(self, x, compiled):
        """Non-contiguous tensor sin must match sin of its contiguous copy."""
        fn = lambda t: torch.sin(t)
        compare_with_cpu(fn, x, compiled=compiled)
        out_strided = fn(x)
        out_contig  = fn(x.contiguous())
        torch.testing.assert_close(
            out_strided, out_contig, atol=0.0, rtol=0.0,
            msg="sin: strided vs contiguous input differ",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestLinear  — torch.nn.functional.linear
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinear(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.nn.functional.linear (F.linear).

    Traced shapes — Prefill (seq=64):
      Q    proj : input (1, 64, 4096)  f16, weight (4096,  4096) f16 → (1, 64,  4096) f16
      K/V  proj : input (1, 64, 4096)  f16, weight (1024,  4096) f16 → (1, 64,  1024) f16
      Gate/Up   : input (1, 64, 4096)  f16, weight (14336, 4096) f16 → (1, 64, 14336) f16
      Down proj : input (1, 64, 14336) f16, weight (4096, 14336) f16 → (1, 64,  4096) f16
      LM head   : input (1, 64, 4096)  f16, weight (128256,4096) f16 → (1, 64,128256) f16

    Traced shapes — Decode (seq=1):
      Q    proj : input (1, 1, 4096)  f16, weight (4096,  4096) f16 → (1, 1,  4096) f16
      K/V  proj : input (1, 1, 4096)  f16, weight (1024,  4096) f16 → (1, 1,  1024) f16
      Gate/Up   : input (1, 1, 4096)  f16, weight (14336, 4096) f16 → (1, 1, 14336) f16
      Down proj : input (1, 1, 14336) f16, weight (4096, 14336) f16 → (1, 1,  4096) f16
      LM head   : input (1, 1, 4096)  f16, weight (128256,4096) f16 → (1, 1,128256) f16
    """

    pytestmark = pytest.mark.torch_linear

    PARAMS = {
        # ── 1. Exact-trace prefill (all 5 projections) ────────────────────
        # Weight strides from trace: (out_features, 1) — contiguous row-major.
        ("test_exact_prefill", "_run_linear_test"): {
            "param_sets": {
                "q_proj_seq64_f16_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), False,
                ),
                "q_proj_seq64_f16_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), True,
                ),
                "kv_proj_seq64_f16_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((1024, 4096), (4096, 1), dtype=F16), False,
                ),
                "kv_proj_seq64_f16_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((1024, 4096), (4096, 1), dtype=F16), True,
                ),
                "gate_up_seq64_f16_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((14336, 4096), (4096, 1), dtype=F16), False,
                ),
                "gate_up_seq64_f16_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    make_strided_tensor((14336, 4096), (4096, 1), dtype=F16), True,
                ),
                "down_proj_seq64_f16_eager": (
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), dtype=F16),
                    make_strided_tensor((4096, 14336), (14336, 1), dtype=F16), False,
                ),
                "down_proj_seq64_f16_compiled": (
                    make_strided_tensor((1, 64, 14336), (917504, 14336, 1), dtype=F16),
                    make_strided_tensor((4096, 14336), (14336, 1), dtype=F16), True,
                ),
                "lm_head_seq64_f16_eager": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    _W, False,
                ),
                "lm_head_seq64_f16_compiled": (
                    make_strided_tensor((1, 64, 4096), (262144, 4096, 1), dtype=F16),
                    _W, True,
                ),
            },
        },

        # ── 2. Exact-trace decode (all 5 projections) ─────────────────────
        ("test_exact_decode", "_run_linear_test"): {
            "param_sets": {
                "q_proj_seq1_f16_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), False,
                ),
                "q_proj_seq1_f16_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), True,
                ),
                "kv_proj_seq1_f16_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((1024, 4096), (4096, 1), dtype=F16), False,
                ),
                "kv_proj_seq1_f16_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((1024, 4096), (4096, 1), dtype=F16), True,
                ),
                "gate_up_seq1_f16_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((14336, 4096), (4096, 1), dtype=F16), False,
                ),
                "gate_up_seq1_f16_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    make_strided_tensor((14336, 4096), (4096, 1), dtype=F16), True,
                ),
                "down_proj_seq1_f16_eager": (
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), dtype=F16),
                    make_strided_tensor((4096, 14336), (14336, 1), dtype=F16), False,
                ),
                "down_proj_seq1_f16_compiled": (
                    make_strided_tensor((1, 1, 14336), (14336, 14336, 1), dtype=F16),
                    make_strided_tensor((4096, 14336), (14336, 1), dtype=F16), True,
                ),
                "lm_head_seq1_f16_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    _W, False,
                ),
                "lm_head_seq1_f16_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), dtype=F16),
                    _W, True,
                ),
            },
        },

        # ── 3. Strided vs contiguous — all 5 projections × prefill + decode ─
        # Input has a doubled batch stride (non-contiguous); weight uses traced
        # strides. Asserts strided-input result == contiguous-input result.
        ("test_strided_vs_contiguous", "_run_linear_strided_vs_contig_test"): {
            "param_sets": {
                # Prefill (seq=64) — batch stride doubled to force non-contiguity
                "prefill_q_proj_eager": (
                    make_strided_tensor((1, 64, 4096), (524288, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), False,
                ),
                "prefill_q_proj_compiled": (
                    make_strided_tensor((1, 64, 4096), (524288, 4096, 1), dtype=F16),
                    make_strided_tensor((4096, 4096), (4096, 1), dtype=F16), True,
                ),
            },
        },
    }

    def _run_linear_test(self, inp, weight, compiled):
        """F.linear(inp, weight): CPU vs Spyre via compare_with_cpu."""
        fn = lambda x, w: F.linear(x, w)
        compare_with_cpu(fn, inp, weight, compiled=compiled, atol=0.01, rtol=0.01)

    def _run_linear_strided_vs_contig_test(self, inp, weight, compiled):
        """Non-contiguous input must produce the same linear output as its
        contiguous copy."""
        fn = lambda x, w: F.linear(x, w)
        compare_with_cpu(fn, inp, weight, compiled=compiled)
        out_strided = fn(inp, weight)
        out_contig  = fn(inp.contiguous(), weight)
        torch.testing.assert_close(
            out_strided, out_contig,
            atol=0.01, rtol=0.01,
            msg="linear: strided vs contiguous input differ",
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SDPA Strided Tensor Factories for Llama-3.1-8B-Instruct
# ═════════════════════════════════════════════════════════════════════════════

# KV-cache configuration
_KV_CACHE_LEN  = 2048   # Max cache length (from config max_position_embeddings)
_PREFILL_SEQ   = 64     # Prefill sequence length from trace
_DECODE_KV_LEN = 65     # Decode KV length: 64 prefill context + 1 new token
_DECODE_SEQ    = 1      # Decode query sequence length


def _make_q_prefill_strides(batch: int = 1, seq_len: int = _PREFILL_SEQ) -> tuple:
    """Calculate Q strides for prefill: [B, 32, S, 128]
    stride = (32*S*128,  128,  32*128,  1)
    """
    batch_stride = NUM_Q_HEADS * seq_len * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride   = NUM_Q_HEADS * HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)


def _make_q_decode_strides(batch: int = 1) -> tuple:
    """Calculate Q strides for decode: [B, 32, 1, 128]
    stride = (32*1*128,  128,  128,  1)  →  (4096, 128, 128, 1)
    """
    batch_stride = NUM_Q_HEADS * _DECODE_SEQ * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride   = HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)


def _make_kv_strides(batch: int = 1, cache_len: int = _DECODE_KV_LEN) -> tuple:
    """Calculate K/V strides for KV-cache: [B, 8, L, 128]
    stride = (8*L*128,  128,  8*128,  1)
    """
    batch_stride = NUM_KV_HEADS * cache_len * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride   = NUM_KV_HEADS * HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)


def _make_q_prefill(
    batch: int = 1,
    seq: int = _PREFILL_SEQ,
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn",
) -> torch.Tensor:
    """Return Q with prefill layout [B, 32, S, 128] with proper strides."""
    strides = _make_q_prefill_strides(batch, seq)
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, seq, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )


def _make_q_decode(
    batch: int = 1,
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn",
) -> torch.Tensor:
    """Return Q with decode layout [B, 32, 1, 128] with proper strides.
    Matches observed stride (4096, 128, 128, 1).
    """
    strides = _make_q_decode_strides(batch)
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, 1, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )


def _make_kv(
    batch: int = 1,
    cache_len: int = _DECODE_KV_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn",
) -> tuple:
    """Return (k, v) with KV-cache layout [B, 8, L, 128].
    Default cache_len=65 matches observed decode KV shape (1, 8, 65, 128).
    """
    strides = _make_kv_strides(batch, cache_len)
    k = make_strided_tensor(
        (batch, NUM_KV_HEADS, cache_len, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )
    v = make_strided_tensor(
        (batch, NUM_KV_HEADS, cache_len, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )
    return k, v


# Prefill param dicts with strided tensors — seq=64, kv=seq
_STRIDED_PREFILL_PARAMS = {
    "bs1_seq64_fp16": (_make_q_prefill(1, 64, F16), *_make_kv(1, 64, F16)),
    "bs1_seq64_fp32": (_make_q_prefill(1, 64, torch.float32), *_make_kv(1, 64, torch.float32)),
    "bs1_seq64_bf16": (_make_q_prefill(1, 64, torch.bfloat16), *_make_kv(1, 64, torch.bfloat16)),
}

# Additional prefill variants with different sequence lengths
_STRIDED_PREFILL_VARIANTS = {
    "bs1_seq1_fp16":   (_make_q_prefill(1,   1, F16), *_make_kv(1,   1, F16)),
    "bs1_seq8_fp16":   (_make_q_prefill(1,   8, F16), *_make_kv(1,   8, F16)),
    "bs1_seq16_fp16":  (_make_q_prefill(1,  16, F16), *_make_kv(1,  16, F16)),
    "bs1_seq32_fp16":  (_make_q_prefill(1,  32, F16), *_make_kv(1,  32, F16)),
    "bs1_seq64_fp16":  (_make_q_prefill(1,  64, F16), *_make_kv(1,  64, F16)),
    "bs1_seq128_fp16": (_make_q_prefill(1, 128, F16), *_make_kv(1, 128, F16)),
}

# Decode param dicts — Q: (1,32,1,128) stride (4096,128,128,1),
#                      K/V: (1,8,65,128) stride (66560,8320,128,1)
_STRIDED_DECODE_PARAMS = {
    "bs1_kv65_fp16":    (_make_q_decode(1, F16),                    *_make_kv(1, _DECODE_KV_LEN, F16)),
    "bs1_kv65_zeros":   (_make_q_decode(1, F16,   fill="zeros"),    *_make_kv(1, _DECODE_KV_LEN, F16, fill="zeros")),
    "bs1_kv65_ones":    (_make_q_decode(1, F16,   fill="ones"),     *_make_kv(1, _DECODE_KV_LEN, F16, fill="ones")),
}

# Growing KV-cache variants for decode
_STRIDED_GROWING_KV_PARAMS = {
    f"kv{kv}": (_make_q_decode(1, F16), *_make_kv(1, kv, F16))
    for kv in [1, 2, 4, 8, 16, 32, 64, 65, 128, 256, 512, 1024, 2048]
}

# Batch variants
_STRIDED_BATCH_PARAMS = {
    "bs2_seq64_fp16": (_make_q_prefill(2, 64, F16), *_make_kv(2, 64, F16)),
    "bs4_seq64_fp16": (_make_q_prefill(4, 64, F16), *_make_kv(4, 64, F16)),
    "bs8_seq64_fp16": (_make_q_prefill(8, 64, F16), *_make_kv(8, 64, F16)),
}

# Sliding window variants (SLIDING_WINDOW is None for this model)
_STRIDED_SLIDING_WINDOW_PARAMS = {
    "seq64_kv64":    (_make_q_prefill(1,  64, F16), *_make_kv(1,  64, F16)),
    "seq64_kv2048":  (_make_q_prefill(1,  64, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "seq128_kv2048": (_make_q_prefill(1, 128, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
}


def _slice_kv_to_seq(kv: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Slice KV cache to specific sequence length while preserving stride patterns.
    Returns a contiguous tensor to avoid stride issues on Spyre.
    """
    return kv[:, :, :seq_len, :].contiguous()


# ═════════════════════════════════════════════════════════════════════════════
#  TestSDPA  —  torch.nn.functional.scaled_dot_product_attention
# ═════════════════════════════════════════════════════════════════════════════

class TestSDPA(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Eager and compiled CPU vs Spyre SDPA comparison tests for
    Llama-3.1-8B-Instruct.

    Observed shapes (from op trace):
      Prefill  Q : (1, 32, 64, 128)  stride (262144, 128, 4096, 1)  float16
               K : (1,  8, 64, 128)  stride  (65536, 128, 1024, 1)  float16
               V : (1,  8, 64, 128)  stride  (65536, 128, 1024, 1)  float16
      Decode   Q : (1, 32,  1, 128)  stride   (4096, 128,  128, 1)  float16
               K : (1,  8, 65, 128)  stride  (66560, 128, 8320, 1)  float16
               V : (1,  8, 65, 128)  stride  (66560, 128, 8320, 1)  float16
    """

    pytestmark = pytest.mark.torch_sdpa

    torch.manual_seed(0xBEEF_2506)

    SDPA_EAGER_ATOL    = 2e-2
    SDPA_EAGER_RTOL    = 2e-2
    SDPA_COMPILED_ATOL = 5e-2
    SDPA_COMPILED_RTOL = 5e-2

    PARAMS = {
        # ── Decode tests (target shape) ───────────────────────────────────────
        ("test_sdpa_decode", "test_sdpa_decode"): {
            "param_sets": {
                "bs1_kv65_fp16_eager":    (*_STRIDED_DECODE_PARAMS["bs1_kv65_fp16"], False),
                "bs1_kv65_fp16_compiled": (*_STRIDED_DECODE_PARAMS["bs1_kv65_fp16"], True),
            },
        },

        # ── Prefill causal tests ──────────────────────────────────────────────
        ("test_sdpa_prefill_causal", "test_sdpa_prefill_causal"): {
            "param_sets": {
                "bs1_seq64_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], False),
                "bs1_seq64_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], True),
                "bs1_seq8_fp16_eager":     (*_STRIDED_PREFILL_VARIANTS["bs1_seq8_fp16"], False),
                "bs1_seq16_fp16_eager":    (*_STRIDED_PREFILL_VARIANTS["bs1_seq16_fp16"], False),
                "bs1_seq32_fp16_eager":    (*_STRIDED_PREFILL_VARIANTS["bs1_seq32_fp16"], False),
                "bs1_seq64_fp16_v2_eager": (*_STRIDED_PREFILL_VARIANTS["bs1_seq64_fp16"], False),
                "bs1_seq128_fp16_eager":   (*_STRIDED_PREFILL_VARIANTS["bs1_seq128_fp16"], False),
            },
        },

        # ── Growing KV cache (autoregressive decode) ──────────────────────────
        ("test_sdpa_growing_kvcache", "test_sdpa_growing_kvcache"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in _STRIDED_GROWING_KV_PARAMS.items()
                   if int(k.replace("kv", "")) >= 8},
                **{f"{k}_compiled": (*v, True)  for k, v in _STRIDED_GROWING_KV_PARAMS.items()
                   if int(k.replace("kv", "")) >= 8},
            },
        },

        # ── Batch consistency tests ───────────────────────────────────────────
        ("test_sdpa_batch_consistency", "test_sdpa_batch_consistency"): {
            "param_sets": {
                "bs2_seq64_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs2_seq64_fp16"], False),
                "bs2_seq64_fp16_compiled": (*_STRIDED_BATCH_PARAMS["bs2_seq64_fp16"], True),
                "bs4_seq64_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs4_seq64_fp16"], False),
            },
        },

        # ── Causal flag vs mask tests ─────────────────────────────────────────
        ("test_sdpa_causal_flag_vs_mask", "test_sdpa_causal_flag_vs_mask"): {
            "param_sets": {
                "bs1_seq64_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], False),
                "bs1_seq64_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], True),
            },
        },

        # ── Attention weights sum to one tests ────────────────────────────────
        ("test_sdpa_weights_sum_to_one", "test_sdpa_weights_sum_to_one"): {
            "param_sets": {
                "bs1_seq64_fp16_eager": (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], False),
            },
        },

        # ── GQA shape tests ───────────────────────────────────────────────────
        ("test_sdpa_gqa_shape", "test_sdpa_gqa_shape"): {
            "param_sets": {
                "bs2_seq64_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs2_seq64_fp16"], False),
                "bs2_seq64_fp16_compiled": (*_STRIDED_BATCH_PARAMS["bs2_seq64_fp16"], True),
            },
        },

        # ── Gradient flow tests (eager only) ──────────────────────────────────
        ("test_sdpa_gradient_flow", "test_sdpa_gradient_flow"): {
            "param_sets": {
                "bs1_seq64_fp16_eager": (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], False),
            },
        },

        # ── Determinism tests ─────────────────────────────────────────────────
        ("test_sdpa_determinism", "test_sdpa_determinism"): {
            "param_sets": {
                "bs1_seq64_fp16_eager":  (*_STRIDED_PREFILL_PARAMS["bs1_seq64_fp16"], False),
                "bs1_decode_fp16_eager": (*_STRIDED_DECODE_PARAMS["bs1_kv65_fp16"],   False),
            },
        },
    }

    # ── Helper to create causal mask without in-place ops ─────────────────────
    def _make_causal_mask(self, seq_len: int, dtype: torch.dtype, device):
        """Create causal mask using torch.where (no in-place modification)."""
        upper_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        mask = torch.where(upper_mask.unsqueeze(0).unsqueeze(0),
                           torch.tensor(float("-inf"), dtype=dtype, device=device),
                           torch.tensor(0.0,           dtype=dtype, device=device))
        return mask

    # ── Helper to create padding mask without in-place ops ────────────────────
    def _make_padding_mask(self, seq_len: int, kv_len: int, dtype: torch.dtype, device):
        """Create padding mask without in-place operations."""
        half = kv_len // 2
        pad_mask_bool = torch.zeros(1, 1, seq_len, kv_len, dtype=torch.bool, device=device)
        pad_mask_bool[:, :, :, half:] = True
        mask = torch.where(pad_mask_bool,
                           torch.tensor(float("-inf"), dtype=dtype, device=device),
                           torch.tensor(0.0,           dtype=dtype, device=device))
        return mask

    # ── Base test methods ─────────────────────────────────────────────────────

    def test_sdpa_prefill_causal(self, q, k, v, compiled):
        """Prefill: is_causal=True — CPU vs Spyre."""
        seq_q = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=True)

        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL

        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled,
                         atol=atol, rtol=rtol)

    def test_sdpa_decode(self, q, k, v, compiled):
        """Decode (seq_q=1): no mask — CPU vs Spyre.
        Q: (1,32,1,128) stride (4096,128,128,1)
        K/V: (1,8,65,128) stride (66560,8320,128,1)
        """
        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=False)

        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL

        compare_with_cpu(fn, q, k, v, compiled=compiled, atol=atol, rtol=rtol)

    def test_sdpa_growing_kvcache(self, q, k, v, compiled):
        """Growing KV cache test for decode with adjusted tolerances."""
        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=False)

        atol = 3e-2 if compiled else 2e-2
        rtol = 3e-2 if compiled else 2e-2

        compare_with_cpu(fn, q, k, v, compiled=compiled, atol=atol, rtol=rtol)

    def test_sdpa_causal_flag_vs_mask(self, q, k, v, compiled):
        """is_causal=True flag and explicit causal mask must agree."""
        seq_q = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        def fn(q, k, v):
            out_flag = sdpa_fn(q, k, v, is_causal=True)
            mask     = self._make_causal_mask(seq_q, q.dtype, q.device)
            out_mask = sdpa_fn(q, k, v, attn_mask=mask, is_causal=False)
            torch.testing.assert_close(
                out_flag, out_mask,
                atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
                msg="is_causal flag vs explicit mask differ",
            )
            return out_flag

        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL

        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled,
                         atol=atol, rtol=rtol)

    def test_sdpa_weights_sum_to_one(self, q, k, v, compiled):
        """Each attention-weight row must sum to 1 (softmax in fp32)."""
        if compiled:
            self.skipTest("Weights sum to one test only runs in eager mode")

        seq_q    = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * SCALE
            mask   = self._make_causal_mask(seq_q, q.dtype, q.device)
            scores = scores + mask
            weights = torch.softmax(scores.float(), dim=-1)
            row_sums = weights.sum(dim=-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums),
                atol=1e-3, rtol=1e-3,
                msg="Attention weights do not sum to 1",
            )
            return row_sums

        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=False,
                         atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL)

    def test_sdpa_gqa_shape(self, q, k, v, compiled):
        """After GQA head expansion K and V shapes must have correct heads."""
        seq_q    = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            assert k_exp.shape[1] == NUM_Q_HEADS, (
                f"K heads mismatch: {k_exp.shape[1]} != {NUM_Q_HEADS}"
            )
            assert v_exp.shape[1] == NUM_Q_HEADS, (
                f"V heads mismatch: {v_exp.shape[1]} != {NUM_Q_HEADS}"
            )
            assert k_exp.shape[0] == q.shape[0], "Batch mismatch"
            assert k_exp.shape[3] == q.shape[3], "Head dim mismatch"
            return sdpa_fn(q, k, v, is_causal=True)

        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL

        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled,
                         atol=atol, rtol=rtol)

    def test_sdpa_batch_consistency(self, q, k, v, compiled):
        """All batch items must produce identical outputs when inputs are identical."""
        B     = q.shape[0]
        seq_q = q.shape[2]

        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        # Replicate batch item 0 across every batch slot so all items are identical.
        # SDPA must then produce identical outputs for every item.
        q_rep = q[0:1].expand(B, -1, -1, -1).contiguous()
        k_rep = k_sliced[0:1].expand(B, -1, -1, -1).contiguous()
        v_rep = v_sliced[0:1].expand(B, -1, -1, -1).contiguous()

        def fn(q, k, v):
            out = sdpa_fn(q, k, v, is_causal=True)
            for i in range(1, B):
                torch.testing.assert_close(
                    out[0], out[i],
                    atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
                    msg=f"Batch item {i} differs from item 0",
                )
            return out

        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL

        compare_with_cpu(fn, q_rep, k_rep, v_rep, compiled=compiled,
                        atol=atol, rtol=rtol)

    def test_sdpa_gradient_flow(self, q, k, v, compiled):
        """Q gradient back-propagates through SDPA without NaNs (eager only)."""
        if compiled:
            self.skipTest("Gradient test only runs in eager mode")

        seq_q    = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()

        q_grad = q.clone().detach().requires_grad_(True)
        k_grad = k_sliced.clone().detach()
        v_grad = v_sliced.clone().detach()

        out = sdpa_fn(q_grad, k_grad, v_grad, is_causal=True)
        out.sum().backward()

        assert q_grad.grad is not None,               "Gradient for Q is None"
        assert not torch.isnan(q_grad.grad).any(),    "NaN in Q gradient"
        assert q_grad.grad.abs().max() > 0,           "Q gradient is all zeros"

        q_cpu = q.clone().detach().requires_grad_(True)
        k_cpu = k_sliced.clone().detach()
        v_cpu = v_sliced.clone().detach()

        out_cpu = sdpa_fn(q_cpu, k_cpu, v_cpu, is_causal=True)
        out_cpu.sum().backward()

        assert q_cpu.grad is not None, "CPU gradient for Q is None"

        torch.testing.assert_close(
            q_grad.grad.cpu(), q_cpu.grad,
            atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
            msg="Q gradient mismatch between CPU and Spyre",
        )

    def test_sdpa_determinism(self, q, k, v, compiled):
        """Two consecutive identical SDPA calls must return the same output."""
        if compiled:
            self.skipTest("Determinism test only runs in eager mode")

        seq_q = q.shape[2]
        if seq_q > 1:
            k = k[:, :, :seq_q, :].contiguous()
            v = v[:, :, :seq_q, :].contiguous()

        def fn(q, k, v):
            out1 = sdpa_fn(q, k, v, is_causal=(seq_q > 1))
            out2 = sdpa_fn(q, k, v, is_causal=(seq_q > 1))
            torch.testing.assert_close(
                out1, out2,
                atol=1e-5, rtol=1e-5,
                msg="Two identical SDPA calls returned different results",
            )
            return out1

        compare_with_cpu(fn, q, k, v, compiled=False,
                         atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL)
        

# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()