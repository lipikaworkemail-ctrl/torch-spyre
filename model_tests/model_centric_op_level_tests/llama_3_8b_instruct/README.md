# Model-Centric Operator-Level Tests

CPU vs Spyre comparison tests for every operator pattern observed during a
forward pass of the `meta-llama/Llama-3.1-8B-Instruct` model.

Each test executes an operation on CPU and on the Spyre accelerator with
identical inputs, then asserts that the outputs are numerically close within
configurable tolerances. Both **eager** (uncompiled) and **compiled**
(`torch.compile`) execution paths are covered for every operator.

---

## Table of Contents

- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Running Tests](#running-tests)
  - [Full Suite](#run-the-full-suite)
  - [By Operator](#run-by-operator-mark)
  - [By Execution Mode](#run-by-execution-mode)
  - [Combined Filters](#combining-filters)
  - [Useful Flags](#useful-flags)
- [Operators Covered](#operators-covered)
- [Tolerances](#tolerances)
- [Debugging Failures](#debugging-failures)
- [Known Constraints](#known-constraints)
- [How Test Expansion Works](#how-test-expansion-works--parameterizedtestmeta)
- [License](#license)

---

## Model Architecture

| Parameter | Constant | Value |
|---|---|---|
| `num_attention_heads` (query heads) | `NUM_Q_HEADS` | `32` |
| `num_key_value_heads` (KV heads, GQA) | `NUM_KV_HEADS` | `8` |
| `head_dim` | `HEAD_DIM` | `128` |
| `gqa_groups` | `GQA_GROUPS` | `4` |
| `num_hidden_layers` | `NUM_LAYERS` | `32` |
| Hidden / residual-stream size | `HIDDEN_SIZE` | `4096` |
| `intermediate_size` (FFN) | `INTERMEDIATE_SIZE` | `14336` |
| `vocab_size` | `VOCAB_SIZE` | `128256` |
| `rope_theta` | `ROPE_THETA` | `500000.0` |
| `sliding_window` | `SLIDING_WINDOW` | `None` |
| Default dtype | `DEFAULT_DTYPE` | `torch.float16` |

---

## Requirements

Set up the torch-spyre environment by following the official installation guide:
https://github.ibm.com/ai-foundation/torch-spyre-docs/blob/main/docs/basic_install.md

```bash
cd torch-spyre

# Create a virtual environment with access to system site packages
uv venv --system-site-packages

# Activate the virtual environment
source .venv/bin/activate

# Install torch-spyre along with all dependencies (including torch_sendnn)
uv sync --all-extras --active
```

---

## Running Tests

```bash
cd llama_3_8b_instruct
pytest test_llama_3_8b_instruct.py -k "eager"
```

Note: Some operations are not yet implemented on Spyre, so test failures are expected when running on this device.

#### Run Tests on CPU

To avoid Spyre-related failures, switch the device to CPU in the following utility file:
`utils_llama_3_8b_instruct.py`

Update the device configuration:
```bash
DEVICE = torch.device("spyre")
```

Change it to:
```bash
DEVICE = torch.device("cpu")
```

### Run by Operator (Mark)

Each test class is tagged with a `pytestmark`. Pass `-m <mark>` to select
one or more operators:

```bash
# torch.add
pytest test_llama_3_8b_instruct.py -m torch_add

# torch.sub
pytest test_llama_3_8b_instruct.py -m torch_sub

# torch.cat
pytest test_llama_3_8b_instruct.py -m torch_cat

# torch.rsqrt
pytest test_llama_3_8b_instruct.py -m torch_rsqrt

# torch.unsqueeze
pytest test_llama_3_8b_instruct.py -m torch_unsqueeze

# torch.pow
pytest test_llama_3_8b_instruct.py -m torch_pow

# torch.ne
pytest test_llama_3_8b_instruct.py -m torch_ne

# torch.matmul
pytest test_llama_3_8b_instruct.py -m torch_matmul

# torch.mean
pytest test_llama_3_8b_instruct.py -m torch_mean

# torch.neg
pytest test_llama_3_8b_instruct.py -m torch_neg

# Tensor.__getitem__  (indexing / slicing)
pytest test_llama_3_8b_instruct.py -m torch_getitem

# torch.cumsum
pytest test_llama_3_8b_instruct.py -m torch_cumsum

# torch.transpose
pytest test_llama_3_8b_instruct.py -m torch_transpose

# torch.cos
pytest test_llama_3_8b_instruct.py -m torch_cos

# torch.mul
pytest test_llama_3_8b_instruct.py -m torch_mul

# Tensor.view
pytest test_llama_3_8b_instruct.py -m torch_view

# Tensor.contiguous
pytest test_llama_3_8b_instruct.py -m torch_contiguous

# torch.reshape
pytest test_llama_3_8b_instruct.py -m torch_reshape

# torch.nn.functional.silu
pytest test_llama_3_8b_instruct.py -m torch_nn_functional_silu

# torch.nn.functional.embedding
pytest test_llama_3_8b_instruct.py -m torch_embedding

# torch.arange
pytest test_llama_3_8b_instruct.py -m torch_arange

# torch.diff
pytest test_llama_3_8b_instruct.py -m torch_diff

# torch.__eq__
pytest test_llama_3_8b_instruct.py -m torch_eq

# torch.all
pytest test_llama_3_8b_instruct.py -m torch_all

# Tensor.float()
pytest test_llama_3_8b_instruct.py -m torch_tensor_float

# Tensor.expand
pytest test_llama_3_8b_instruct.py -m torch_tensor_expand

# Tensor.to  (dtype transfer)
pytest test_llama_3_8b_instruct.py -m torch_tensor_to

# torch.sin
pytest test_llama_3_8b_instruct.py -m torch_sin

# torch.nn.functional.linear
pytest test_llama_3_8b_instruct.py -m torch_linear

# Scaled dot-product attention
pytest test_llama_3_8b_instruct.py -m torch_sdpa
```

### Run by Execution Mode

```bash
# Eager (uncompiled) variants only — all operators
pytest test_llama_3_8b_instruct.py -k "eager"

# Compiled (torch.compile) variants only — all operators
pytest test_llama_3_8b_instruct.py -k "compiled"
```

### Useful Flags

```bash
# Stop at the first failure with a full traceback and local variable values
pytest test_llama_3_8b_instruct.py -m torch_mul -x --tb=long

# Compact summary — suppress per-test verbose output
pytest test_llama_3_8b_instruct.py -q

# Disable output capture so print() appears immediately
pytest test_llama_3_8b_instruct.py -s

# Collect (list) tests without executing them
pytest test_llama_3_8b_instruct.py -m torch_view --collect-only

# Run tests in parallel (requires pytest-xdist)
pytest test_llama_3_8b_instruct.py -n auto

# Show the slowest 10 tests after the run
pytest test_llama_3_8b_instruct.py --durations=10

# Re-run only the tests that failed in the previous session
pytest test_llama_3_8b_instruct.py --lf

# Re-run failures first, then the rest of the suite
pytest test_llama_3_8b_instruct.py --ff
```

---

## Operators Covered

### `torch.add` — `TestAdd`

**Mark:** `torch_add`

All `torch.add` call signatures observed in the model:

| Patterns | Signature | Example |
|---|---|---|
| 000–002 | `tensor + scalar` | Arange offset, RMSNorm epsilon |
| 003–006 | Binary `tensor + tensor` | RoPE rotary embed, residual add |

```bash
pytest test_llama_3_8b_instruct.py -m torch_add
pytest test_llama_3_8b_instruct.py -m torch_add -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_add -k "compiled"
```

---

### `torch.sub` — `TestSub`

**Mark:** `torch_sub`

One call signature observed in the model — position index offset from a
non-contiguous slice of the position-ids buffer:

| Pattern | Signature | Example |
|---|---|---|
| 000 | `tensor - scalar` | `[1,1] int64 non-contiguous − 1` |

```bash
pytest test_llama_3_8b_instruct.py -m torch_sub
pytest test_llama_3_8b_instruct.py -m torch_sub -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_sub -k "compiled"
```

---

### `torch.cat` — `TestCat`

**Mark:** `torch_cat`

Covers all `torch.cat` call-sites — RoPE sin/cos split/concat and KV-cache
concatenation (patterns 000–006).

| Variant | Pattern | Description |
|---|---|---|
| `same_dim` | 000–005 | Both inputs have the same shape; concat on last dim (`-1`) |
| `mixed_seq` | 006 | Inputs differ in seq dim (KV-cache append); concat on `dim=-2` |

```bash
pytest test_llama_3_8b_instruct.py -m torch_cat
pytest test_llama_3_8b_instruct.py -m torch_cat -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_cat -k "compiled"
pytest test_llama_3_8b_instruct.py -m torch_cat -k "pattern_003"
```

---

### `torch.rsqrt` — `TestRsqrt`

**Mark:** `torch_rsqrt`

Appears in RMSNorm: `rsqrt(variance + eps)`.

| Pattern | Shape | Usage |
|---|---|---|
| 000 | `[1, 64, 1]` | Variance prefill |
| 001 | `[1, 1, 1]` | Variance decode |

```bash
pytest test_llama_3_8b_instruct.py -m torch_rsqrt
pytest test_llama_3_8b_instruct.py -m torch_rsqrt -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_rsqrt -k "compiled"
```

---

### `torch.unsqueeze` — `TestUnsqueeze`

**Mark:** `torch_unsqueeze`

| Pattern | Shape | `dim` | Usage |
|---|---|---|---|
| 000 | `(64,)` | `0` | Position-ids prefill |
| 001 | `(1,)` | `0` | Position-ids decode |
| 002 | `(1, 64, 128)` | `1` | Hidden state expand for attention |

```bash
pytest test_llama_3_8b_instruct.py -m torch_unsqueeze
pytest test_llama_3_8b_instruct.py -m torch_unsqueeze -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_unsqueeze -k "compiled"
```

---

### `torch.pow` — `TestPow`

**Mark:** `torch_pow`

Integer exponent `2` for variance computation in RMSNorm:

| Pattern | Shape | Usage |
|---|---|---|
| 000 | `[1, 64, 4096]` | Variance prefill |
| 001 | `[1, 1, 4096]` | Variance decode |

```bash
pytest test_llama_3_8b_instruct.py -m torch_pow
pytest test_llama_3_8b_instruct.py -m torch_pow -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_pow -k "compiled"
```

---

### `torch.ne` — `TestNe`

**Mark:** `torch_ne`

Appears in attention mask generation: `input_ids != pad_token_id`.

| Pattern | Shape | Usage |
|---|---|---|
| 000 | `[1, 64] int64 non-contiguous` | Attention mask boolean |

```bash
pytest test_llama_3_8b_instruct.py -m torch_ne
pytest test_llama_3_8b_instruct.py -m torch_ne -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_ne -k "compiled"
```

---

### `torch.matmul` — `TestMatmul`

**Mark:** `torch_matmul`

| Shape A | Shape B | Output | Coverage |
|---|---|---|---|
| `(1, 64, 1)` | `(1, 1, 64)` | `(1, 64, 64)` | Attention score outer-product (prefill) |
| `(1, 64, 1)` | `(1, 1, 1)` | `(1, 64, 1)` | Scalar scaling / projection (decode) |

Variants cover `torch.matmul`, `a.matmul(b)`, `a @ b`, zero/ones inputs,
matmul+bias, and CPU-only `±inf` / `NaN` propagation checks.

```bash
pytest test_llama_3_8b_instruct.py -m torch_matmul
pytest test_llama_3_8b_instruct.py -m torch_matmul -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_matmul -k "compiled"
```

---

### `torch.mean` — `TestMean`

**Mark:** `torch_mean`

Input shapes `(1, 1, 4096)` and `(1, 64, 4096)`. Variants cover global
mean, `dim=-1` with and without `keepdim`, `dim=1` (token pooling), `dim=0`
(batch mean), method alias `t.mean()`, zero/one inputs, `mean → float16`
cast, and CPU-only NaN / inf propagation checks.

```bash
pytest test_llama_3_8b_instruct.py -m torch_mean
pytest test_llama_3_8b_instruct.py -m torch_mean -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_mean -k "compiled"
```

---

### `torch.neg` — `TestNeg`

**Mark:** `torch_neg`

Unary elementwise negation from the `rotate_half` function in the RoPE path:

| Pattern | Shape | Source |
|---|---|---|
| 000 | `[1, 32, 64, 64]` | Q prefill |
| 001 | `[1, 8, 64, 64]` | K/V prefill |
| 002 | `[1, 32, 1, 64]` | Q decode |
| 003 | `[1, 8, 1, 64]` | K/V decode |

Additional variants cover `x.neg()` method alias, `-x` operator, zeros/ones
inputs, and a CPU-only double-negation identity check.

```bash
pytest test_llama_3_8b_instruct.py -m torch_neg
pytest test_llama_3_8b_instruct.py -m torch_neg -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_neg -k "compiled"
```

---

### `Tensor.__getitem__` — `TestGetitem`

**Mark:** `torch_getitem`

Index expressions are plain Python (integer, `slice`, tuple of slices) and
are passed through unchanged.

Sub-group marks (auto-derived by `ParameterizedTestMeta`):

| Sub-mark | Base method | What it tests |
|---|---|---|
| `torch_getitem_shape` | `_run_getitem_shape_test` | Output shape is correct after indexing |
| `torch_getitem_values` | `_run_getitem_values_test` | Indexed values match CPU reference |
| `torch_getitem_dtype` | `_run_getitem_dtype_test` | dtype is not changed after indexing |

```bash
pytest test_llama_3_8b_instruct.py -m torch_getitem
pytest test_llama_3_8b_instruct.py -m torch_getitem -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_getitem -k "compiled"
```

---

### `torch.cumsum` — `TestCumsum`

**Mark:** `torch_cumsum`

Covers `bool` and `int64` input dtypes along `dim=-1`:

| Variant | Shape | dtype |
|---|---|---|
| Prefill zeros / ones / mixed | `[1, 64]` | `bool` |
| Prefill arange | `[1, 64]` | `int64` |

```bash
pytest test_llama_3_8b_instruct.py -m torch_cumsum
pytest test_llama_3_8b_instruct.py -m torch_cumsum -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_cumsum -k "compiled"
```

---

### `torch.transpose` — `TestTranspose`

**Mark:** `torch_transpose`

Input shapes cover all `transpose` call-sites in the attention and projection
layers:

| 3-D shapes | 4-D shapes |
|---|---|
| `[1, 64, 1]`, `[1, 64, 64]` (float32) | `[1, 1, 32, 128]`, `[1, 1, 8, 128]` (float16) |
| | `[1, 32, 1, 128]`, `[1, 64, 32, 128]` (float16) |
| | `[1, 64, 8, 128]`, `[1, 32, 64, 128]` (float16) |

Sub-groups test shape correctness, value correctness, negative dimension
indexing (e.g. `(-2, -1) == (2, 3)`), and dtype preservation.

```bash
pytest test_llama_3_8b_instruct.py -m torch_transpose
pytest test_llama_3_8b_instruct.py -m torch_transpose -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_transpose -k "compiled"
```

---

### `torch.cos` — `TestCos`

**Mark:** `torch_cos`

Pointwise cosine from the RoPE rotary embedding path. Shapes `[1, 64, 128]`
(prefill) and `[1, 1, 128]` (decode). Sub-groups test shape correctness,
value correctness (random and zero inputs), and dtype preservation.

```bash
pytest test_llama_3_8b_instruct.py -m torch_cos
pytest test_llama_3_8b_instruct.py -m torch_cos -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_cos -k "compiled"
```

---

### `torch.mul` — `TestMul`

**Mark:** `torch_mul`

All 12 `mul.*` entries sourced from `Llama-3.1-8B-Instruct_spyre.yaml`:

| yaml entry | Shapes | Description |
|---|---|---|
| `mul.1` | `[1,64,128]` × scalar `1.0` | Attention scaling prefill |
| `mul.2` | `[1,64,4096]` × `[1,64,1]` | rsqrt normalisation prefill |
| `mul.3` | `[4096]` × `[1,64,4096]` | Weight × hidden (broadcast) prefill |
| `mul.4` | `[1,32,64,128]` × `[1,1,64,128]` | Q × cos (broadcast) prefill |
| `mul.5` | `[1,8,64,128]` × `[1,1,64,128]` | K × cos (broadcast) prefill |
| `mul.6` | `[1,64,14336]` × `[1,64,14336]` | Gate × up (SwiGLU elementwise) prefill |
| `mul.7` | `[1,1,128]` × scalar `1.0` | Attention scaling decode |
| `mul.8` | `[1,1,4096]` × `[1,1,1]` | rsqrt normalisation decode |
| `mul.9` | `[4096]` × `[1,1,4096]` | Weight × hidden (broadcast) decode |
| `mul.10` | `[1,32,1,128]` × `[1,1,1,128]` | Q × cos (broadcast) decode |
| `mul.11` | `[1,8,1,128]` × `[1,1,1,128]` | K × cos (broadcast) decode |
| `mul.12` | `[1,1,14336]` × `[1,1,14336]` | Gate × up (SwiGLU elementwise) decode |

```bash
pytest test_llama_3_8b_instruct.py -m torch_mul
pytest test_llama_3_8b_instruct.py -m torch_mul -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_mul -k "compiled"
```

---

### `Tensor.view` — `TestView`

**Mark:** `torch_view`

Zero-copy reshape via `.view()`.

| Patterns | Shape | Description |
|---|---|---|
| 000 | `[1, 64, 4096] → (1, 64, -1, 128)` | Multi-head Q/K/V split prefill |
| 001 | `[1, 64, 1024] → (1, 64, -1, 128)` | GQA K/V split prefill |
| 002 | `[1, 1, 4096] → (1, 1, -1, 128)` | Multi-head Q/K/V split decode |
| 003 | `[1, 1, 1024] → (1, 1, -1, 128)` | GQA K/V split decode |

```bash
pytest test_llama_3_8b_instruct.py -m torch_view
pytest test_llama_3_8b_instruct.py -m torch_view -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_view -k "compiled"
```

---

### `Tensor.contiguous` — `TestContiguous`

**Mark:** `torch_contiguous`

Input shapes: `[1, 64, 32, 128]`, `[1, 1, 32, 128]`, `[1, 64, 4096]`,
`[1, 1, 4096]`. Sub-groups test shape preservation, value correctness on
already-contiguous inputs, correctness after a non-contiguous `transpose`
view (only dim pairs where both swapped sizes > 1 guarantee
non-contiguity), and dtype preservation.

```bash
pytest test_llama_3_8b_instruct.py -m torch_contiguous
pytest test_llama_3_8b_instruct.py -m torch_contiguous -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_contiguous -k "compiled"
```

---

### `torch.reshape` — `TestReshape`

**Mark:** `torch_reshape`

| Group | Pattern | Example |
|---|---|---|
| A | Attention output `[B,S,32,128] → [B,S,4096]` | Prefill and decode |
| L | Non-contiguous: `transpose → reshape` | Exact model path (line 173) |
| M | Full chain: `.reshape().contiguous()` | In-place materialisation |
| S | CPU-only contiguity assertion | Structural sanity check |

```bash
pytest test_llama_3_8b_instruct.py -m torch_reshape
pytest test_llama_3_8b_instruct.py -m torch_reshape -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_reshape -k "compiled"
```

---

### `torch.nn.functional.silu` — `TestFunctionalSilu`

**Mark:** `torch_nn_functional_silu`

`F.silu` appears in every FFN block as the SwiGLU gate activation:
`F.silu(gate_proj(x)) * up_proj(x)`. `intermediate_size = 14336`.

| Group | Pattern |
|---|---|
| A | FFN gate decode `[B, 1, 14336]` |
| B | FFN gate prefill `[B, S, 14336]` |
| D | Full SwiGLU product: `F.silu(gate) * up` |
| F | Non-contiguous input: `transpose → silu` |
| H | CPU-only identity: `F.silu(x) == x * sigmoid(x)` |

```bash
pytest test_llama_3_8b_instruct.py -m torch_nn_functional_silu
pytest test_llama_3_8b_instruct.py -m torch_nn_functional_silu -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_nn_functional_silu -k "compiled"
```

---

### `torch.nn.functional.embedding` — `TestEmbedding`

**Mark:** `torch_embedding`

| Parameter | Value |
|---|---|
| Weight shape | `[128256, 4096]` (`VOCAB_SIZE × HIDDEN_SIZE`) |
| Index dtype | `torch.int64` |
| Weight dtype | `torch.float16` |
| Index shapes | `[1, 64]` (prefill), `[1, 1]` (decode) |

Sub-groups test exact-trace prefill/decode, fill sweeps (zeros, ones),
index boundary (first and last vocab index), and strided vs contiguous inputs.

```bash
pytest test_llama_3_8b_instruct.py -m torch_embedding
pytest test_llama_3_8b_instruct.py -m torch_embedding -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_embedding -k "compiled"
```

---

### `torch.arange` — `TestArange`

**Mark:** `torch_arange`

Factory operation producing position IDs. Both CPU and Spyre receive
`device=` automatically (`needs_device=True`).

| Pattern | End | Output shape |
|---|---|---|
| Prefill | `64` | `(64,) int64` |
| Decode | `1` | `(1,) int64` |

```bash
pytest test_llama_3_8b_instruct.py -m torch_arange
pytest test_llama_3_8b_instruct.py -m torch_arange -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_arange -k "compiled"
```

---

### `torch.diff` — `TestDiff`

**Mark:** `torch_diff`

Present in prefill only — the causal-mask creation path is skipped in decode.

| Parameter | Value |
|---|---|
| Input shape | `[1, 64] int64` stride `(64, 1)` |
| Append shape | `[1, 1] int64` stride `(1, 1)` |
| dim | `-1` |
| Output shape | `[1, 64] int64` |

Sub-groups cover exact-trace prefill and strided vs contiguous input.

```bash
pytest test_llama_3_8b_instruct.py -m torch_diff
pytest test_llama_3_8b_instruct.py -m torch_diff -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_diff -k "compiled"
```

---

### `torch.__eq__` — `TestEq`

**Mark:** `torch_eq`

Present in prefill only. Tests `tensor == scalar` with a non-unit-stride
input reproducing the exact trace layout.

| Parameter | Value |
|---|---|
| Input shape | `(1,) int64` stride `(64,)` |
| Scalar | `0` |
| Output dtype | `bool` |

Sub-groups cover exact-trace, non-trivial stride assertion, and fill sweeps
(zeros → `True`, ones → `False`).

```bash
pytest test_llama_3_8b_instruct.py -m torch_eq
pytest test_llama_3_8b_instruct.py -m torch_eq -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_eq -k "compiled"
```

---

### `torch.all` — `TestAll`

**Mark:** `torch_all`

Present in prefill only. Produces a scalar `()` bool tensor.

| Parameter | Value |
|---|---|
| Input shape | `(1,) bool` stride `(1,)` |
| Output shape | `()` (scalar) |

Sub-groups cover exact-trace, True/False coverage, and scalar output shape
assertion.

```bash
pytest test_llama_3_8b_instruct.py -m torch_all
pytest test_llama_3_8b_instruct.py -m torch_all -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_all -k "compiled"
```

---

### `Tensor.float()` — `TestTensorFloat`

**Mark:** `torch_tensor_float`

Dtype cast shorthand method. Covers all four traced occurrences:

| Occurrence | Shape | Source → Target | Model usage |
|---|---|---|---|
| Prefill occ1 | `(1, 64, 1)` | `fp32 → fp32` | `inv_freq_expanded` cast |
| Prefill occ2 | `(1, 1, 64)` | `int64 → fp32` | `position_ids` promotion |
| Decode occ1 | `(1, 64, 1)` | `fp32 → fp32` | `inv_freq_expanded` cast |
| Decode occ2 | `(1, 1, 1)` | `int64 → fp32` | `position_ids` promotion |

```bash
pytest test_llama_3_8b_instruct.py -m torch_tensor_float
pytest test_llama_3_8b_instruct.py -m torch_tensor_float -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_tensor_float -k "compiled"
```

---

### `Tensor.expand` — `TestTensorExpand`

**Mark:** `torch_tensor_expand`

Zero-copy broadcast view via `.expand()`. Expands `inv_freq_expanded` along
the batch dimension: `inv_freq_expanded.expand(bs, -1, 1)`.

| Parameter | Value |
|---|---|
| Input shape | `(1, 64, 1)` stride `(64, 1, 1)` float32 |
| Expand sizes | `[1, -1, 1]` |
| Output shape | `(1, 64, 1)` |

Sub-groups cover exact-trace, strided vs contiguous, and output shape
assertion.

```bash
pytest test_llama_3_8b_instruct.py -m torch_tensor_expand
pytest test_llama_3_8b_instruct.py -m torch_tensor_expand -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_tensor_expand -k "compiled"
```

---

### `Tensor.to` — `TestTensorTo`

**Mark:** `torch_tensor_to`

`Tensor.to(dtype)` — covers dtype transfer across all six traced occurrences:

| Occurrence | Shape | Transfer | Model usage |
|---|---|---|---|
| Prefill occ1 | `(1, 64, 1)` | `fp32 → fp32` | `.to(x.device)` after inv_freq expand |
| Prefill occ2 | `(1, 64, 128)` | `fp32 → fp16` | `cos/sin.to(dtype=x.dtype)` |
| Prefill occ3 | `(1, 64, 4096)` | `fp16 → fp32` | `hidden_states.to(torch.float32)` RMSNorm |
| Decode occ1 | `(1, 64, 1)` | `fp32 → fp32` | `.to(x.device)` after inv_freq expand |
| Decode occ2 | `(1, 1, 128)` | `fp32 → fp16` | `cos/sin.to(dtype=x.dtype)` |
| Decode occ3 | `(1, 1, 4096)` | `fp16 → fp32` | `hidden_states.to(torch.float32)` RMSNorm |

```bash
pytest test_llama_3_8b_instruct.py -m torch_tensor_to
pytest test_llama_3_8b_instruct.py -m torch_tensor_to -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_tensor_to -k "compiled"
```

---

### `torch.sin` — `TestSin`

**Mark:** `torch_sin`

Pointwise sine from the RoPE rotary embedding path. Shapes `[1, 64, 128]`
(prefill) and `[1, 1, 128]` (decode). Also includes a numerical stability
variant with extreme large/small float32 values and a strided vs contiguous
check.

```bash
pytest test_llama_3_8b_instruct.py -m torch_sin
pytest test_llama_3_8b_instruct.py -m torch_sin -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_sin -k "compiled"
```

---

### `torch.nn.functional.linear` — `TestLinear`

**Mark:** `torch_linear`

`F.linear(input, weight, bias=None)` implements `y = x A^T`.

| Input shape | Weight shape | Layer |
|---|---|---|
| `[1, 64, 4096]` | `[4096, 4096]` | Q projection prefill |
| `[1, 64, 4096]` | `[1024, 4096]` | K/V projection prefill |
| `[1, 64, 4096]` | `[14336, 4096]` | Gate/Up FFN projection prefill |
| `[1, 64, 14336]` | `[4096, 14336]` | Down FFN projection prefill |
| `[1, 64, 4096]` | `[128256, 4096]` | LM head prefill |
| `[1, 1, 4096]` | `[4096, 4096]` | Q projection decode |
| *(+ decode variants for all projection layers above)* | | |

> **Note:** Tolerances are `atol=0.01, rtol=0.01` because large matrix
> multiplications accumulate `float16` rounding error beyond the default
> thresholds.

```bash
pytest test_llama_3_8b_instruct.py -m torch_linear
pytest test_llama_3_8b_instruct.py -m torch_linear -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_linear -k "compiled"
```

---

### `scaled_dot_product_attention` — `TestSDPA`

**Mark:** `torch_sdpa`

Tests run in both **eager** and **compiled** mode.

Observed shapes from op trace:

| Mode | Tensor | Shape | Stride | dtype |
|---|---|---|---|---|
| Prefill | Q | `(1, 32, 64, 128)` | `(262144, 128, 4096, 1)` | float16 |
| Prefill | K/V | `(1, 8, 64, 128)` | `(65536, 128, 1024, 1)` | float16 |
| Decode | Q | `(1, 32, 1, 128)` | `(4096, 128, 128, 1)` | float16 |
| Decode | K/V | `(1, 8, 65, 128)` | `(66560, 128, 8320, 1)` | float16 |

Sub-marks for each test scenario:

| Sub-mark | Base method | What it tests |
|---|---|---|
| `torch_sdpa_prefill_causal` | `test_sdpa_prefill_causal` | `is_causal=True`, varying batch and sequence lengths |
| `torch_sdpa_decode` | `test_sdpa_decode` | `seq_q=1`, no mask, varying KV-cache lengths |
| `torch_sdpa_growing_kvcache` | `test_sdpa_growing_kvcache` | Autoregressive growing KV cache `[1, 2, 4, ..., 2048]` |
| `torch_sdpa_causal_flag_vs_mask` | `test_sdpa_causal_flag_vs_mask` | `is_causal=True` must equal explicit causal mask |
| `torch_sdpa_weights_sum_to_one` | `test_sdpa_weights_sum_to_one` | Softmax rows must sum to 1 |
| `torch_sdpa_gqa_shape` | `test_sdpa_gqa_shape` | K/V shape after GQA expansion equals Q shape |
| `torch_sdpa_batch_consistency` | `test_sdpa_batch_consistency` | Identical batch items produce identical outputs |
| `torch_sdpa_gradient_flow` | `test_sdpa_gradient_flow` | Q gradient back-propagates without NaNs (eager only) |
| `torch_sdpa_determinism` | `test_sdpa_determinism` | Two identical calls return identical outputs |

```bash
pytest test_llama_3_8b_instruct.py -m torch_sdpa
pytest test_llama_3_8b_instruct.py -m torch_sdpa -k "eager"
pytest test_llama_3_8b_instruct.py -m torch_sdpa -k "compiled"
pytest test_llama_3_8b_instruct.py -m torch_sdpa_decode
pytest test_llama_3_8b_instruct.py -m torch_sdpa_prefill_causal
pytest test_llama_3_8b_instruct.py -m torch_sdpa_gradient_flow
```

---

## Combining Filters

`-m` selects by operator mark; `-k` filters by any substring of the
generated test method name. They compose freely:

```bash
# Eager path only for torch.mul
pytest test_llama_3_8b_instruct.py -m torch_mul -k "eager"

# Compiled path only for torch.add
pytest test_llama_3_8b_instruct.py -m torch_add -k "compiled"

# A single cat pattern
pytest test_llama_3_8b_instruct.py -m torch_cat -k "pattern_003"

# Multiple operators in one run
pytest test_llama_3_8b_instruct.py -m "torch_neg or torch_rsqrt or torch_pow"

# Everything except SDPA (faster CI pass)
pytest test_llama_3_8b_instruct.py -m "not torch_sdpa"

# Everything except SDPA and linear (skip the heaviest matmul patterns)
pytest test_llama_3_8b_instruct.py -m "not torch_sdpa and not torch_linear"

# All linear tests for the strided vs contiguous check
pytest test_llama_3_8b_instruct.py -m torch_linear -k "strided"

# All eager silu tests
pytest test_llama_3_8b_instruct.py -m torch_nn_functional_silu -k "eager"

# Stop at the first failure with a full traceback
pytest test_llama_3_8b_instruct.py -m torch_mul -x --tb=long

# Collect (list) tests without executing them
pytest test_llama_3_8b_instruct.py -m torch_view --collect-only
```

---

## Tolerances

`compare_with_cpu` selects tolerances automatically based on execution mode:

| Mode | `atol` | `rtol` | Constant |
|---|---|---|---|
| Eager (uncompiled) | `5e-3` | `5e-3` | `EAGER_ATOL` / `EAGER_RTOL` |
| Compiled (`torch.compile`) | `1e-1` | `1e-2` | `COMPILED_ATOL` / `COMPILED_RTOL` |

SDPA tests use dedicated per-mode tolerances defined on `TestSDPA`:

| Mode | `atol` | `rtol` |
|---|---|---|
| Eager | `2e-2` | `2e-2` |
| Compiled | `5e-2` | `5e-2` |
| Growing KV cache (eager) | `2e-2` | `2e-2` |

`TestLinear` uses relaxed tolerances (`atol=0.01, rtol=0.01`) because
accumulation of `float16` rounding error in large matrix multiplications
can exceed the default thresholds.

Compiled mode uses looser bounds because `torch.compile` may fuse or reorder
operations, introducing small additional rounding differences compared to
plain eager execution. **Failures exceeding `EAGER_ATOL` should be
investigated as backend bugs rather than addressed by widening the
tolerance.**

---

## Debugging Failures

When a test fails on Spyre, enable Dynamo's internal logging to get a full
backend trace:

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_llama_3_8b_instruct.py \
    -m "<mark>" -k "<filter>" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

| Flag / variable | Purpose |
|---|---|
| `TORCHDYNAMO_VERBOSE=1` | Prints the internal Dynamo stack trace (normally suppressed) |
| `TORCH_LOGS="+dynamo"` | Enables all Dynamo-level log output |
| `-m "<mark>"` | Restrict to one operator — e.g. `-m torch_cat` |
| `-k "<filter>"` | Narrow further — e.g. `-k "eager"` or `-k "pattern_003"` |
| `-x` | Stop at the first failure |
| `--tb=long` | Show the full Python traceback including local variable values |
| `-s` | Disable output capture so `print()` appears immediately |
| `2>&1` | Merge stderr (Dynamo logs) into stdout |
| `tee debug_output.txt` | Write everything to a file while still printing to the terminal |

**Example — debug an eager `torch.cat` failure:**

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_llama_3_8b_instruct.py \
    -m "torch_cat" -k "eager" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

**Example — debug a compiled `torch.linear` failure:**

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_llama_3_8b_instruct.py \
    -m "torch_linear" -k "compiled" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

**Search the captured log for the root cause:**

```bash
grep -A 10 "Error\|Exception\|FAILED" debug_output.txt
```

---

## Known Constraints

| Constraint | Detail |
|---|---|
| **SDPA gradient flow — eager only** | `test_sdpa_gradient_flow` has no `compiled=True` variant because the compiled backward pass is not supported on Spyre. |
| **SDPA weights sum to one — eager only** | `test_sdpa_weights_sum_to_one` is skipped in compiled mode; the manual softmax decomposition used for the row-sum check requires eager execution. |
| **SDPA determinism — eager only** | `test_sdpa_determinism` skips compiled mode to avoid recompilation overhead masking non-determinism. |
| **GQA `expand_kv` runs on CPU** | Head expansion via `repeat_interleave` is performed on CPU before moving tensors to Spyre, to avoid a crash inside the Spyre `maybe_get_squeezed_layout` allocator that affects `view` / `reshape` / `unsafe_view`. |
| **`torch.diff` — prefill only** | The causal-mask creation path that calls `torch.diff` is skipped during decode; no decode test exists. |
| **`torch.eq` / `torch.all` — prefill only** | Similarly, these ops appear exclusively in the prefill causal-mask creation path. |

---

## How Test Expansion Works — `ParameterizedTestMeta`

The `ParameterizedTestMeta` metaclass expands the `PARAMS` dict defined on
each `unittest.TestCase` subclass into individual `test_*` methods at class
creation time. `functools.wraps` is deliberately **not** used: it sets
`__wrapped__`, which causes pytest to misidentify each test's source location
and silently deselect it. `__name__` and `__qualname__` are set explicitly
instead.

**Expected `PARAMS` structure:**

```python
PARAMS = {
    # Key: (generated_test_prefix, base_method_name_in_class)
    ("test_my_op_pattern_000", "_run_my_op_test"): {
        # Optional: when present, tests are expanded as ops × cases.
        "ops_dict": {
            "op_name": op_callable,
            ...
        },
        "param_sets": {
            "case_name_eager":    (arg0, arg1, ..., False),
            "case_name_compiled": (arg0, arg1, ..., True),
            ...
        },
    },
    ...
}
```

**Pytest mark derivation** (applied when the class does not already define
`pytestmark`):

```
_run_cat_test              → @pytest.mark.torch_cat
_run_getitem_shape_test    → @pytest.mark.torch_getitem_shape
_run_transpose_values_test → @pytest.mark.torch_transpose_values
```

When `pytestmark` is set on the class (e.g. `pytest.mark.torch_sdpa`),
that mark stamps every method instead and no additional mark is derived.

---

## License

```
Copyright 2025 The Torch-Spyre Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
