# Lesson 07 — Inference Precision Layers: Weights, Activations, and KV Cache

> **Puzzle:** When a model is called INT4, which tensors are actually four-bit?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Calling a model INT4 usually describes only part of its state. Weight-only layers may
store four-bit codes while activations and accumulators use BF16, the KV cache grows
with context, and temporary workspaces appear only at runtime. Capacity planning fails
when those objects are collapsed into one advertised precision.

## Predict before reading the result

1. Write the KV-cache byte formula before looking at the projected values.
2. Predict which memory account grows with sequence length and which stays fixed for a loaded model.
3. Explain why checkpoint size cannot predict peak CUDA allocation by itself.

## 1. Start from concrete tensors and state

Inference precision belongs to separate ledgers: persistent weights, per-step
activations/workspaces, accumulators, and persistent-per-request KV cache. Weight-only
INT4 normally leaves activation and accumulation formats wider.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Weight-only quantization leaves activations and accumulation in a floating-point compute dtype. |
| 2 | KV cache grows with layers, sequence length, key/value heads, head dimension, batch, and cache dtype. |
| 3 | Peak memory also includes temporary workspaces and allocator reserve. |

## 2. Derive the mechanism

For a standard cache, `bytes = 2 × layers × batch × sequence × kv_heads × head_dim ×
bytes_per_element`; the leading two is for keys and values. Grouped-query attention
changes `kv_heads`, not the number of query heads.

For a decoder cache with batch B, layers L, sequence S, KV heads H, head dimension D,
two tensors K and V, and b bytes per element, the leading storage is `2·B·L·S·H·D·b`.
Weight storage is roughly `parameters × effective bits/8` plus scales and unquantized
tensors. Activations depend on execution phase and liveness, while workspaces and
allocator reserve depend on backend behavior.

These terms have different lifetimes. Weights persist after load, KV cache persists per
active request, and many activations are temporary. That makes concurrency a
multiplication on the cache term, not on the model weights. The ledger must keep bytes,
lifecycle, and ownership together.

## 3. Translate the theory into an experiment

**Experiment:** Build a memory ledger and allocate representative BF16 and INT8 KV tensors on CUDA to validate element-count arithmetic.

| Experimental role | Frozen definition |
|---|---|
| Baseline | BF16 KV-cache projection and a real BF16 K/V allocation |
| Candidate | INT8 cache projection for the same model geometry |
| Held constant | batch 1, 32 layers, 8 KV heads, head dimension 128, identical context lengths |
| Measurements | projected cache GiB by context and byte count of an allocated representative tensor pair |
| Evidence label | `pytorch-gpu` |

The lab validates the KV element-count formula with a live allocation and projects
several context lengths without pretending to allocate a full model.

### Code walk-through

The notebook first calculates the formula for three sequence lengths, then allocates
representative K and V tensors on CUDA and checks their exact element-count bytes. This
joins arithmetic with a live tensor object without pretending to load a full model.

Scales, paging fragmentation, prefix-cache blocks, and temporary attention workspaces
are intentionally outside the simple projection. They belong in the next ledger revision
when a named serving backend is tested.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| BF16 KV at 2,048 tokens | 0.250 GiB |
| BF16 KV at 8,192 tokens | 1.000 GiB |
| BF16 KV at 32,768 tokens | 4.000 GiB |
| INT8 KV at 32,768 tokens | 2.000 GiB |
| Live allocation probe | 16,777,216 bytes |

### What the numbers mean

For the fixed 32-layer geometry, projected BF16 KV storage was 0.25 GiB at 2,048 tokens,
1.0 GiB at 8,192, and 4.0 GiB at 32,768. The INT8 arithmetic projection was exactly half
each value. The live probe allocated two BF16 tensors of shape `[2, 4096, 8, 128]`
totaling 16,777,216 bytes.

The linear fourfold growth from 8K to 32K is the important systems result. Weight
quantization does not change it. Cache quantization may increase feasible context or
concurrency, but only after scale overhead, attention compatibility, error, and latency
are measured.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Name the object and lifecycle whenever you name a precision: weights, activations, accumulators, or cache.

### Acceptance and rollback gate

Measure allocated/reserved/peak memory separately and reconcile them with object-level
arithmetic. A checkpoint byte count is not a runtime memory result.

### How this conclusion can fail

A common error is multiplying weight memory by request count or forgetting to multiply
cache by layers and by both K and V. Another is treating free memory reported before
model load as deployable capacity. Allocator reserve, CUDA graphs, kernels, and safety
margin must be added before setting concurrency.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map BF16 KV-cache projection and a real BF16 K/V
allocation and INT8 cache projection for the same model geometry back to the derivation.
Verify the printed environment, then check that batch 1, 32 layers, 8 KV heads, head
dimension 128, identical context lengths stayed fixed. Read projected cache GiB by
context and byte count of an allocated representative tensor pair before applying the
acceptance gate; the artifact-writing cell retains the complete structured result from
the recorded run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/07-inference-precision-layers/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Extend the ledger with grouped-query attention variants, tensor parallel sharding, cache
block size, scale metadata, and allocator fragmentation. Then run a vLLM or TensorRT-LLM
server and compare predicted versus observed cache capacity at 2K, 8K, and 32K contexts.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 07's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM cache configuration](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
