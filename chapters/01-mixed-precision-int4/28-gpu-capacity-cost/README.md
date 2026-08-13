<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 28 — GPU Memory, Concurrency, and Cost Estimation

> **Puzzle:** How many requests fit after INT4 weight compression, and which hidden assumptions can invalidate that number?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Cloud cost begins with a memory feasibility ledger, but it cannot end there. Ideal
weight bits, unquantized layers, scale metadata, KV cache per request, workspace,
fragmentation, tensor parallelism, throughput, utilization, and hourly price all
determine whether one GPU is usable and economical.

## Predict before reading the result

1. Estimate ideal BF16 and INT4 weight GiB for 70B parameters.
2. Compute one-request KV cache for 80 layers, 8 KV heads, dimension 128, and 8K context.
3. Predict whether ideal INT4 weights fit a 32,607 MiB RTX 5090 after a 10% reserve.

## 1. Start from concrete tensors and state

Capacity uses total/usable HBM, weight and scale bytes, runtime reserve, workspaces, KV
per request, fragmentation, tensor parallelism, and traffic context distribution.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Capacity starts from usable memory after runtime reserve, weights, workspaces, and fragmentation allowance. |
| 2 | Per-request KV cache depends on context and cache dtype. |
| 3 | Cost per token also depends on achieved throughput and utilization, not GPU price alone. |

## 2. Derive the mechanism

A first bound is `requests = floor((usable - weights - workspace) / KV_per_request)`.
Cost per token then depends on hourly price divided by achieved, quality-approved tokens
per hour.

Weight bytes start at `P·bits/8`. KV bytes per request are `2·L·S·Hkv·D·cache_bytes`,
then concurrency multiplies that term. A safety reserve should cover kernels, graph
capture, allocator behavior, and unexpected peaks before dividing remaining bytes by
per-request cache.

Even a memory fit does not produce a cost result. Cost per million tokens depends on
achieved tokens/s, utilization, batching, power/cloud price, failure rate, and replica
count. The notebook intentionally stops at arithmetic capacity when no engine throughput
exists.

## 3. Translate the theory into an experiment

**Experiment:** Read live free memory from the RTX GPU and build BF16 versus INT4 capacity projections for a 70B-class model without allocating the model.

| Experimental role | Frozen definition |
|---|---|
| Baseline | 70B BF16 weights with BF16 KV cache |
| Candidate | ideal INT4 weights with BF16 or INT8 KV cache |
| Held constant | 70B parameters, 80 layers, 8 KV heads, head dimension 128, context 8192, 10% reserve |
| Measurements | live total/free GiB, weight GiB, KV GiB/request, fit boolean, projected request count |
| Evidence label | `capacity-model` |

The lab seeds a 70B arithmetic model with live RTX 5090 memory but explicitly does not
allocate or benchmark a 70B model.

### Code walk-through

The notebook reads live RTX 5090 memory, calculates three plans, reserves 10%, and only
then computes request capacity. It records zero rather than a negative or optimistic
concurrency when weights already exceed usable memory.

The INT4 term is explicitly ideal: it excludes scales, padding, embeddings/norms
retained in higher precision, engine, and workspace. That label prevents the arithmetic
from being mistaken for a successful model load.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Live total memory | 31.358 GiB |
| BF16 weight projection | 130.385 GiB |
| Ideal INT4 weight projection | 32.596 GiB |
| BF16 KV per request | 2.500 GiB |
| INT8 KV per request | 1.250 GiB |
| Ideal INT4 single-GPU fit | no |

### What the numbers mean

Live total memory was 31.358 GiB. BF16 weights projected to 130.385 GiB; ideal INT4
still required 32.596 GiB, already larger than total memory and larger still relative to
the 10% reserve. BF16 KV cache was 2.5 GiB/request and INT8 KV 1.25 GiB/request, but
every single-GPU plan correctly returned zero requests because weights did not fit.

KV compression cannot rescue a base model that fails the weight-fit gate. A real 70B
deployment therefore needs further compression/overhead reduction, multi-GPU sharding,
CPU offload, or a different GPU class before concurrency is discussed.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Use ranges and safety margins, then validate the chosen point with the actual engine and traffic distribution.

### Acceptance and rollback gate

Use ranges and safety margins, then validate with the actual engine's measured peak,
sustained concurrency, SLO, utilization, and cloud billing unit.

### How this conclusion can fail

Using decimal GB instead of binary GiB can create misleading margin near capacity. Ideal
four-bit arithmetic omits metadata and high-precision tensors, and free memory on an
otherwise empty process is not engine capacity. Cost comparisons without throughput and
quality at equal SLO are also meaningless.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/28-gpu-capacity-cost/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Add measured overhead from a real engine, tensor-parallel sharding/communication,
fragmentation, and batch-dependent workspaces. Once the model loads, benchmark sustained
tokens/s and compute cost per million tokens at equal quality and p95 latency across
candidate GPU plans.

## Evidence boundary

**Evidence label:** [`capacity-model`](../README.md#evidence-labels).

## References

- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM cache configuration](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
