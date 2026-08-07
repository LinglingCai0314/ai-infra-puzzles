# Lesson 29 — Custom Kernels: Packing, Dequantization, and CUTLASS Boundaries

> **Puzzle:** When is an INT4 pack/dequant kernel worth building instead of using an existing backend?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A custom INT4 kernel earns its complexity only if it removes work from the end-to-end
path. Packing weights is helpful, but materializing a full dequantized matrix before
calling BF16 GEMM adds reads, writes, conversions, and launches. The target is a fused
load–unpack–scale–MMA–epilogue path with a supported tile layout.

## Predict before reading the result

1. Calculate packed bytes for a 4096×4096 INT4 weight matrix.
2. Predict the latency of a composed unpack/dequant/GEMM path relative to direct BF16 GEMM at M=32.
3. Name the evidence needed before calling an implementation a CUTLASS or custom-kernel result.

## 1. Start from concrete tensors and state

An INT4 execution path contains pack/storage, scale loads, unpack/dequant, GEMM,
epilogue, launches, and integration with framework layouts and streams.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | End-to-end gain includes unpack, scale loads, dequantization, GEMM, launch overhead, and integration cost. |
| 2 | A Python or composed PyTorch prototype validates semantics but is not a fused CUTLASS kernel. |
| 3 | The target shape distribution determines whether specialization pays off. |

## 2. Derive the mechanism

The end-to-end budget is `T = T_pack/load + T_dequant + T_gemm + T_epilogue + overhead`.
Fusing stages can remove intermediate traffic; a composed PyTorch reference
intentionally exposes that unfused cost.

The logical pipeline is packed global load → nibble extraction/sign extension → scale
load → dequantized fragments → matrix multiply/accumulate → epilogue. If dequantization
writes a full BF16 matrix to global memory, the path pays both packed reads and a large
materialized write/read before GEMM. Fusion keeps reconstructed values in
registers/fragments and amortizes scale work across a tile.

Kernel profitability depends on M, N, K, group size, memory coalescing, register
pressure, occupancy, and epilogue fusion. A semantic PyTorch composition is a
correctness baseline and an upper-bound warning, not a custom kernel.

## 3. Translate the theory into an experiment

**Experiment:** Validate vectorized INT4 nibble packing/unpacking and time the composed PyTorch dequantize-plus-matmul path against BF16.

| Experimental role | Frozen definition |
|---|---|
| Baseline | direct BF16 GEMM for shape M=32, K=N=4096 |
| Candidate | PyTorch-composed unpack, sign restore, dequantization, and GEMM |
| Held constant | same X/W values, group size 128, packed layout, GPU timing helper |
| Measurements | packed bytes, BF16 median/p90, composed median/p90, implementation identity |
| Evidence label | `pytorch-gpu` |

The lab validates nibble semantics and times a composed unpack-dequant-matmul reference,
labeling it explicitly as non-fused and non-CUTLASS.

### Code walk-through

The notebook packs two codes per byte, reconstructs signed codes, applies block scales,
materializes BF16 weights, and multiplies. It times the complete composed function
rather than timing only the final GEMM. The result field explicitly says `not fused
CUTLASS`.

This gives a readable semantic reference for testing a future CUDA/Triton/CUTLASS
implementation. The future kernel must match its outputs while eliminating
materialization and reducing launches.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Shape M×K×N | 32 × 4096 × 4096 |
| Packed code bytes | 8,388,608 bytes |
| BF16 median | 0.027136 ms |
| Composed path median | 0.328720 ms |
| Implementation | composed PyTorch reference, not fused CUTLASS |

### What the numbers mean

Packed storage was 8,388,608 bytes for 16,777,216 weights, exactly 0.5 byte per code
before scales. Direct BF16 GEMM took 0.027136 ms median. The composed
unpack/dequant/matmul path took 0.328720 ms—about 12.1x slower.

The slowdown is not evidence that INT4 hardware is slow. It is evidence that the unfused
reference performs too much integration work and memory traffic. It establishes the
optimization target and a correctness oracle.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Build custom code when the existing backend misses an important, repeated shape and the recoverable end-to-end budget exceeds integration cost.

### Acceptance and rollback gate

First locate a repeated shape-level gap, verify pack/dequant semantics, profile roofline
and memory traffic, implement, then require end-to-end gain and quality across the
target shape distribution.

### How this conclusion can fail

Timing only GEMM after pre-dequantizing outside the measurement hides the dominant cost.
Calling a Python composition a custom kernel is false. A fused kernel can also regress
if register pressure lowers occupancy or if unsupported shapes fall back, so shape
coverage and dispatch must be audited.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map direct BF16 GEMM for shape M=32, K=N=4096 and
PyTorch-composed unpack, sign restore, dequantization, and GEMM back to the derivation.
Verify the printed environment, then check that same X/W values, group size 128, packed
layout, GPU timing helper stayed fixed. Read packed bytes, BF16 median/p90, composed
median/p90, implementation identity before applying the acceptance gate; the
artifact-writing cell retains the complete structured result from the recorded run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/29-custom-int4-kernels/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Implement a minimal fused kernel in CUTLASS, CUDA, or Triton for one frozen shape.
Verify packed-layout compatibility and numerical parity, then profile instruction mix,
global bytes, occupancy, tensor-pipe utilization, and end-to-end latency across M
values. Add a safe fallback for unsupported shapes.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 29's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [CUTLASS repository](https://github.com/NVIDIA/cutlass)
