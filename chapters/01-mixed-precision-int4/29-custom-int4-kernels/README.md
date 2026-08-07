# Lesson 29 — Custom Kernels: Packing, Dequantization, and CUTLASS Boundaries

> **Puzzle:** When is an INT4 pack/dequant kernel worth building instead of using an existing backend?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

An INT4 execution path contains pack/storage, scale loads, unpack/dequant, GEMM, epilogue, launches, and integration with framework layouts and streams.

Quick mental model:

- End-to-end gain includes unpack, scale loads, dequantization, GEMM, launch overhead, and integration cost.
- A Python or composed PyTorch prototype validates semantics but is not a fused CUTLASS kernel.
- The target shape distribution determines whether specialization pays off.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

The end-to-end budget is `T = T_pack/load + T_dequant + T_gemm + T_epilogue + overhead`. Fusing stages can remove intermediate traffic; a composed PyTorch reference intentionally exposes that unfused cost.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

A specialized CUTLASS/Triton kernel may win on stable shapes but costs engineering, testing, portability, and maintenance. Mature libraries remain the baseline to beat.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Validate vectorized INT4 nibble packing/unpacking and time the composed PyTorch dequantize-plus-matmul path against BF16.

The lab validates nibble semantics and times a composed unpack-dequant-matmul reference, labeling it explicitly as non-fused and non-CUTLASS.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`pytorch-gpu`**.

## 5. Inspect, accept, or roll back

Use the result to locate overhead, not to claim CUTLASS performance. A custom-kernel project begins only after a measured gap and stable shapes.

First locate a repeated shape-level gap, verify pack/dequant semantics, profile roofline and memory traffic, implement, then require end-to-end gain and quality across the target shape distribution.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** The composed reference exposed integration overhead; it is a semantic baseline, not a custom-kernel performance claim.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Build custom code when the existing backend misses an important, repeated shape and the recoverable end-to-end budget exceeds integration cost.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/29-custom-int4-kernels/lab.ipynb
```

Use **Run All**. Optional production backends are intentionally not hidden in
the base requirements; install the version appropriate for your GPU and follow
its official compatibility matrix before attempting a native path.

## Evidence boundary

- The checked-in notebook was executed on the GPU recorded inside the artifact;
  results on another GPU or software release may differ.
- Synthetic tensors isolate the mechanism and keep the lab downloadable. They
  do not establish full-model task quality or service throughput.
- Missing optional packages are recorded as `not_installed`, `failed`, or
  `not_measured`; no substitute backend is presented as native evidence.
- This is independently written tutorial material. It does not redistribute the
  source-course HTML, model weights, or private profiler traces.

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
