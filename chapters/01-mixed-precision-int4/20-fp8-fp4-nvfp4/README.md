# Lesson 20 — FP8, FP4, NVFP4, and Hardware Boundaries

> **Puzzle:** Does Blackwell hardware support mean every framework build exposes the same FP8 or NVFP4 path?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Keep four layers distinct: numerical format, hardware instruction, library recipe, and framework/operator API. `torch.float8_*` existing does not alone prove an FP8 GEMM path.

Quick mental model:

- A format definition, hardware instruction, library API, and framework kernel are four separate layers.
- FP8 variants trade exponent range against fraction precision.
- NVFP4 adds block scaling; it is not ordinary uniform INT4.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

E4M3 favors precision with less range; E5M2 favors range. Scaled FP8 matmul applies explicit scale factors. Blackwell-specific MXFP8/NVFP4 add block-scale structure and require matching recipes and kernels.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Smaller formats reduce traffic and raise theoretical throughput but add scale selection, saturation risk, metadata, and software compatibility constraints.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Attempt native PyTorch FP8 GEMM on the RTX GPU, record error and timing when supported, and separately probe Transformer Engine and NVFP4 APIs.

The lab calls PyTorch scaled FP8 matmul when available and leaves Transformer Engine/NVFP4 unmeasured rather than equating hardware generation with framework support.

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

A successful float8 PyTorch GEMM proves that path only. NVFP4 remains unmeasured without its library recipe and operator evidence.

Record compute capability, dtype/API, scaling recipe, operator success, numerical error, timing, and library version separately for FP8, MXFP8, and NVFP4.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Framework-level FP8 was tested independently; NVFP4 requires a supported library recipe and operator evidence.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Publish a format-by-hardware-by-library matrix, not a single `supported` checkbox.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/lab.ipynb
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

- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
