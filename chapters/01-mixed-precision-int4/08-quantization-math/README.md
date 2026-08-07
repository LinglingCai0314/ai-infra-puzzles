# Lesson 08 — Quantization Math: Scale, Zero Point, Group Size, and Error

> **Puzzle:** Why does changing group size alter both model size and reconstruction error?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Uniform quantization stores integer codes plus scale metadata and, for asymmetric schemes, zero points. Granularity may be per tensor, row/channel, or group/block.

Quick mental model:

- Scale maps a floating-point interval to a finite code range.
- Symmetric quantization fixes zero point at zero; asymmetric quantization can spend codes more efficiently on shifted data.
- Smaller groups adapt to local ranges but require more scale metadata.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

A common mapping is `q = clamp(round(x/s)+z, qmin, qmax)` and `x_hat = s(q-z)`. Symmetric INT4 typically uses `z=0` and a signed range near `[-8,7]`. Smaller groups estimate local ranges and reduce outlier sharing.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Smaller groups add scale loads and metadata and may miss a backend's supported block sizes. Larger groups are cheaper but one outlier can enlarge the step for many ordinary weights.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Quantize an outlier-containing matrix with INT4 group sizes 16, 64, and 128 and compare error plus metadata overhead.

The notebook holds the weight matrix fixed, changes only group size, and records both error and effective bits per weight.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`numerical-model`**.

## 5. Inspect, accept, or roll back

Check saturation, error, and effective bits per value. Do not report the nominal four bits without scale overhead.

Report nominal bits, scale/zero-point overhead, clipping rate, reconstruction error, group axis, and kernel-compatible group size together.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Smaller groups reduced local range sharing at the cost of more scale metadata.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Group size is an error–metadata–kernel compatibility decision, not a cosmetic configuration value.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/08-quantization-math/lab.ipynb
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

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
