# Lesson 16 — TensorRT INT4 Block Quantization: Q/DQ, Packing, and WoQ

> **Puzzle:** What must be present in a graph and serialized weight buffer before TensorRT can consume INT4 weights?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

TensorRT explicit quantization represents quantization choices with Q/DQ semantics and consumes packed low-bit weights plus scales under supported block/layout constraints.

Quick mental model:

- Explicit quantization represents scale decisions with Quantize/Dequantize semantics.
- Signed INT4 codes occupy two nibbles per byte when packed.
- TensorRT support has specific block-size and placement rules that a generic fake-quant experiment cannot prove.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

For signed INT4, two 4-bit two's-complement codes occupy one byte. Block Q/DQ applies one scale to a supported group, reconstructing floating-point values for the consuming operation or enabling a fused weight-only implementation.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

A valid packer can still produce an engine-incompatible graph; a valid graph can still select a slow tactic. Semantics, serialization, build, kernel selection, and runtime are separate gates.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Perform block INT4 Q/DQ and nibble packing on CUDA, verify exact unpacking, and separately probe the TensorRT package.

The CUDA lab validates block Q/DQ and exact nibble round-trip while an independent package probe prevents a false TensorRT-engine claim.

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

Packing correctness and Q/DQ error are real; engine build and latency remain unmeasured unless TensorRT is installed and executes.

Round-trip every packed code, verify scale axis/block size and ONNX Q/DQ placement, inspect the built engine, then benchmark the engine against the same baseline.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Block Q/DQ and nibble packing were validated; TensorRT engine execution was not inferred from the reference path.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Validate graph semantics, packing, scales, engine inspection, and timing as separate gates.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/16-tensorrt-int4/lab.ipynb
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
