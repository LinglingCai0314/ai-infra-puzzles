# Lesson 10 — INT8 SmoothQuant and Activation Outliers

> **Puzzle:** Can we make activations easier to quantize without changing the floating-point linear layer?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

SmoothQuant operates on matching input channels of activation `X` and weight `W` for a linear layer `Y=XWᵀ`.

Quick mental model:

- SmoothQuant applies reciprocal channel scaling to activations and weights, preserving the floating-point product.
- The alpha parameter allocates quantization difficulty between activation and weight channels.
- The best alpha depends on observed activation and weight ranges.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

For positive channel scales `s`, `(X / s)(W · s)ᵀ = XWᵀ`. Choosing `s_j` from activation and weight maxima moves channel difficulty without changing the floating-point function. The exponent `alpha` decides how much range moves toward weights.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Activation ranges become easier for INT8 while weight ranges become harder. The correct objective is combined W8A8 output error and backend performance, not activation amax alone.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Apply SmoothQuant-style channel scaling to an outlier-heavy linear layer, verify floating-point equivalence, and compare W8A8 reconstruction error over alpha values.

The notebook checks the algebraic invariant before quantizing both sides and comparing output error across alpha values.

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

First verify algebraic equivalence; then compare quantized output error. A lower activation range alone is incomplete evidence.

Verify floating-point equivalence first, freeze calibration statistics, sweep alpha on calibration data, and accept using held-out output/quality plus native W8A8 evidence.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Reciprocal scaling preserved the floating-point layer while changing combined W8A8 error.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Outlier migration is useful only when the combined activation-plus-weight quantized path improves under a frozen calibration protocol.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/10-smoothquant/lab.ipynb
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

- [SmoothQuant paper](https://arxiv.org/abs/2211.10438)
