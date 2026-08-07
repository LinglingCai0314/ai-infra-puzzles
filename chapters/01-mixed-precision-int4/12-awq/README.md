# Lesson 12 — AWQ: Protecting Salient Weights in W4A16

> **Puzzle:** Can activation statistics tell us which weight channels deserve more protection?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

AWQ studies which weight channels are salient under observed activations and protects them within a weight-only W4A16 deployment path.

Quick mental model:

- AWQ identifies salient weights through activation-aware evidence.
- Equivalent scaling can move quantization difficulty while leaving the original floating-point function unchanged.
- W4A16 describes weight and activation precision; it does not mean the full graph is four-bit.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Channel scaling can preserve the floating-point linear transform while changing how weight ranges are shared before INT4 rounding. Activation statistics guide the scale search because frequently excited channels can amplify small weight errors.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Protecting more channels or searching more scales costs calibration time and may reduce compression or kernel regularity. Lower weight MAE does not guarantee lower language-model loss.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Search activation-aware per-channel scaling strengths for a toy W4A16 layer and compare output error with naive INT4.

The notebook freezes calibration activations, searches scaling strength, and chooses by held-out layer-output error rather than weight error.

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

Use held-out output error and a frozen search set. Weight-only mean error is not the optimization target.

Separate search/calibration from held-out evaluation, report protected fraction and group size, and prove a W4A16 operator executed before making speed claims.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Activation-aware scaling changed held-out W4A16 output error; no production AWQ kernel was claimed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Activation-aware protection is a model-quality method; deployment speed still requires a compatible W4A16 kernel.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/12-awq/lab.ipynb
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

- [AWQ paper](https://arxiv.org/abs/2306.00978)
