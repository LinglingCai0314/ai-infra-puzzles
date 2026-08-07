# Lesson 05 — Diagnosing FP16 Overflow and Gradient Scaling Failures

> **Puzzle:** When loss becomes NaN, how do we distinguish forward overflow, backward overflow, and gradient underflow?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Diagnose four checkpoints: forward outputs, scaled loss/gradients, unscaled gradients, and post-step parameters. A final NaN has already discarded the location of the first failure.

Quick mental model:

- Overflow creates Inf before it becomes NaN in later arithmetic.
- Underflow silently rounds small gradients to zero.
- Loss scaling moves gradients into a representable interval but cannot repair an already-overflowed forward pass.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

FP16 normal values end near `6.55e4`; very small values enter a sparse subnormal region and can become zero. Loss scaling shifts gradient magnitudes upward during storage, but unscaling must happen before clipping and parameter updates.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

An aggressive scale protects small gradients but increases overflow risk. A conservative scale avoids Inf yet may leave many gradients at zero, so the useful interval is workload-dependent.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Sweep synthetic gradient magnitudes and loss scales in FP16 on CUDA, counting finite, infinite, and zero gradient values.

The CUDA sweep crosses both tiny and large magnitudes at several scales and records zero and Inf fractions, making the failure stage observable.

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

The useful evidence is the first stage where finiteness changes. A final NaN without intermediate checks is not a diagnosis.

Log finite/Inf/zero fractions and the current scale. If the forward pass is already non-finite, change the operation or dtype; if only scaled gradients overflow, adjust scale policy.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Scaling changed gradient representability but could not repair an FP16 value that had already overflowed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Place finiteness and zero-rate probes at forward outputs, scaled gradients, unscaled gradients, and parameters before changing the scaler policy.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/05-fp16-overflow/lab.ipynb
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

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
