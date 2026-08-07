# Lesson 09 — PTQ Calibration Data: Sampling and Coverage

> **Puzzle:** Can a small calibration set represent the activation ranges that production traffic will exercise?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A PTQ pipeline has a calibration distribution used to freeze quantization parameters and a disjoint evaluation distribution used to test the frozen result.

Quick mental model:

- Calibration estimates ranges or statistics; evaluation tests the frozen decision on held-out data.
- Rare domains and long sequences can dominate worst-case activation ranges.
- More samples do not help if sampling repeats the same narrow distribution.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Max calibration protects observed extremes but can waste most codes; percentile or learned clipping trades a controlled tail for smaller steps. Either choice fails when the calibration set omits a deployment domain.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

More examples reduce sampling noise only when they add coverage. Long prompts, code, multilingual text, tool schemas, and rare outliers may need explicit strata rather than random repetition.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Calibrate INT8 activation scales on narrow, balanced, and outlier-aware synthetic datasets, then evaluate all scales on a mixed held-out distribution.

The lab freezes scales from narrow, balanced, and outlier-aware sets and evaluates all three on one mixed held-out tensor.

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

Compare held-out clipping rate and error, not calibration-set reconstruction error.

Publish sampling rules, lengths/domains, seed, statistic, sample count, and held-out clipping/error. Never tune the range on the same examples used for the final quality gate.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Held-out coverage, not calibration reconstruction, determined clipping and error.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Choose calibration data by coverage of deployment modes, and keep it separate from the regression set.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/09-ptq-calibration/lab.ipynb
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
