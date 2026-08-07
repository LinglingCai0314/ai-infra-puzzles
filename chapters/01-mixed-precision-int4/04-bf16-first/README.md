# Lesson 04 — Why BF16 Is Often the First Low-Precision Choice

> **Puzzle:** FP16 and BF16 both use 16 bits. Why can their numerical behavior differ dramatically?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

FP16 and BF16 both occupy 16 bits, but FP16 uses 5 exponent and 10 fraction bits whereas BF16 uses 8 exponent and 7 fraction bits. The former offers finer local spacing; the latter offers a much larger dynamic range.

Quick mental model:

- BF16 keeps an eight-bit exponent, so its range resembles FP32 while its fraction is shorter.
- FP16 has more fraction bits but a much smaller exponent range.
- A stable dtype is not automatically the fastest dtype; measure the actual workload.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Rounding error is governed by representable spacing near a value, while overflow is governed by exponent range. Accumulation policy adds a third variable: low-precision inputs may still accumulate into a wider type depending on the operator.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

BF16 can avoid FP16 overflow but may show larger rounding error on well-scaled values. FP32 is a useful numerical reference, not automatically the production throughput winner.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Compare range, matrix-multiplication error, and CUDA timing for FP32, FP16, and BF16.

The lab separates large-value representability, GEMM error, and GEMM latency into three observations so one does not stand in for the others.

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

Look separately at overflow behavior, error against FP32, and latency. No single column decides every workload.

Test both a range probe and workload output error against FP32, then measure latency on the target shape. Keep BF16 only when stability and performance meet the frozen thresholds.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** BF16 preserved the large-value range while FP16 and BF16 showed different accuracy/performance trade-offs.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

BF16 is a pragmatic stability-first baseline on supported hardware, but workload-specific error and speed still need measurement.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/04-bf16-first/lab.ipynb
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
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
