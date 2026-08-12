# Lesson 04 — Why BF16 Is Often the First Low-Precision Choice

> **Puzzle:** FP16 and BF16 both use 16 bits. Why can their numerical behavior differ dramatically?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

FP16 and BF16 consume the same two bytes, but they spend those bits differently. BF16
inherits FP32's eight-bit exponent and sacrifices fraction precision; FP16 keeps a
longer fraction but only five exponent bits. That trade changes where overflow occurs
and how much rounding error accumulates, so a format decision cannot be made from byte
count alone.

## Predict before reading the result

1. Predict which 16-bit format represents `1e5` without Inf and which produces the lower GEMM error.
2. Predict whether equal storage implies equal GEMM latency on this GPU.
3. Decide which metric would make you choose FP16 despite BF16's wider range.

## 1. Start from concrete tensors and state

FP16 and BF16 both occupy 16 bits, but FP16 uses 5 exponent and 10 fraction bits whereas
BF16 uses 8 exponent and 7 fraction bits. The former offers finer local spacing; the
latter offers a much larger dynamic range.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | BF16 keeps an eight-bit exponent, so its range resembles FP32 while its fraction is shorter. |
| 2 | FP16 has more fraction bits but a much smaller exponent range. |
| 3 | A stable dtype is not automatically the fastest dtype; measure the actual workload. |

## 2. Derive the mechanism

Rounding error is governed by representable spacing near a value, while overflow is
governed by exponent range. Accumulation policy adds a third variable: low-precision
inputs may still accumulate into a wider type depending on the operator.

A normalized binary floating-point value has the form `(-1)^s × 2^e × (1.f)`. Exponent
bits determine dynamic range; fraction bits determine spacing between adjacent
representable numbers at a fixed exponent. BF16's range is close to FP32, but its seven
stored fraction bits make unit-roundoff much larger than FP16's ten. In a dot product,
inputs are rounded before multiplication and partial sums may use a wider accumulator,
so input format and accumulation format must be named separately.

This predicts a three-way trade: BF16 should survive large magnitudes, FP16 should often
reconstruct ordinary-range values more accurately, and either 16-bit format may use a
faster matrix path than FP32. The benchmark tests each axis independently instead of
collapsing them into one winner.

## 3. Translate the theory into an experiment

**Experiment:** Compare range, matrix-multiplication error, and CUDA timing for FP32, FP16, and BF16.

| Experimental role | Frozen definition |
|---|---|
| Baseline | FP32 GEMM and FP32 reference output |
| Candidate | FP16 and BF16 GEMMs on the same 1536×1536 matrices |
| Held constant | shape, random source values, GPU, warm-up, repetitions, comparison reference |
| Measurements | finite-range probe, RMSE/cosine error, median and p90 latency |
| Evidence label | `pytorch-gpu` |

The lab separates large-value representability, GEMM error, and GEMM latency into three
observations so one does not stand in for the others.

### Code walk-through

The range probe casts `1e5` into both 16-bit formats and records finiteness. The GEMM
probe uses the same logical matrices, evaluates output error against FP32, and times
each path with twelve post-warm-up CUDA-event samples. Keeping the error and timing
records side by side prevents a fast but numerically invalid path from looking
successful.

Because the tensors are random and the shape is one square GEMM, the result is a format
demonstration rather than a universal training recommendation. Real networks can amplify
rounding through normalization, softmax, optimizer state, and long reductions.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| FP16 represents 1e5 | no |
| BF16 represents 1e5 | yes |
| FP16 RMSE | 0.014106 |
| BF16 RMSE | 0.112772 |
| FP16 median | 0.044608 ms |
| BF16 median | 0.044160 ms |
| FP32 median | 0.133376 ms |

### What the numbers mean

BF16 represented `1e5` while FP16 overflowed; the recorded maximum finite values were
approximately `3.39e38` and `65504`. On the ordinary-range GEMM, FP16 had lower RMSE
(0.014106) than BF16 (0.112772), exactly the fraction-bit trade predicted by the format
layouts. Median latency was nearly tied—0.044608 ms for FP16 and 0.044160 ms for
BF16—while FP32 took 0.133376 ms.

The evidence supports BF16 as a stability-first default for wide-range workloads, not as
an accuracy or speed winner in every column. FP16 remained more precise for this input
distribution and equally fast within the measured spread.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> BF16 is a pragmatic stability-first baseline on supported hardware, but workload-specific error and speed still need measurement.

### Acceptance and rollback gate

Test both a range probe and workload output error against FP32, then measure latency on
the target shape. Keep BF16 only when stability and performance meet the frozen
thresholds.

### How this conclusion can fail

Selecting BF16 solely because it did not overflow can hide unacceptable rounding error;
selecting FP16 solely for lower RMSE can fail as soon as activations exceed its range.
Another failure is to assume the accumulator shares the input dtype. Record autocast
policy and operator behavior when reduction accuracy matters.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/04-bf16-first/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Repeat the experiment after scaling inputs across several orders of magnitude and add
long reductions, softmax, and layer normalization. For training, compare loss curves and
gradient-finiteness rates rather than one GEMM. A useful decision chart marks the
magnitude range where FP16 first fails and the error tolerance where BF16 becomes
unacceptable.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [PyTorch tensor attributes and dtypes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
