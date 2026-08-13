<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 09 — PTQ Calibration Data: Sampling and Coverage

> **Puzzle:** Can a small calibration set represent the activation ranges that production traffic will exercise?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Post-training quantization freezes scales from examples. If those examples omit long
prompts, a rare domain, or activation outliers, the quantizer can look excellent on
calibration data and clip production traffic. Calibration quality is therefore a
coverage problem before it is a sample-count problem.

## Predict before reading the result

1. Predict which calibration set minimizes clipping and which minimizes average rounding error on the mixed held-out set.
2. Explain why evaluation data must remain separate after scale selection.
3. List deployment strata that random sampling might under-represent.

## 1. Start from concrete tensors and state

A PTQ pipeline has a calibration distribution used to freeze quantization parameters and
a disjoint evaluation distribution used to test the frozen result.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Calibration estimates ranges or statistics; evaluation tests the frozen decision on held-out data. |
| 2 | Rare domains and long sequences can dominate worst-case activation ranges. |
| 3 | More samples do not help if sampling repeats the same narrow distribution. |

## 2. Derive the mechanism

Max calibration protects observed extremes but can waste most codes; percentile or
learned clipping trades a controlled tail for smaller steps. Either choice fails when
the calibration set omits a deployment domain.

A max-range calibrator chooses `s=max(|x_cal|)/qmax`; a percentile calibrator
deliberately clips a tail to shrink the step. Both estimate a property of the
calibration distribution. Generalization fails when the deployment distribution has
larger or differently located tails. More copies of the same narrow prompts reduce
estimator noise but do not reduce distribution bias.

The held-out clipping fraction measures values outside the frozen representable range.
RMSE measures the combined cost of clipped tails and quantization steps. Those
objectives can disagree: an outlier-aware scale can avoid clipping yet waste resolution
on most ordinary values.

## 3. Translate the theory into an experiment

**Experiment:** Calibrate INT8 activation scales on narrow, balanced, and outlier-aware synthetic datasets, then evaluate all scales on a mixed held-out distribution.

| Experimental role | Frozen definition |
|---|---|
| Baseline | scales frozen from a narrow synthetic calibration distribution |
| Candidate | balanced and explicitly outlier-aware calibration sets |
| Held constant | INT8 formula, held-out mixed tensor, evaluation metrics, seed |
| Measurements | frozen scale, held-out clipping fraction, RMSE, MAE, cosine, max error |
| Evidence label | `numerical-model` |

The lab freezes scales from narrow, balanced, and outlier-aware sets and evaluates all
three on one mixed held-out tensor.

### Code walk-through

The notebook creates three calibration populations, freezes one scale from each, and
evaluates all of them on the same mixed held-out tensor. It never recomputes a scale on
evaluation data. That makes the comparison a small distribution-shift test rather than a
reconstruction demo.

The examples are synthetic so domain labels are controllable. A model study would
replace them with stratified prompts and layer activation captures while preserving the
same calibration/evaluation separation.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Narrow clipping | 2.6478% |
| Narrow RMSE | 0.317395 |
| Balanced clipping | 0.0250% |
| Balanced RMSE | 0.085046 |
| Outlier-aware clipping | 0.0000% |
| Outlier-aware RMSE | 0.077629 |

### What the numbers mean

The narrow scale clipped 2.647752% of held-out values and produced RMSE 0.317395 with a
max error of 27.9039. Balanced calibration reduced clipping to 0.025001% and RMSE to
0.085046. Outlier-aware calibration eliminated clipping, but its larger scale raised MAE
to 0.067231; its RMSE, 0.077629, remained slightly better because it avoided
catastrophic tail errors.

There is no universally best row without a deployment objective. If tail failures are
unacceptable, the outlier-aware scale wins this probe. If average small-value resolution
dominates, a clipped or mixed policy may be preferable.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Choose calibration data by coverage of deployment modes, and keep it separate from the regression set.

### Acceptance and rollback gate

Publish sampling rules, lengths/domains, seed, statistic, sample count, and held-out
clipping/error. Never tune the range on the same examples used for the final quality
gate.

### How this conclusion can fail

Tuning percentiles on the final regression set leaks evaluation into calibration.
Reporting only mean error can hide rare catastrophic clipping, while reporting only max
error can let one outlier consume the entire code range. Coverage metadata—domain,
length, language, tool use, and frequency—is part of the quantization artifact.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/09-ptq-calibration/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Build a stratified calibration manifest for real prompts and compare random, balanced,
and tail-oversampled selections at fixed sample count. Evaluate per-layer clipping and
task slices on a disjoint set, then test whether the selected scale policy remains
stable across model revisions.

## Evidence boundary

**Evidence label:** [`numerical-model`](../README.md#evidence-labels).

## References

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [NVIDIA Model Optimizer PTQ documentation](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
- [TensorRT quantization workflows](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)
