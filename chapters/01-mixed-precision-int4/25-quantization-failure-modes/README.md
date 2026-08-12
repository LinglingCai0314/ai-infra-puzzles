# Lesson 25 — Failure Modes: Outliers, Long Context, MoE, and Small Batches

> **Puzzle:** Where should a quantized system be expected to fail first?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Quantization often fails by regime rather than on average. Activation outliers amplify
weight error, a shifted domain changes channel importance, tiny batches expose
launch/dequant overhead, long context expands cache pressure, and MoE routing
concentrates work unevenly. A failure matrix makes those reversals visible before
production does.

## Predict before reading the result

1. Predict which synthetic case produces the largest W4 output RMSE.
2. Predict whether the reference W4 path is faster in every batch/distribution case.
3. Design separate tests for long-context cache and MoE routing, which this linear probe does not contain.

## 1. Start from concrete tensors and state

Failure modes map to mechanisms: range outliers, distribution shift, long-context
cache/attention, MoE routing imbalance, and small irregular GEMMs.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Outliers enlarge scale and waste codes on ordinary values. |
| 2 | Long context expands cache and can expose positional or attention regressions. |
| 3 | MoE routing and small batches create irregular, overhead-sensitive shapes. |

## 2. Derive the mechanism

One outlier can enlarge a group scale; shifted inputs change layer-output sensitivity;
batch-one and routed experts reduce matrix sizes and make launch/dequant overhead
visible.

For fixed weight error ΔW, output error is `XΔWᵀ`; scaling or shifting X directly
changes its magnitude and direction. This explains why a quantizer calibrated on
ordinary activations can degrade under outliers or domain shift without any weight bytes
changing. Small batches add a systems failure mode because fixed launch, unpack, or
scale overhead is amortized over less work.

Long context and MoE require additional objects: cache bytes/attention error and
expert-routing load balance. They belong in the matrix but cannot be inferred from one
dense linear layer.

## 3. Translate the theory into an experiment

**Experiment:** Stress an INT4 linear reference with ordinary inputs, activation outliers, narrow batches, and shifted distributions on CUDA.

| Experimental role | Frozen definition |
|---|---|
| Baseline | BF16 matrix multiplication in four controlled input regimes |
| Candidate | the same multiplication with group-128 INT4-dequantized weights |
| Held constant | weight matrix and quantizer; only batch/distribution regime changes |
| Measurements | output RMSE/cosine/max error and median/p90 timing per regime |
| Evidence label | `pytorch-gpu` |

The lab holds weights fixed and stresses ordinary, outlier, shifted, and small-batch
inputs, preserving each condition instead of averaging them together.

### Code walk-through

The notebook quantizes one weight matrix once, then evaluates ordinary, batch-1,
activation-outlier, and shifted-domain inputs. Each row carries both numerical error and
timing for baseline/candidate. That paired design prevents a quality failure from being
hidden by a small speed result.

The candidate is a dequantized PyTorch reference tensor, not a packed production W4
kernel. Timing differences therefore illustrate regime sensitivity of the composed path,
not an INT4 hardware speed claim.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Ordinary RMSE | 3.729115 |
| Small-batch RMSE | 3.694930 |
| Activation-outlier RMSE | 14.075421 |
| Shifted-domain RMSE | 13.752637 |
| Largest shifted max error | 67.176849 |

### What the numbers mean

Ordinary and batch-1 RMSE were about 3.73 and 3.69. Activation outliers raised RMSE to
14.0754 and max error to 59.1648; the shifted domain produced RMSE 13.7526 and max error
67.1768. Timing changes stayed tiny and varied by row.

An aggregate over all four cases could hide the roughly 3.7x error jump in the shifted
regimes. The correct response is a targeted calibration, fallback, or rejection rule—not
a global statement that W4 is acceptable.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Design negative tests from known mechanisms and preserve a fallback for the slice that fails.

### Acceptance and rollback gate

Maintain a condition-by-metric failure matrix with reversal thresholds and reproduce
each failure independently before assigning a fallback.

### How this conclusion can fail

One stress tensor cannot represent production tail frequency, and synthetic timing with
dequantized weights is not a native backend result. A matrix that lists long context or
MoE without actually constructing cache or routing evidence would also be misleading;
unexecuted axes must remain marked as future gates.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/25-quantization-failure-modes/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Add a real KV-cache length sweep, rare-language/code/tool-use activation captures, and a
toy MoE with expert-load imbalance. Define acceptance by slice, not only aggregate. Use
the failing rows to design mixed-bit fallbacks and then rerun the full matrix.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
