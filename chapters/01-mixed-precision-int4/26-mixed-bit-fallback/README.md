<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 26 — Mixed-Bit Strategies and Sensitive-Layer Fallback

> **Puzzle:** If only a few layers cause most quantization error, should every layer use more bits?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Uniform four-bit quantization spends the same precision on layers with different
sensitivity. A mixed-bit policy measures how much each layer perturbs the end-to-end
output, then allocates a fixed higher-precision budget to the worst offenders. The
budget and the reassembled model result are as important as the ranking.

## Predict before reading the result

1. Predict which layers receive INT8 when only two fallbacks are allowed.
2. Compute the expected average bit width for two INT8 and four INT4 equal-size layers.
3. Explain why isolated layer sensitivity must be followed by an assembled-model evaluation.

## 1. Start from concrete tensors and state

Mixed-bit design assigns a precision/configuration to each layer or group under a
memory, latency, and quality budget.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Layer sensitivity is measured by the downstream objective under representative inputs. |
| 2 | Mixed-bit allocation trades metadata and kernel diversity against quality. |
| 3 | Fallback layers need a deterministic rule and a fixed memory budget. |

## 2. Derive the mechanism

A sensitivity scan replaces one layer at a time and measures downstream change. A simple
allocation then spends extra bits on the largest marginal quality benefit per added
byte; interactions require re-evaluating the assembled model.

Let candidate bit assignment b_l minimize model error subject to `Σ n_l b_l / Σ n_l ≤
B`, where n_l is layer size and B is the average-bit budget. A simple greedy policy
measures the output RMSE caused by quantizing one layer at a time and assigns extra
precision to the largest scores. Interactions make this only a heuristic: two
individually safe layers can amplify each other when quantized together.

Therefore the procedure has two stages—rank under a fixed probe, then assemble and
retest the complete assignment. Storage, kernel compatibility, and latency must also be
recalculated because mixed formats can add dispatch boundaries.

## 3. Translate the theory into an experiment

**Experiment:** Quantize a six-layer CUDA MLP one layer at a time, rank sensitivity, then construct a budgeted INT4/INT8 mixed-bit candidate.

| Experimental role | Frozen definition |
|---|---|
| Baseline | six-layer floating-point MLP and an all-INT4 candidate |
| Candidate | INT8 for the two most sensitive layers, INT4 for the remaining four |
| Held constant | equal layer sizes, calibration input, quantizers, two-layer fallback budget |
| Measurements | per-layer isolated RMSE, selected layers, average weight bits, assembled output error |
| Evidence label | `pytorch-gpu` |

The six-layer CUDA lab ranks INT4 substitutions, gives two layers INT8, computes average
bits, and re-runs end to end.

### Code walk-through

The notebook quantizes each of six equal-size matrices to INT4 and INT8. It replaces one
layer at a time to measure sensitivity against the full-precision network, selects the
top two, constructs the mixed model, and re-evaluates end to end.

Because layers are equal size, the budget is transparent: `(2×8 + 4×4)/6 = 5.333` bits
per weight. A real transformer would weight layers by parameter count and
backend-compatible groupings.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| INT8 fallback layers | 0, 1 |
| Average weight bits | 5.333 bits/weight |
| Layer 0 isolated RMSE | 0.001379 |
| Layer 1 isolated RMSE | 0.001304 |
| Assembled RMSE | 0.002484 |
| Assembled cosine | 0.976198 |

### What the numbers mean

Layers 0 and 1 had the highest isolated RMSE, 0.0013786 and 0.00130424, so they received
INT8. The mixed assignment used 5.333 average bits and produced assembled RMSE
0.00248394 with cosine 0.976198.

The assembled error is larger than any isolated score, demonstrating interaction across
layers. The ranking still gives a reproducible budgeted candidate, but whether it beats
all-INT4 or another allocation must be judged with a frozen quality target and actual
storage/runtime measurements.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Use sensitivity scans to spend precision where it protects the objective, then re-measure the assembled model.

### Acceptance and rollback gate

Freeze calibration/evaluation, record isolated sensitivities, budget, chosen fallback
layers, final assembled quality, storage, operator coverage, and latency.

### How this conclusion can fail

Selecting fallback layers on the final task set overfits deployment evaluation.
Comparing mixed-bit quality without reporting average bits is unfair. Backend
fragmentation can also erase theoretical benefit if INT4 and INT8 layers use
incompatible packing or force synchronization/materialization.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/26-mixed-bit-fallback/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Add all-INT4 and all-INT8 assembled baselines, search several budgets, and plot quality
versus effective bytes. Repeat sensitivity on multiple domains and sequence lengths.
Then run a backend that supports the mixed formats and measure operator boundaries,
memory, and latency.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
