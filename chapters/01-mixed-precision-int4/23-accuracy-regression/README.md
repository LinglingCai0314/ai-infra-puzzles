# Lesson 23 — Accuracy Regression Tests for Quantized Models

> **Puzzle:** Can one aggregate score hide a serious quantization regression?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Quantization quality is not one cosine score. A release can preserve average logits
while changing top-1 decisions, rare domains, long-context behavior,
calibration-sensitive layers, or safety-critical outputs. Regression testing turns those
failure modes into frozen gates that can block a numerically small but behaviorally
important change.

## Predict before reading the result

1. Predict whether cross-entropy, perplexity, and top-1 agreement will all move in the same relative direction.
2. Explain why the synthetic perplexity magnitude is not meaningful as a language-model score.
3. Design at least three deployment slices that an aggregate metric could hide.

## 1. Start from concrete tensors and state

Quality evidence spans token likelihood (cross-entropy/perplexity), task metrics,
output/logit agreement, safety/alignment cases, and business-specific slices.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Perplexity measures token likelihood, task accuracy measures decisions, and alignment samples cover product behavior. |
| 2 | Thresholds should be frozen before examining the candidate. |
| 3 | Slice-level failures can be hidden by a stable global average. |

## 2. Derive the mechanism

Perplexity is `exp(mean token cross-entropy)`; a small average loss change can coexist
with large ranking changes on a rare slice. Top-1 agreement reveals decision changes but
not whether either answer is correct.

For targets y and logits z, cross-entropy measures probability assigned to y; perplexity
is `exp(loss)` and can magnify small loss changes. Top-1 agreement instead asks whether
the candidate preserves the baseline decision, regardless of whether either decision is
correct. Logit distance, task accuracy, exact-match, calibration, and human/safety
checks answer still different questions.

A release gate should define baselines, datasets, seeds, tolerances, and slice policies
before the candidate is evaluated. Otherwise thresholds drift to accommodate the
observed regression.

## 3. Translate the theory into an experiment

**Experiment:** Run a tiny CUDA language-model head before and after INT4 weight Q/DQ, then compare cross-entropy, perplexity, top-1 agreement, and slice metrics.

| Experimental role | Frozen definition |
|---|---|
| Baseline | floating-point synthetic classifier logits over 4,096 tokens |
| Candidate | INT4-dequantized weight logits for the same hidden states and targets |
| Held constant | tokens, vocabulary, targets, hidden states, weight matrix, seed |
| Measurements | loss, derived perplexity, overall and half-slice top-1 agreement |
| Evidence label | `pytorch-gpu` |

The CUDA probe computes loss, perplexity, overall agreement, and two slices from
identical hidden states before and after INT4 Q/DQ.

### Code walk-through

The notebook generates one fixed synthetic classification problem, computes baseline and
quantized logits, and evaluates the same targets. It reports the complete set and two
halves so a slice disagreement cannot be hidden by one aggregate.

Because random logits yield enormous losses and perplexities, the absolute values are
intentionally labeled synthetic. The exercise demonstrates metric relationships and gate
structure, not language modeling ability.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Baseline loss | 32.049492 |
| Candidate loss | 32.212620 |
| Baseline synthetic perplexity | 8.297e+13 |
| Candidate synthetic perplexity | 9.767e+13 |
| Top-1 agreement | 83.6914% |

### What the numbers mean

Candidate loss increased from 32.049492 to 32.212620. Exponentiation turned that modest
difference into synthetic perplexities of about `8.30e13` and `9.77e13`. Overall top-1
agreement was 0.836914; the two halves were 0.838379 and 0.835449.

The near-equal slices do not reveal a concentrated failure in this constructed set, but
roughly 16% decision disagreement is clearly visible. A real release would need task
correctness, not only agreement with the baseline.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Use a layered quality gate and retain the baseline outputs needed to explain a regression.

### Acceptance and rollback gate

Freeze datasets, prompts, decoding, baseline revision, thresholds, and slice
definitions. Fail on a critical slice even if the global average passes.

### How this conclusion can fail

Perplexity can overflow or become hard to interpret at extreme synthetic losses.
Baseline agreement can preserve a baseline mistake, and average accuracy can hide a
critical slice. Reusing calibration prompts for regression also lets quantizer selection
overfit the gate.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map floating-point synthetic classifier logits over
4,096 tokens and INT4-dequantized weight logits for the same hidden states and targets
back to the derivation. Verify the printed environment, then check that tokens,
vocabulary, targets, hidden states, weight matrix, seed stayed fixed. Read loss, derived
perplexity, overall and half-slice top-1 agreement before applying the acceptance gate;
the artifact-writing cell retains the complete structured result from the recorded run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/23-accuracy-regression/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Replace synthetic logits with a small named model and a frozen, redistribution-safe
suite: perplexity on held-out text, task accuracy, long-context slices,
multilingual/code/tool-use samples, and answer/logit agreement. Publish thresholds and
reversal criteria before running the candidate.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 23's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
