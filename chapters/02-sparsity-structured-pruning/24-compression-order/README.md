# Lesson 24 — Ordering Distillation, Quantization, and Pruning

> **Puzzle:** Does pruning before calibration produce the same quantized model as pruning after calibration?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Compression operators do not generally commute. Pruning changes distributions and
structure; quantization freezes scales or codebooks; distillation changes the recovery
objective. A combination roadmap should compare explicit orders at the same
storage/compute budget rather than concatenating technique names.

For **Ordering Distillation, Quantization, and Pruning**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Does pruning before calibration produce the same quantized model
as pruning after calibration?* The lab therefore changes the mechanism described below,
retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict whether dense-calibrated and prune-calibrated INT8 scales match.
2. Write the two operator compositions and identify their different state.
3. Choose when a teacher loss should observe the compressed student.

Before opening Lesson 24's retained output, answer the first prompt— *Predict whether
dense-calibrated and prune-calibrated INT8 scales match.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A linear teacher, student weight, calibration inputs, held-out inputs, magnitude
pruning, symmetric INT8 fake quantization, two operator orders, and an optional short
distillation recovery are compared.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Pruning and calibrated quantization are generally non-commutative. |
| 2 | Every order needs its own calibration and recovery protocol. |
| 3 | Combination fairness requires equal final budgets and held-out metrics. |

Lesson 24 tracks three layers through Ordering Distillation, Quantization, and Pruning:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

Let P be pruning and Q_s quantization under scale s. If s is calibrated on dense W, then
`P(Q_s(W))` uses a range influenced by values later deleted. `Q_{s'}(P(W))` recalibrates
after pruning and can use a different step. They are equal only under special masks and
scales. Distillation adds a loss on teacher outputs and should occur while the student's
actual compression constraints are active if it is meant to recover that candidate.

The inspectable invariant for **Ordering Distillation, Quantization, and Pruning** is
tested by: Compare prune-then-quantize, quantize-then-prune, and constrained
distillation recovery at one final sparsity/bit budget. Its purpose is to prevent the
specific category error behind this puzzle. An algorithmic change, a stored
representation, and a runtime observation remain separate until the candidate and
measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare prune-then-quantize, quantize-then-prune, and constrained distillation recovery at one final sparsity/bit budget.

| Experimental role | Frozen definition |
|---|---|
| Baseline | quantize dense weights with a dense-calibrated scale, then prune |
| Candidate | prune first, recalibrate INT8, and optionally recover under teacher outputs |
| Held constant | teacher, starting student, calibration/held-out tensors, sparsity, quantizer, recovery steps, and seed |
| Measurements | quantization scales, held-out RMSE/cosine, final sparsity, and recovery improvement |
| Evidence label | `numerical-model` |

This Lesson 24 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **teacher, starting student, calibration/held-out tensors, sparsity,
quantizer, recovery steps, and seed**. That frozen condition preserves the dependency or
runtime boundary at issue; the small scale limits transfer to larger models but does not
permit the baseline and candidate to answer different questions.

### Code walk-through

The notebook stores both scales and masks so the order is auditable. The short recovery
optimizes the materialized pruned/quantized simulation against teacher outputs and
reapplies the mask. This illustrates a combination dependency, not full QAT or
production distillation.

For **Ordering Distillation, Quantization, and Pruning**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements prune first,
recalibrate INT8, and optionally recover under teacher outputs and records quantization
scales, held-out RMSE/cosine, final sparsity, and recovery improvement. The artifact
cell serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Quantize-first scale | 0.107549 |
| Prune-first scale | 0.107549 |
| Quantize-first RMSE | 10.841832 |
| Prune-first RMSE | 10.841832 |
| Recovered RMSE | 11.202020 |
| Final sparsity | 60.00% |

### What the numbers mean

Dense-calibrated quantization used scale 0.107549; recalibration after pruning used
0.107549. Held-out RMSE was 10.841832 for quantize-then-prune and 10.841832 for
prune-then-quantize. A 35-step constrained teacher recovery reached 11.202020 at 60.0%
sparsity.

Lesson 24's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Compression order is part of the model recipe because pruning, calibration, and recovery state do not commute.

### Acceptance and rollback gate

Accept a compression order only after its calibration data, recovery objective, final
representation, quality, and runtime are tested as one immutable recipe.

The gate for **Ordering Distillation, Quantization, and Pruning** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Allowing one route to recalibrate or recover while the other cannot makes the order
comparison unfair. Fake quantization does not establish an INT8 kernel. A tiny
teacher-student layer cannot predict end-to-end task accuracy.

## 6. Follow the theory inside the notebook

In Lesson 24's [`lab.ipynb`](lab.ipynb), first identify **quantize dense weights with a
dense-calibrated scale, then prune** and **prune first, recalibrate INT8, and optionally
recover under teacher outputs** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **Ordering Distillation, Quantization, and Pruning** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/24-compression-order/lab.ipynb
```

To reproduce **Ordering Distillation, Quantization, and Pruning**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Create a factorial experiment over order, calibration source, and recovery; then export
every final candidate to the same backend and compare storage, quality, latency, and
operational complexity.

For Lesson 24, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **Ordering Distillation, Quantization, and Pruning** observation belongs
to Lesson 24's RTX 5090 environment, shapes, seed, and protocol. It does not establish
the unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
