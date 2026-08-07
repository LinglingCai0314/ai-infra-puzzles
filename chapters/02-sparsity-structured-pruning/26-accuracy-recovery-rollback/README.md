# Lesson 26 — Accuracy Recovery, Rollback, and Slice Error Analysis

> **Puzzle:** What should happen when overall accuracy recovers but one long-tail class remains below its rollback threshold?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Recovery is not one scalar. A pruned model may recover aggregate accuracy by favoring
common classes while a rare or high-risk class remains degraded. The release artifact
must preserve the dense baseline, pruned checkpoint identity, confusion matrix, slice
deltas, acceptance thresholds, and deterministic rollback decision.

For **Accuracy Recovery, Rollback, and Slice Error Analysis**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *What should happen when overall accuracy recovers but one
long-tail class remains below its rollback threshold?* The lab therefore changes the
mechanism described below, retains its measured state, and names the evidence that would
still be needed for deployment.

## Predict before reading the result

1. Predict which class contributes least to aggregate accuracy.
2. Compute recall from one confusion-matrix row.
3. Write a two-part acceptance rule that protects the tail.

Before opening Lesson 26's retained output, answer the first prompt— *Predict which
class contributes least to aggregate accuracy.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

An imbalanced three-class CUDA classification task, dense checkpoint, magnitude-pruned
candidate, short recovery, confusion matrices, per-class recall, aggregate accuracy, and
a frozen worst-class gate form the experiment.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Aggregate recovery can conceal minority-class regression. |
| 2 | Thresholds must be frozen before the candidate is evaluated. |
| 3 | Rollback is a derived decision linked to an immutable baseline. |

Lesson 26 tracks three layers through Accuracy Recovery, Rollback, and Slice Error
Analysis: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

Accuracy weights each example equally, so a 5% class can fall sharply while changing the
aggregate by less than one point. Per-class recall `TP_c/(TP_c+FN_c)` and a confusion
matrix expose the shift. The gate can require both `accuracy_drop <= a` and `min
recall_drop >= -r`. Recovery selects the best checkpoint only on validation criteria;
the rollback revision remains immutable.

The inspectable invariant for **Accuracy Recovery, Rollback, and Slice Error Analysis**
is tested by: Prune and recover an imbalanced classifier, then derive release or
rollback from aggregate and per-class gates. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Prune and recover an imbalanced classifier, then derive release or rollback from aggregate and per-class gates.

| Experimental role | Frozen definition |
|---|---|
| Baseline | dense checkpoint with frozen validation confusion matrix |
| Candidate | 70%-pruned recovered checkpoint evaluated under the same split |
| Held constant | dataset, imbalance, checkpoint, mask, recovery steps, seed, thresholds, and evaluation code |
| Measurements | aggregate accuracy, per-class recall, worst recall drop, confusion matrices, and rollback decision |
| Evidence label | `pytorch-gpu` |

This Lesson 26 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **dataset, imbalance, checkpoint, mask, recovery steps, seed, thresholds, and
evaluation code**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

The notebook trains a compact baseline, freezes it, prunes a clone, and records both
immediate and recovered metrics. Confusion matrices are computed explicitly on GPU
predictions and stored as lists. The final decision is programmatically derived from the
predeclared aggregate and tail thresholds.

For **Accuracy Recovery, Rollback, and Slice Error Analysis**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements 70%-pruned
recovered checkpoint evaluated under the same split and records aggregate accuracy,
per-class recall, worst recall drop, confusion matrices, and rollback decision. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense accuracy | 100.00% |
| Pruned immediate accuracy | 99.19% |
| Recovered accuracy | 100.00% |
| Worst recall drop | 0.00% |
| Rollback required | no |
| Final sparsity | 70.02% |

### What the numbers mean

Dense, immediate-pruned, and recovered aggregate accuracies were 100.0%, 99.2%, and
100.0%. The worst class-recall change after recovery was 0.0%; under the frozen 3-point
aggregate and 10-point recall gates, rollback_required=False.

Lesson 26's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Recovery is complete only when aggregate and protected slices pass a predeclared gate with a tested rollback path.

### Acceptance and rollback gate

Release only when every frozen aggregate and critical-slice gate passes; otherwise
select the recorded dense revision as rollback.

The gate for **Accuracy Recovery, Rollback, and Slice Error Analysis** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

A toy class imbalance is not a production taxonomy, and recall alone ignores precision
or calibration. Choosing thresholds after seeing the candidate invalidates the gate.
Recovery on the final test split leaks evaluation.

## 6. Follow the theory inside the notebook

In Lesson 26's [`lab.ipynb`](lab.ipynb), first identify **dense checkpoint with frozen
validation confusion matrix** and **70%-pruned recovered checkpoint evaluated under the
same split** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Accuracy Recovery, Rollback, and Slice Error Analysis** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/26-accuracy-recovery-rollback/lab.ipynb
```

To reproduce **Accuracy Recovery, Rollback, and Slice Error Analysis**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Add precision, calibration, and domain slices; separate train/validation/test; rehearse
the actual model-registry rollback and monitor the same gates during canary.

For Lesson 26, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Accuracy Recovery, Rollback, and Slice Error Analysis** observation
belongs to Lesson 26's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
