# Lesson 21 — Safe Pruning for Detection and Segmentation

> **Puzzle:** Can an unchanged average metric hide a large regression on small objects or a rare mask class?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Detection and segmentation heads consume multi-scale features, and business risk is
rarely uniform across sizes and classes. A pruning candidate can preserve an aggregate
proxy while degrading the feature-pyramid level responsible for small objects. Safety
therefore requires slice metrics and per-branch budgets.

For **Safe Pruning for Detection and Segmentation**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Can an unchanged average metric hide a large regression on small
objects or a rare mask class?* The lab therefore changes the mechanism described below,
retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict which pyramid branch is most sensitive for the small-object proxy.
2. Construct an example where mean error falls but worst-slice error rises.
3. Choose both aggregate and slice-level rollback gates.

Before opening Lesson 21's retained output, answer the first prompt— *Predict which
pyramid branch is most sensitive for the small-object proxy.*—and write one observation
that would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A three-scale feature-pyramid toy head, synthetic large/medium/small targets, a
uniform-pruning candidate, a protected-high-resolution candidate, and per-slice
reconstruction errors form the controlled lab.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Multi-scale branches have different semantic responsibilities. |
| 2 | Aggregate quality can pass while a protected slice fails. |
| 3 | Risk-weighted budgets require explicit slice thresholds. |

Lesson 21 tracks three layers through Safe Pruning for Detection and Segmentation:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

High-resolution pyramid features carry more spatial positions and often serve small
objects. If aggregate loss weights every tensor element or sample uniformly, a large
branch can dominate or a rare slice can disappear in the mean. Define `E_slice`
separately and an acceptance rule such as `max slice regression <= tau` in addition to
aggregate change. Budget allocation then becomes risk-weighted rather than purely
parameter-weighted.

The inspectable invariant for **Safe Pruning for Detection and Segmentation** is tested
by: Compare uniform channel pruning with a high-resolution-protected budget at equal
total retained channels. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare uniform channel pruning with a high-resolution-protected budget at equal total retained channels.

| Experimental role | Frozen definition |
|---|---|
| Baseline | uniform pruning across three feature-pyramid branches |
| Candidate | risk-weighted pruning that protects the high-resolution/small-object branch |
| Held constant | feature tensors, targets, total retained-channel budget, head weights, seed, and slice definitions |
| Measurements | aggregate error, large/medium/small slice error, worst-slice regression, and retained channels |
| Evidence label | `numerical-model` |

This Lesson 21 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **feature tensors, targets, total retained-channel budget, head weights,
seed, and slice definitions**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The notebook constructs target outputs so each slice depends most strongly on its
corresponding scale. Both candidates spend the same total channel budget, but allocate
it differently. Reporting every slice next to the aggregate exposes whether the
protected policy trades average error for a safer worst case.

For **Safe Pruning for Detection and Segmentation**, the environment cell asserts CUDA
and fixes a lesson-specific seed. The experiment cell implements risk-weighted pruning
that protects the high-resolution/small-object branch and records aggregate error,
large/medium/small slice error, worst-slice regression, and retained channels. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Uniform aggregate RMSE | 3.688635 |
| Protected aggregate RMSE | 2.925958 |
| Uniform small-slice RMSE | 5.187524 |
| Protected small-slice RMSE | 2.448860 |
| Worst uniform slice | small |
| Total retained channels | 36 |

### What the numbers mean

Both policies retained 36 channels across three branches. Uniform allocation produced
aggregate RMSE 3.688635 and small-slice RMSE 5.187524; protecting the high-resolution
branch produced 2.925958 and 2.448860, respectively. The per-slice table—not the
aggregate alone—determines whether the risk trade is acceptable.

Lesson 21's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Safe pruning treats the worst critical slice as a first-class constraint rather than trusting an aggregate metric.

### Acceptance and rollback gate

Accept only when aggregate detection/segmentation quality and every business-critical
size/class slice stay within frozen thresholds.

The gate for **Safe Pruning for Detection and Segmentation** is stricter than “the code
ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

A synthetic reconstruction proxy is not COCO AP, mask AP, recall, or calibration. Slice
definitions chosen after observing failures can overfit the report. Feature channels
also interact across the neck and head in real architectures.

## 6. Follow the theory inside the notebook

In Lesson 21's [`lab.ipynb`](lab.ipynb), first identify **uniform pruning across three
feature-pyramid branches** and **risk-weighted pruning that protects the
high-resolution/small-object branch** without running them. Next inspect the dimensions
or lifecycle state that implements the derivation. After **Run All**, verify the RTX
5090 environment and the frozen fields before reconciling the result table with the
artifact.

The reader loop for **Safe Pruning for Detection and Segmentation** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/21-detection-segmentation-safety/lab.ipynb
```

To reproduce **Safe Pruning for Detection and Segmentation**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Run the policy on a real detector with COCO `AP`, `AP_S`, `AP_M`, `AP_L`, class recall,
and mask metrics, then bind each gate to a rollback action.

For Lesson 21, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **Safe Pruning for Detection and Segmentation** observation belongs to
Lesson 21's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [COCO evaluation](https://cocodataset.org/#detection-eval)
- [DepGraph paper](https://arxiv.org/abs/2301.12900)
