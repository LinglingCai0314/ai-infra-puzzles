# Lesson 10 — Taylor Importance: Ranking Channels by Loss Change

> **Puzzle:** Can a small-norm channel still have a large effect on the loss?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Magnitude sees the parameter but not the data or objective. Taylor pruning uses the
local product of an activation and its loss gradient to estimate how much removing a
channel changes the loss. The approximation is cheap enough to rank many structures
without a full retraining run for each one.

For **Taylor Importance: Ranking Channels by Loss Change**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Can a small-norm channel still have a large effect on the loss?*
The lab therefore changes the mechanism described below, retains its measured state, and
names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict whether L1 and Taylor produce identical rankings.
2. Write the first-order term for zeroing one activation channel.
3. Choose the correlation that validates each ranking against actual ablations.

Before opening Lesson 10's retained output, answer the first prompt— *Predict whether L1
and Taylor produce identical rankings.*—and write one observation that would falsify the
answer. If the result is already visible, hide it and make the commitment first;
otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A small classifier exposes one hidden activation tensor, its retained gradient,
per-channel L1 weight scores, Taylor scores, and actual held-out loss increases from
channel ablation.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Taylor scores are objective- and data-dependent. |
| 2 | Ablation loss change is the validation target for an importance ranking. |
| 3 | First-order scores ignore interactions and distribution shift. |

Lesson 10 tracks three layers through Taylor Importance: Ranking Channels by Loss
Change: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

If channel activation `h_c` is replaced by zero, first-order expansion gives `Delta L_c
≈ |∂L/∂h_c · (-h_c)|`, aggregated across samples and positions. Weight L1 instead ranks
`sum |W_c|`. Taylor incorporates the current data and loss but remains local:
interactions between channels and higher-order curvature are omitted. Correlation with
actual one-channel ablation is the direct diagnostic for this toy problem.

The inspectable invariant for **Taylor Importance: Ranking Channels by Loss Change** is
tested by: Compare L1 and Taylor channel rankings with exhaustive one-channel loss
ablations on a held-out batch. Its purpose is to prevent the specific category error
behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare L1 and Taylor channel rankings with exhaustive one-channel loss ablations on a held-out batch.

| Experimental role | Frozen definition |
|---|---|
| Baseline | channel ranking by outgoing/associated weight L1 magnitude |
| Candidate | channel ranking by absolute activation-gradient product |
| Held constant | trained toy classifier, calibration batch, held-out ablation batch, channel set, and loss |
| Measurements | Spearman correlation with actual loss increase, top-ranked channel, and loss deltas |
| Evidence label | `numerical-model` |

This Lesson 10 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **trained toy classifier, calibration batch, held-out ablation batch, channel
set, and loss**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

The notebook retains gradients on the hidden activation, performs one calibration
backward pass, and aggregates `|h × grad|` per channel. It then runs controlled
ablations on held-out inputs to construct the target ranking. The comparison measures
ranking agreement rather than claiming a production pruning algorithm.

For **Taylor Importance: Ranking Channels by Loss Change**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements channel ranking by
absolute activation-gradient product and records Spearman correlation with actual loss
increase, top-ranked channel, and loss deltas. The artifact cell serializes those same
fields. Only optional-backend import or API failures become compatibility evidence; an
error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| L1 Spearman | 0.867133 |
| Taylor Spearman | 0.895105 |
| L1 top channel | 5 |
| Taylor top channel | 5 |
| Actual top channel | 7 |
| Baseline loss | 0.126262 |

### What the numbers mean

Against exhaustive held-out ablations, L1 ranking had Spearman 0.8671 and Taylor had
0.8951. Their top channels were 5 and 5, while the largest actual loss increase came
from channel 7. The result tests one local ranking on one calibration batch.

Lesson 10's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Taylor importance estimates local loss sensitivity; its value is established by held-out ablation agreement, not by the formula alone.

### Acceptance and rollback gate

Accept an importance metric only if its ranking is stable across representative batches
and improves the declared quality-cost objective after pruning.

The gate for **Taylor Importance: Ranking Channels by Loss Change** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

One calibration batch can reverse scores, negative and positive first-order terms can
cancel depending on aggregation, and simultaneous removal invalidates
independent-channel estimates. Correlation on a tiny network does not establish ImageNet
behavior.

## 6. Follow the theory inside the notebook

In Lesson 10's [`lab.ipynb`](lab.ipynb), first identify **channel ranking by
outgoing/associated weight L1 magnitude** and **channel ranking by absolute
activation-gradient product** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **Taylor Importance: Ranking Channels by Loss Change** is **predict
→ execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/10-taylor-importance/lab.ipynb
```

To reproduce **Taylor Importance: Ranking Channels by Loss Change**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Repeat across batches, compare signed, absolute, and second-order approximations, then
prune several channels jointly and measure how ranking quality degrades with sparsity.

For Lesson 10, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **Taylor Importance: Ranking Channels by Loss Change** observation
belongs to Lesson 10's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Pruning CNNs for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)
- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
