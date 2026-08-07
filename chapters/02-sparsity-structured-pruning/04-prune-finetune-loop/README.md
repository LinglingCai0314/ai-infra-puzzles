# Lesson 04 — Closing the Loop: Train, Prune, Recover, and Re-evaluate

> **Puzzle:** Why does a one-shot 70% mask often fail when the same target reached gradually can recover?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Pruning changes an optimization problem, not just a checkpoint file. The model must
absorb a perturbation while the remaining weights adapt. A closed loop records the dense
state, pruning event, recovery budget, best recovered metric, final mask, and rollback
decision instead of reporting only the target sparsity.

For **Closing the Loop: Train, Prune, Recover, and Re-evaluate**, the engineering
question is not whether a definition can be repeated; it is whether the following claim
survives a controlled GPU test: *Why does a one-shot 70% mask often fail when the same
target reached gradually can recover?* The lab therefore changes the mechanism described
below, retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict the immediate accuracy drop after one-shot 70% pruning.
2. Predict whether staged pruning will finish with a smaller or larger recovery gap.
3. Name the artifact needed to prove that zeros did not regrow.

Before opening Lesson 04's retained output, answer the first prompt— *Predict the
immediate accuracy drop after one-shot 70% pruning.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A synthetic but separable classification dataset, a small MLP, one dense initialization,
one-shot and staged pruning schedules, optimizer steps, masks reapplied after every
update, and held-out accuracy form the experiment.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Pruning is a state transition followed by constrained optimization. |
| 2 | Masks must remain enforced during recovery. |
| 3 | One-shot and gradual routes need equal initialization and declared budgets. |

Lesson 04 tracks three layers through Closing the Loop: Train, Prune, Recover, and
Re-evaluate: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

Magnitude pruning projects weights onto a sparse support. An abrupt 70% projection can
remove several co-adapted paths at once and move the loss far from the local basin. A
staged schedule introduces smaller support changes followed by recovery. Because
optimizers can regrow masked weights, the mask must be enforced after updates unless the
parameterization guarantees zeros. Fairness requires both routes to begin from the
identical dense checkpoint and consume a declared recovery budget.

The inspectable invariant for **Closing the Loop: Train, Prune, Recover, and
Re-evaluate** is tested by: Train one dense toy classifier, then compare one-shot and
staged 70% pruning with equal recovery steps. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Train one dense toy classifier, then compare one-shot and staged 70% pruning with equal recovery steps.

| Experimental role | Frozen definition |
|---|---|
| Baseline | one-shot 70% magnitude pruning from the frozen dense checkpoint |
| Candidate | three-stage pruning to the same target with interleaved recovery |
| Held constant | initial checkpoint, dataset, split, optimizer family, total recovery steps, seed, and final sparsity |
| Measurements | dense accuracy, immediate drop, recovered accuracy, final sparsity, and best recovery step |
| Evidence label | `pytorch-gpu` |

This Lesson 04 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **initial checkpoint, dataset, split, optimizer family, total recovery steps,
seed, and final sparsity**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The lab trains one baseline and clones it before either pruning route. A mask helper
chooses the global threshold and a recovery helper reapplies the mask after every
optimizer step. Both routes consume the same total number of updates; the staged route
only changes when support is removed. This makes schedule the independent variable.

For **Closing the Loop: Train, Prune, Recover, and Re-evaluate**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements
three-stage pruning to the same target with interleaved recovery and records dense
accuracy, immediate drop, recovered accuracy, final sparsity, and best recovery step.
The artifact cell serializes those same fields. Only optional-backend import or API
failures become compatibility evidence; an error in the core comparison still fails the
notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense accuracy | 93.15% |
| One-shot immediate | 90.48% |
| One-shot recovered | 91.37% |
| Gradual recovered | 91.37% |
| Final sparsity | 69.97% |

### What the numbers mean

The dense toy classifier reached 93.2%. One-shot 70% pruning changed validation accuracy
immediately to 90.5% and recovered to 91.4% after 36 updates. The staged route finished
at 91.4% with 70.0% zeros under the same update budget. This isolates the support
trajectory on one synthetic task; it is not a universal schedule ranking.

Lesson 04's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> A pruning result is a trajectory with a support constraint and recovery budget, not a mask applied once.

### Acceptance and rollback gate

Accept the pruned checkpoint only when held-out accuracy and sparsity both pass and the
exact dense checkpoint remains available for rollback.

The gate for **Closing the Loop: Train, Prune, Recover, and Re-evaluate** is stricter
than “the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

A toy separable dataset can favor either schedule and does not predict ImageNet
recovery. Comparing unequal training steps, learning rates, or data order also
invalidates the causal claim. The lab establishes the control-loop mechanics, not a
universal schedule ranking.

## 6. Follow the theory inside the notebook

In Lesson 04's [`lab.ipynb`](lab.ipynb), first identify **one-shot 70% magnitude pruning
from the frozen dense checkpoint** and **three-stage pruning to the same target with
interleaved recovery** without running them. Next inspect the dimensions or lifecycle
state that implements the derivation. After **Run All**, verify the RTX 5090 environment
and the frozen fields before reconciling the result table with the artifact.

The reader loop for **Closing the Loop: Train, Prune, Recover, and Re-evaluate** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/04-prune-finetune-loop/lab.ipynb
```

To reproduce **Closing the Loop: Train, Prune, Recover, and Re-evaluate**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Repeat with several seeds and recovery budgets, plot accuracy immediately before and
after each pruning event, and add a distillation term as a separately controlled
intervention.

For Lesson 04, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Closing the Loop: Train, Prune, Recover, and Re-evaluate** observation
belongs to Lesson 04's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
