# Lesson 12 — Sparse Regularization and Learnable Structural Gates

> **Puzzle:** Can training produce a reproducible structural ranking instead of choosing a threshold after the fact?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Learnable gates attach a continuous variable to each channel and optimize it with the
task. A sparsity penalty can separate useful and dispensable structures, but
thresholding still creates a discrete architecture and must be evaluated. Gate values,
penalty strength, temperature, and threshold belong in the artifact.

For **Sparse Regularization and Learnable Structural Gates**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *Can training produce a reproducible structural ranking instead
of choosing a threshold after the fact?* The lab therefore changes the mechanism
described below, retains its measured state, and names the evidence that would still be
needed for deployment.

## Predict before reading the result

1. Predict whether sigmoid gates reach exact zero under an L1 penalty.
2. Predict how increasing lambda changes active channels and task loss.
3. Choose a threshold using a frozen validation objective rather than the training batch.

Before opening Lesson 12's retained output, answer the first prompt— *Predict whether
sigmoid gates reach exact zero under an L1 penalty.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A frozen feature tensor, a trainable gated linear predictor, per-channel sigmoid gates,
a task loss, an L1 gate penalty, and a threshold sweep form the lab.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Regularization shapes a ranking but does not physically remove channels. |
| 2 | Continuous and thresholded objectives must both be reported. |
| 3 | Lambda and threshold define different parts of the sparsity trade-off. |

Lesson 12 tracks three layers through Sparse Regularization and Learnable Structural
Gates: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

With gate `g_c=sigmoid(a_c)`, a hidden feature becomes `g_c h_c`. Optimizing `L_task +
lambda sum(g_c)` trades fit against active width. Sigmoid gates rarely become exact
zeros, so deployment selects a threshold or top-k budget and physically rebuilds the
layer. The continuous optimum and discrete candidate are different models; both losses
must be measured. Gate scale can also trade with neighboring weights unless those
degrees of freedom are controlled.

The inspectable invariant for **Sparse Regularization and Learnable Structural Gates**
is tested by: Train channel gates under two regularization strengths and evaluate a
frozen threshold sweep. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Train channel gates under two regularization strengths and evaluate a frozen threshold sweep.

| Experimental role | Frozen definition |
|---|---|
| Baseline | weak gate regularization with a mostly dense effective width |
| Candidate | stronger regularization plus discrete threshold candidates |
| Held constant | features, targets, predictor initialization, optimizer steps, thresholds, seed, and validation split |
| Measurements | gate distribution, active channels, continuous loss, thresholded loss, and selected threshold |
| Evidence label | `numerical-model` |

This Lesson 12 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **features, targets, predictor initialization, optimizer steps, thresholds,
seed, and validation split**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The feature generator deliberately makes only a subset of channels predictive. Two gate
models begin from identical logits, and the threshold sweep evaluates held-out loss
without further fitting. This isolates whether the learned ranking exposes the known
sparse structure.

For **Sparse Regularization and Learnable Structural Gates**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements stronger
regularization plus discrete threshold candidates and records gate distribution, active
channels, continuous loss, thresholded loss, and selected threshold. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Weak active channels | 6 |
| Strong active channels | 6 |
| Weak gate mean | 0.468424 |
| Strong gate mean | 0.259656 |
| Selected threshold | 0.200000 |
| Thresholded validation MSE | 0.019543 |

### What the numbers mean

Weak regularization left 6 gates above 0.5 with mean 0.4684; strong regularization left
6 with mean 0.2597. The frozen threshold sweep selected 0.2, 6 active channels, and
validation MSE 0.019543. Continuous gates still require physical rebuilding.

Lesson 12's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Learnable gates turn structure selection into optimization, but deployment still requires a separately validated discrete architecture.

### Acceptance and rollback gate

Accept a gate-derived architecture only when threshold selection is frozen, held-out
quality passes, and the physical model reproduces the gated candidate.

The gate for **Sparse Regularization and Learnable Structural Gates** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

Jointly trainable downstream weights can absorb inverse gate scaling, making raw gates
misleading. Selecting lambda or threshold on the final test set leaks evaluation. Hard
thresholding may also remove interacting channels that looked individually small.

## 6. Follow the theory inside the notebook

In Lesson 12's [`lab.ipynb`](lab.ipynb), first identify **weak gate regularization with
a mostly dense effective width** and **stronger regularization plus discrete threshold
candidates** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Sparse Regularization and Learnable Structural Gates** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/12-sparse-regularization-gates/lab.ipynb
```

To reproduce **Sparse Regularization and Learnable Structural Gates**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Add hard-concrete or top-k gates, rebuild a physical narrow layer, and test ranking
stability across seeds and task slices.

For Lesson 12, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **Sparse Regularization and Learnable Structural Gates** observation
belongs to Lesson 12's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Network Slimming](https://arxiv.org/abs/1708.06519)
- [To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)
