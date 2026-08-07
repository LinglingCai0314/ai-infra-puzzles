# Lesson 08 — BatchNorm Scale Factors and Network Slimming

> **Puzzle:** When does a small BatchNorm gamma become a removable channel rather than merely a small multiplier?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Network Slimming creates a train-time ranking signal by regularizing BatchNorm scale
factors. The scale does not remove a channel by itself. Deployment still requires
selecting indices, rebuilding the producing convolution, slicing BatchNorm state, and
propagating the same indices into every consumer.

For **BatchNorm Scale Factors and Network Slimming**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *When does a small BatchNorm gamma become a removable channel
rather than merely a small multiplier?* The lab therefore changes the mechanism
described below, retains its measured state, and names the evidence that would still be
needed for deployment.

## Predict before reading the result

1. Predict whether simply zeroing gamma matches physical deletion when beta is nonzero.
2. List every BatchNorm tensor that must be sliced.
3. Predict the output drift between a properly masked control and narrowed model.

Before opening Lesson 08's retained output, answer the first prompt— *Predict whether
simply zeroing gamma matches physical deletion when beta is nonzero.*—and write one
observation that would falsify the answer. If the result is already visible, hide it and
make the commitment first; otherwise this becomes post-hoc explanation rather than a
pruning experiment.

## 1. Start from concrete tensors and state

A Conv-BN-ReLU-Conv block supplies convolution filters, BatchNorm gamma/beta/running
statistics, retained channel indices, a gamma-masked control, and a physically narrowed
copy.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Gamma is an importance signal, not a structural deletion. |
| 2 | BatchNorm affine and running-state tensors share the channel axis. |
| 3 | Consumer weights must receive the identical retained indices. |

Lesson 08 tracks three layers through BatchNorm Scale Factors and Network Slimming:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

BatchNorm output per channel is `y_c = gamma_c (x_c - mu_c)/sqrt(var_c+eps) + beta_c`. A
small gamma suppresses normalized variation, but beta can still contribute a constant
and downstream weights can amplify it. Ranking by `|gamma|` is therefore a pruning
heuristic learned under a sparsity regularizer. Physical removal is valid only when the
chosen channel and all coupled parameters are sliced consistently and the resulting
function is evaluated.

The inspectable invariant for **BatchNorm Scale Factors and Network Slimming** is tested
by: Rank channels by gamma, create a semantics-preserving masked control, and rebuild
the block at half width. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Rank channels by gamma, create a semantics-preserving masked control, and rebuild the block at half width.

| Experimental role | Frozen definition |
|---|---|
| Baseline | gamma-masked full-width Conv-BN-ReLU-Conv block |
| Candidate | physically narrowed block using the same retained gamma-ranked channels |
| Held constant | input, retained indices, all copied Conv/BN parameters, eval mode, dtype, and timing protocol |
| Measurements | gamma threshold, retained channels, output max error, parameters, and median latency |
| Evidence label | `numerical-model` |

This Lesson 08 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **input, retained indices, all copied Conv/BN parameters, eval mode, dtype,
and timing protocol**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The notebook sets the removed channels to a neutral post-BN value in the control before
copying the retained convolution filters, BN state, and second-layer input slices. Eval
mode freezes running statistics. The equivalence check isolates structural bookkeeping
from the separate question of whether gamma ranking preserves task quality.

For **BatchNorm Scale Factors and Network Slimming**, the environment cell asserts CUDA
and fixes a lesson-specific seed. The experiment cell implements physically narrowed
block using the same retained gamma-ranked channels and records gamma threshold,
retained channels, output max error, parameters, and median latency. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Retained channels | 12 |
| Gamma threshold | 0.635652 |
| Output max error | 0.000244 |
| Full parameters | 4,368 |
| Narrow parameters | 2,184 |
| Narrow median | 0.044560 ms |

### What the numbers mean

The gamma ranking retained 12 channels above an absolute threshold of 0.635652. After
slicing convolution and every BatchNorm state tensor, the narrow output matched the
gamma-masked control within 2.438e-04. Parameters fell from 4,368 to 2,184; ranking
quality on a real task remains unmeasured.

Lesson 08's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Network Slimming turns BatchNorm scales into a ranking mechanism; deployment benefit begins only after consistent structural removal.

### Acceptance and rollback gate

Accept the ranking only after held-out quality, coupled slicing, physical width, and
runtime evidence all pass.

The gate for **BatchNorm Scale Factors and Network Slimming** is stricter than “the code
ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Small gamma values can be scale-invariant with neighboring weights, and nonzero beta
breaks naive zero-gamma reasoning. Training without the intended L1 pressure may produce
an uninformative ranking. Residual and concatenation consumers need a dependency graph
beyond this local block.

## 6. Follow the theory inside the notebook

In Lesson 08's [`lab.ipynb`](lab.ipynb), first identify **gamma-masked full-width
Conv-BN-ReLU-Conv block** and **physically narrowed block using the same retained
gamma-ranked channels** without running them. Next inspect the dimensions or lifecycle
state that implements the derivation. After **Run All**, verify the RTX 5090 environment
and the frozen fields before reconciling the result table with the artifact.

The reader loop for **BatchNorm Scale Factors and Network Slimming** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/08-network-slimming/lab.ipynb
```

To reproduce **BatchNorm Scale Factors and Network Slimming**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Train gamma with an explicit sparsity penalty, compare rankings across seeds, and
propagate selected channels through a residual model with a graph-level pruning tool.

For Lesson 08, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **BatchNorm Scale Factors and Network Slimming** observation belongs to
Lesson 08's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Network Slimming](https://arxiv.org/abs/1708.06519)
- [DepGraph paper](https://arxiv.org/abs/2301.12900)
