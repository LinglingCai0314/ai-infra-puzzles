# Lesson 09 — Residual, Concat, and Dependency-Graph Pruning

> **Puzzle:** Which tensors must change together when one residual branch loses channels?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Structural pruning becomes a graph problem at merges. Addition requires shape equality;
concatenation changes downstream channel offsets; normalization and projections carry
the same channel semantics. A local low-importance decision therefore expands into a
coupled pruning group.

For **Residual, Concat, and Dependency-Graph Pruning**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Which tensors must change together when one residual branch loses
channels?* The lab therefore changes the mechanism described below, retains its measured
state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict the exception produced by pruning only one additive branch.
2. Enumerate the tensors coupled to one output-channel deletion.
3. Explain how concat propagation differs from addition.

Before opening Lesson 09's retained output, answer the first prompt— *Predict the
exception produced by pruning only one additive branch.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A two-branch residual block with Conv-BN paths, an addition, and a consumer convolution
is used. The lab records the failure from pruning one branch alone, then constructs a
synchronized narrower group.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Merge semantics determine dependency rules. |
| 2 | A root channel decision propagates through producers, normalization, and consumers. |
| 3 | A valid group must be checked for shape and over-pruning before mutation. |

Lesson 09 tracks three layers through Residual, Concat, and Dependency-Graph Pruning:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

For `z = f(x) + g(x)`, both branch outputs must have identical shapes. Removing output
indices I from f requires a compatible transformation in g and changes the consumer's
input dimension. With concatenation, the retained index mapping is an offset union
rather than equality. Dependency graphs encode these propagation rules so one root
operation yields a complete group and can be rejected before it removes every channel.

The inspectable invariant for **Residual, Concat, and Dependency-Graph Pruning** is
tested by: Trigger and capture an unsynchronized residual-shape failure, then build a
synchronized narrow residual block. Its purpose is to prevent the specific category
error behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Trigger and capture an unsynchronized residual-shape failure, then build a synchronized narrow residual block.

| Experimental role | Frozen definition |
|---|---|
| Baseline | an invalid one-branch channel deletion caught as a diagnostic |
| Candidate | a coupled deletion across both branches, BatchNorm state, and the consumer |
| Held constant | source block, retained indices, input, eval mode, dtype, and copied parameters |
| Measurements | captured mismatch, synchronized output shape, output drift, parameters, and latency |
| Evidence label | `pytorch-gpu` |

This Lesson 09 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **source block, retained indices, input, eval mode, dtype, and copied
parameters**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

The invalid path is wrapped in a try/except so the notebook remains successfully
executed while preserving the error message as evidence. The valid path rebuilds both
branches with the same retained indices and slices the consumer input channels. This is
a manual miniature of a dependency group.

For **Residual, Concat, and Dependency-Graph Pruning**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements a coupled deletion
across both branches, BatchNorm state, and the consumer and records captured mismatch,
synchronized output shape, output drift, parameters, and latency. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Mismatch captured | yes |
| Retained channels | 8 |
| Valid output channels | 12 |
| Valid max error | 0.000460 |
| Full parameters | 1,536 |
| Narrow parameters | 768 |

### What the numbers mean

Pruning only one additive branch produced a captured shape failure: `The size of tensor
a (8) must match the size of tensor b (16) at non-singleton dimension 1`. The
synchronized group retained 8 channels across both branches, normalization state, and
the consumer; it produced 12 output channels with 4.603e-04 control drift.

Lesson 09's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Structural pruning at graph merges is a coupled group operation, never an isolated tensor slice.

### Acceptance and rollback gate

Accept a structural mutation only when a graph-level forward check, group-size guard,
and downstream shape audit pass.

The gate for **Residual, Concat, and Dependency-Graph Pruning** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Matching shapes does not prove semantic correctness: different branches may require
coordinated importance scores, grouped-convolution divisibility, or static attribute
updates. Dynamic control flow can also escape a trace-based dependency graph.

## 6. Follow the theory inside the notebook

In Lesson 09's [`lab.ipynb`](lab.ipynb), first identify **an invalid one-branch channel
deletion caught as a diagnostic** and **a coupled deletion across both branches,
BatchNorm state, and the consumer** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **Residual, Concat, and Dependency-Graph Pruning** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/09-dependency-graph-pruning/lab.ipynb
```

To reproduce **Residual, Concat, and Dependency-Graph Pruning**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Install Torch-Pruning, print the group details for an equivalent block, compare them
with the manual ledger, and add a concat branch to test offset mappings.

For Lesson 09, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Residual, Concat, and Dependency-Graph Pruning** observation belongs to
Lesson 09's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [DepGraph paper](https://arxiv.org/abs/2301.12900)
- [Torch-Pruning reference implementation](https://github.com/VainF/Torch-Pruning)
