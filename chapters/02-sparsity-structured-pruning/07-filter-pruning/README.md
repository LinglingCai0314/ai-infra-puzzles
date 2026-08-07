# Lesson 07 — Filter Pruning: Making Convolution Physically Narrower

> **Puzzle:** Why does zeroing filters differ from deleting them?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A dense convolution library receives input/output channel counts, kernel size, stride,
and dtype. Setting complete filters to zero preserves those dimensions. Physical filter
pruning constructs a smaller convolution and propagates the selected channels to the
next layer, allowing ordinary dense kernels to execute less work.

For **Filter Pruning: Making Convolution Physically Narrower**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *Why does zeroing filters differ from deleting them?* The lab
therefore changes the mechanism described below, retains its measured state, and names
the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict the output shapes of masked and physically pruned blocks.
2. Predict which candidate reduces analytical convolution work.
3. Identify the exact slice that must be applied to the second convolution.

Before opening Lesson 07's retained output, answer the first prompt— *Predict the output
shapes of masked and physically pruned blocks.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

Two consecutive convolutions form the concrete dependency. One candidate masks half of
the first layer's output filters; another physically copies the retained filters and the
matching input-channel slices of the second convolution.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Filter zeros preserve the dense convolution descriptor. |
| 2 | Physical pruning changes two coupled channel dimensions. |
| 3 | Equivalence should be checked before a latency claim. |

Lesson 07 tracks three layers through Filter Pruning: Making Convolution Physically
Narrower: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

For a convolution, leading work scales with `N × Hout × Wout × Cout × Cin × Kh × Kw`. A
zeroed filter leaves Cout unchanged in the operator descriptor. Deleting it halves the
first layer's Cout and the next layer's Cin when dependencies are propagated. Copying
the same retained weights provides an equivalence check between masked and narrowed
functions before timing.

The inspectable invariant for **Filter Pruning: Making Convolution Physically Narrower**
is tested by: Mask and physically remove the same convolution filters, then compare
equivalence, parameters, FLOPs, and CUDA latency. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Mask and physically remove the same convolution filters, then compare equivalence, parameters, FLOPs, and CUDA latency.

| Experimental role | Frozen definition |
|---|---|
| Baseline | same-shape filter-masked two-convolution block |
| Candidate | physically narrowed block with propagated second-layer input channels |
| Held constant | input tensor, retained filter indices, copied weights, batch, spatial shape, dtype, and timing |
| Measurements | output max error, parameters, analytical FLOPs, median latency, and channel shapes |
| Evidence label | `pytorch-gpu` |

This Lesson 07 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **input tensor, retained filter indices, copied weights, batch, spatial
shape, dtype, and timing**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The physical model is rebuilt with smaller module dimensions and receives exact weight
slices from the masked model. Summing the retained first-layer channels into the second
layer is not approximated: the matching input-channel axis is sliced. Near-zero output
drift verifies the dependency before speed and parameter numbers are interpreted.

For **Filter Pruning: Making Convolution Physically Narrower**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements physically
narrowed block with propagated second-layer input channels and records output max error,
parameters, analytical FLOPs, median latency, and channel shapes. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Masked parameters | 11,520 |
| Narrow parameters | 5,760 |
| FLOP reduction | 50.00% |
| Equivalence max error | 0.001953 |
| Masked median | 0.056320 ms |
| Narrow median | 0.044656 ms |

### What the numbers mean

Masking retained 11,520 parameters, while physical propagation reduced the block to
5,760 and analytical convolution work by 50.0%. The copied narrow block matched the
masked control within 1.953e-03. Median latency changed from 0.056320 to 0.044656 ms on
this shape.

Lesson 07's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Filter pruning accelerates ordinary dense convolution only after channel dimensions are physically rebuilt and propagated.

### Acceptance and rollback gate

Accept structural filter pruning only when all consumer shapes are updated, functional
drift is understood, and the target runtime improves under representative spatial sizes.

The gate for **Filter Pruning: Making Convolution Physically Narrower** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

Selecting channels independently in adjacent convolutions breaks equivalence. BatchNorm,
residual adds, groups, and concatenations introduce additional dependencies not present
in this two-layer probe. Awkward channel counts can also reduce kernel efficiency
despite lower FLOPs.

## 6. Follow the theory inside the notebook

In Lesson 07's [`lab.ipynb`](lab.ipynb), first identify **same-shape filter-masked
two-convolution block** and **physically narrowed block with propagated second-layer
input channels** without running them. Next inspect the dimensions or lifecycle state
that implements the derivation. After **Run All**, verify the RTX 5090 environment and
the frozen fields before reconciling the result table with the artifact.

The reader loop for **Filter Pruning: Making Convolution Physically Narrower** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/07-filter-pruning/lab.ipynb
```

To reproduce **Filter Pruning: Making Convolution Physically Narrower**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Insert BatchNorm and a residual branch, then use a dependency graph to enumerate every
coupled slice. Sweep retained widths that align and misalign with the target convolution
backend.

For Lesson 07, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Filter Pruning: Making Convolution Physically Narrower** observation
belongs to Lesson 07's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [DepGraph paper](https://arxiv.org/abs/2301.12900)
- [Torch-Pruning reference implementation](https://github.com/VainF/Torch-Pruning)
