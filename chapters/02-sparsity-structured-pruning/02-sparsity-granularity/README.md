# Lesson 02 — The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M

> **Puzzle:** Can two tensors with exactly 50% zeros demand different kernels and deployment formats?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

The word sparsity hides a layout contract. Unstructured zeros, contiguous blocks,
removed channels, and 2:4 groups can share the same global nonzero rate while exposing
very different metadata, vectorization, and library opportunities. Choosing a
granularity is therefore a joint algorithm-runtime decision, not a cosmetic choice made
after training.

For **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M**, the
engineering question is not whether a definition can be repeated; it is whether the
following claim survives a controlled GPU test: *Can two tensors with exactly 50% zeros
demand different kernels and deployment formats?* The lab therefore changes the
mechanism described below, retains its measured state, and names the evidence that would
still be needed for deployment.

## Predict before reading the result

1. Predict which 50% mask will pass an exact 2:4 compliance check.
2. Predict whether ordinary dense matmul notices unstructured or block zeros.
3. Choose a granularity when custom kernels are forbidden.

Before opening Lesson 02's retained output, answer the first prompt— *Predict which 50%
mask will pass an exact 2:4 compliance check.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

The lab uses one weight matrix and derives four representations: unstructured magnitude
zeros, block zeros, exact 2:4 groups, and a physically narrowed matrix. It tracks global
sparsity, 2:4 compliance, shape, and dense-path latency.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Equal nonzero counts do not imply equal layouts. |
| 2 | N:M compliance is a local invariant, not a global percentage. |
| 3 | Channel removal can use a smaller dense operator without sparse metadata. |

Lesson 02 tracks three layers through The Sparsity Granularity Spectrum: Weights,
Channels, Blocks, and N:M: *value state* says which entries are zero, *shape state* says
which axes physically changed, and *execution state* says which operator actually ran.
The anchors above identify where this lesson's claim lives, so a zero count cannot
silently turn into a latency claim.

## 2. Derive the mechanism

A global rate `1 - nnz/numel` discards where the nonzeros live. For 2:4 sparsity, every
consecutive group of four along the contracted dimension must contain exactly two
retained values; 50% zeros placed elsewhere are non-compliant. Block sparsity adds a
block shape and index structure. Channel pruning removes a complete axis and can reuse
dense kernels at a smaller dimension. Runtime value comes from matching one of these
contracts to an implementation.

The inspectable invariant for **The Sparsity Granularity Spectrum: Weights, Channels,
Blocks, and N:M** is tested by: Construct four 50%-budget representations and compare
compliance, shape, and ordinary dense CUDA timing. Its purpose is to prevent the
specific category error behind this puzzle. An algorithmic change, a stored
representation, and a runtime observation remain separate until the candidate and
measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Construct four 50%-budget representations and compare compliance, shape, and ordinary dense CUDA timing.

| Experimental role | Frozen definition |
|---|---|
| Baseline | original dense matrix and unstructured 50% magnitude mask |
| Candidate | block mask, exact 2:4 mask, and a physically narrowed dense matrix |
| Held constant | source weights, input batch, dtype, target zero budget, and timing method |
| Measurements | global sparsity, 2:4 compliance, physical shape, and median latency |
| Evidence label | `pytorch-gpu` |

This Lesson 02 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **source weights, input batch, dtype, target zero budget, and timing
method**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

Each mask is generated explicitly so its local structure can be inspected. The
experiment intentionally multiplies masked tensors through the ordinary dense PyTorch
path; it does not claim cuSPARSELt dispatch. The narrow candidate changes the contracted
work and provides a useful control for the claim that structure, not zero count, is what
the kernel sees.

For **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M**, the
environment cell asserts CUDA and fixes a lesson-specific seed. The experiment cell
implements block mask, exact 2:4 mask, and a physically narrowed dense matrix and
records global sparsity, 2:4 compliance, physical shape, and median latency. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Unstructured sparsity | 50.00% |
| 2:4 compliance | 100.00% |
| Dense median | 0.017920 ms |
| Unstructured median | 0.017888 ms |
| 2:4 dense-path median | 0.018384 ms |
| Narrow median | 0.018480 ms |

### What the numbers mean

All three masks were near 50% sparse, but exact 2:4 compliance was 100.0% versus 37.5%
for the unstructured mask. The ordinary dense path measured 0.017920 ms for dense,
0.017888 ms for unstructured, and 0.018384 ms for compliant values. Only the narrow
control changed the matrix shape; no sparse-kernel dispatch is inferred from these
timings.

Lesson 02's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Sparsity granularity is an interface between optimization and execution; a global zero rate is only one field of that interface.

### Acceptance and rollback gate

Select a granularity only after the target runtime's supported patterns and the model's
accuracy sensitivity are both written down.

The gate for **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M**
is stricter than “the code ran” because it binds this lesson's tensor or model identity,
quality tolerance, workload, runtime path, and rollback evidence. A missing optional
package can settle a compatibility question, but it cannot satisfy the
native-performance decision stated above.

### How this conclusion can fail

A 2:4-compliant tensor can still use a dense tactic when it is not compressed into the
required backend format, dtype, alignment, or build flag. A channel-pruned tensor can
also be slower at awkward widths. Compliance is necessary for some paths, never
sufficient for speed.

## 6. Follow the theory inside the notebook

In Lesson 02's [`lab.ipynb`](lab.ipynb), first identify **original dense matrix and
unstructured 50% magnitude mask** and **block mask, exact 2:4 mask, and a physically
narrowed dense matrix** without running them. Next inspect the dimensions or lifecycle
state that implements the derivation. After **Run All**, verify the RTX 5090 environment
and the frozen fields before reconciling the result table with the artifact.

The reader loop for **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and
N:M** is **predict → execute → inspect → explain → decide**. Transferring its final
number to another architecture, workload shape, or backend requires a new run because
those variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/02-sparsity-granularity/lab.ipynb
```

To reproduce **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M**,
use a PyTorch build compiled for the target GPU and select `Run All`. Compare the
measurements in the frozen protocol with the checked-in artifact. If this lesson touches
an optional toolchain, install that named backend before claiming native execution;
otherwise only the compatibility fields are valid.

## Extend the experiment

Run the compliant matrix through cuSPARSELt or TensorRT, capture its tactic log, and
sweep dimensions around alignment boundaries while holding the nonzero budget fixed.

For Lesson 02, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **The Sparsity Granularity Spectrum: Weights, Channels, Blocks, and N:M**
observation belongs to Lesson 02's RTX 5090 environment, shapes, seed, and protocol. It
does not establish the unmeasured task quality or platform properties named in the
failure analysis. This independently written tutorial uses the study topic as a
question, without redistributing source HTML, model weights, private paths, or
infrastructure.

## References

- [NVIDIA cuSPARSELt documentation](https://docs.nvidia.com/cuda/cusparselt/)
- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
