# Lesson 13 — N:M Semi-structured Sparsity and the 2:4 Contract

> **Puzzle:** Does a tensor with 50% zeros qualify for Sparse Tensor Core execution?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

NVIDIA's 2:4 path imposes a local pattern: within each group of four values along the
required dimension, at least two are zero and the representation must be compressed for
a supported sparse GEMM. Global sparsity, pattern compliance, backend conversion, and
selected tactic are separate gates.

For **N:M Semi-structured Sparsity and the 2:4 Contract**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Does a tensor with 50% zeros qualify for Sparse Tensor Core
execution?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict the compliance of a random 50% mask.
2. Prove that top-2-of-4 masking reaches exactly 50% sparsity.
3. List the evidence needed after compliance before claiming acceleration.

Before opening Lesson 13's retained output, answer the first prompt— *Predict the
compliance of a random 50% mask.*—and write one observation that would falsify the
answer. If the result is already visible, hide it and make the commitment first;
otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A BF16 weight matrix, a random 50% mask, a magnitude-based exact 2:4 mask, a local
compliance checker, ordinary dense timing, and an optional PyTorch semi-structured
conversion probe are recorded.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | 2:4 is checked per local group, not over the whole tensor. |
| 2 | Pattern compliance precedes backend compression and tactic selection. |
| 3 | Dense-path timing cannot establish sparse Tensor Core execution. |

Lesson 13 tracks three layers through N:M Semi-structured Sparsity and the 2:4 Contract:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

Partition the contracted dimension into groups of four and retain the two largest
magnitudes per group. This guarantees exactly 2 nonzeros per group while preserving 50%
globally. A random half mask only satisfies a fraction of groups. Even a compliant
tensor remains dense storage until converted into the backend's compressed format, and
the hardware/library combination must support its shape and dtype.

The inspectable invariant for **N:M Semi-structured Sparsity and the 2:4 Contract** is
tested by: Compare random and exact 2:4 masks, then attempt a native semi-structured
conversion without hiding incompatibility. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Compare random and exact 2:4 masks, then attempt a native semi-structured conversion without hiding incompatibility.

| Experimental role | Frozen definition |
|---|---|
| Baseline | random global 50% sparsity executed through ordinary dense matmul |
| Candidate | exact magnitude 2:4 sparsity plus an optional native conversion probe |
| Held constant | source weights, shape, dtype, input, zero budget, GPU, and timing protocol |
| Measurements | global sparsity, local compliance, dense-path latency, conversion availability, and conversion error |
| Evidence label | `compatibility-probe` |

This Lesson 13 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **source weights, shape, dtype, input, zero budget, GPU, and timing
protocol**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The compliance function reshapes the K dimension into groups of four and counts
nonzeros. The native conversion attempt is wrapped and stores either a successful sparse
result or the exact exception text. Ordinary dense timing remains a control and is never
relabeled as a sparse-kernel benchmark.

For **N:M Semi-structured Sparsity and the 2:4 Contract**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements exact magnitude
2:4 sparsity plus an optional native conversion probe and records global sparsity, local
compliance, dense-path latency, conversion availability, and conversion error. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Random-mask compliance | 37.45% |
| 2:4 compliance | 100.00% |
| 2:4 sparsity | 50.00% |
| Dense baseline median | 0.018544 ms |
| 2:4 dense-path median | 0.018512 ms |
| Native conversion | no |

### What the numbers mean

Random 50% masking achieved 37.4% local compliance, while top-2-of-4 reached 100.0% at
50.0% sparsity. Ordinary dense-path medians were 0.018544 and 0.018512 ms. Native
semi-structured conversion success was False; the retained probe message is
`RuntimeError: cuSPARSELt not supported on your machine.`.

Lesson 13's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **compatibility-probe** evidence; the printed notebook
payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Exact 2:4 values are a necessary data invariant; native representation and tactic evidence complete the execution claim.

### Acceptance and rollback gate

Accept a 2:4 speed claim only after compliance, supported compression, a sparse operator
trace, numerical validation, and a matched dense baseline all pass.

The gate for **N:M Semi-structured Sparsity and the 2:4 Contract** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Zeros can be arranged along the wrong axis, shapes can violate alignment, and a library
can fall back to dense tactics. A failed PyTorch conversion on one stack does not imply
the GPU lacks all 2:4 support; it bounds only that API path.

## 6. Follow the theory inside the notebook

In Lesson 13's [`lab.ipynb`](lab.ipynb), first identify **random global 50% sparsity
executed through ordinary dense matmul** and **exact magnitude 2:4 sparsity plus an
optional native conversion probe** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **N:M Semi-structured Sparsity and the 2:4 Contract** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/13-nm-2-4-sparsity/lab.ipynb
```

To reproduce **N:M Semi-structured Sparsity and the 2:4 Contract**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Run the same compliant weights through cuSPARSELt or TensorRT, retain build logs and
kernel names, and sweep supported shapes and FP16/BF16/INT8 dtypes.

For Lesson 13, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The notebook records real package/API availability and preserves the native success or
failure state. Missing backend execution remains unmeasured.

The checked-in **N:M Semi-structured Sparsity and the 2:4 Contract** observation belongs
to Lesson 13's RTX 5090 environment, shapes, seed, and protocol. It does not establish
the unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [NVIDIA cuSPARSELt documentation](https://docs.nvidia.com/cuda/cusparselt/)
- [TensorRT sparsity requirements](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html)
