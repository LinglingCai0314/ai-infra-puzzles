# Lesson 01 — Pruning Objectives, Constraints, and Delivery Boundaries

> **Puzzle:** If half the weights become zero, has a mobile deployment objective been achieved?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Pruning is an engineering change with several possible objectives: package size,
resident memory, first-token or first-frame latency, steady-state throughput, energy,
and hardware cost. A sparsity percentage answers none of them by itself. The first
deliverable is therefore a target card that connects one model revision and workload to
a measurable deployment gate and a rollback condition.

For **Pruning Objectives, Constraints, and Delivery Boundaries**, the engineering
question is not whether a definition can be repeated; it is whether the following claim
survives a controlled GPU test: *If half the weights become zero, has a mobile
deployment objective been achieved?* The lab therefore changes the mechanism described
below, retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict whether a masked 50% sparse matrix will materially beat its dense copy in ordinary dense GEMM.
2. Predict how the narrower layer changes parameters, FLOPs, and output shape.
3. Write one acceptance gate and one rollback gate for a latency-driven project.

Before opening Lesson 01's retained output, answer the first prompt— *Predict whether a
masked 50% sparse matrix will materially beat its dense copy in ordinary dense
GEMM.*—and write one observation that would falsify the answer. If the result is already
visible, hide it and make the commitment first; otherwise this becomes post-hoc
explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

The concrete objects are a dense linear layer, a same-shape masked layer, a physically
narrower layer, their parameter tensors, the runtime input shape, and a latency
distribution. The mask changes values; the narrower layer changes dimensions and the
amount of dense work presented to the library.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Logical zeros do not imply fewer dense instructions. |
| 2 | A physical dimension change is visible to both the graph and the runtime. |
| 3 | The acceptance metric must match the deployment objective and workload. |

Lesson 01 tracks three layers through Pruning Objectives, Constraints, and Delivery
Boundaries: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

For a dense matrix multiplication, leading work is approximately `2MKN`. Replacing half
of W with zeros leaves M, N, and K unchanged when the operator still dispatches a dense
GEMM. Physically reducing the output width changes N and therefore both arithmetic and
output storage. A valid target card distinguishes logical sparsity, serialized
representation, physical shape, kernel path, and end-to-end metric. This is why a
parameter-count goal and an 80 ms first-frame SLO are related but not interchangeable.

The inspectable invariant for **Pruning Objectives, Constraints, and Delivery
Boundaries** is tested by: Compare dense, same-shape masked, and physically narrower
CUDA linear layers under one timing protocol. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Compare dense, same-shape masked, and physically narrower CUDA linear layers under one timing protocol.

| Experimental role | Frozen definition |
|---|---|
| Baseline | dense BF16 linear layer at the original shape |
| Candidate | 50% same-shape masking and a 50% physically narrower dense layer |
| Held constant | GPU, input batch, input width, dtype, warm-up, repetitions, and random seed |
| Measurements | logical sparsity, physical parameters, median/p95 latency, and output width |
| Evidence label | `pytorch-gpu` |

This Lesson 01 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **GPU, input batch, input width, dtype, warm-up, repetitions, and random
seed**. That frozen condition preserves the dependency or runtime boundary at issue; the
small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The lab constructs all three candidates from one base weight tensor. The masked
candidate preserves the dense shape, while the narrow candidate copies a selected subset
of rows. CUDA events bracket only repeated forward calls after warm-up. Reading these
rows together shows which optimization changed values and which changed the work exposed
to the runtime.

For **Pruning Objectives, Constraints, and Delivery Boundaries**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements 50%
same-shape masking and a 50% physically narrower dense layer and records logical
sparsity, physical parameters, median/p95 latency, and output width. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense parameters | 4,194,304 |
| Masked logical sparsity | 50.00% |
| Narrow parameters | 2,097,152 |
| Dense median | 0.020192 ms |
| Masked median | 0.020112 ms |
| Narrow median | 0.019808 ms |

### What the numbers mean

The mask created 50.0% logical sparsity but kept 4,194,304 dense parameters and a
2048-wide output. Its median was 0.020112 ms versus 0.020192 ms. The physical candidate
reduced parameters to 2,097,152, output width to 1024, and measured 0.019808 ms. These
numbers answer this CUDA operator workload only; they do not establish mobile
first-frame latency.

Lesson 01's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Zero weights satisfy a sparsity statistic; only a supported representation and a measured deployment path satisfy a performance objective.

### Acceptance and rollback gate

Accept a pruning route only if the deployment metric improves under the frozen workload
and the quality gate passes; otherwise keep the dense revision as the explicit rollback.

The gate for **Pruning Objectives, Constraints, and Delivery Boundaries** is stricter
than “the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

A narrow microbenchmark can still mislead if first-frame setup, preprocessing, memory
allocation, or a mobile runtime dominates. Conversely, masking may compress well on disk
even when it does not accelerate the measured dense operator. Never transfer one
objective's success to another objective without new evidence.

## 6. Follow the theory inside the notebook

In Lesson 01's [`lab.ipynb`](lab.ipynb), first identify **dense BF16 linear layer at the
original shape** and **50% same-shape masking and a 50% physically narrower dense
layer** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Pruning Objectives, Constraints, and Delivery Boundaries** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/01-pruning-objectives/lab.ipynb
```

To reproduce **Pruning Objectives, Constraints, and Delivery Boundaries**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Add model serialization and a real deployment runtime, then repeat cold-start and
steady-state tests separately. Record the graph dimensions and operator trace beside the
latency table.

For Lesson 01, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Pruning Objectives, Constraints, and Delivery Boundaries** observation
belongs to Lesson 01's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
