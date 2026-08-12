# Lesson 01 — Pruning Objectives, Constraints, and Delivery Boundaries

> **Puzzle:** If half the weights become zero, has a mobile deployment objective been achieved?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Pruning is an engineering change with several possible objectives: package size,
resident memory, first-token or first-frame latency, steady-state throughput, energy,
and hardware cost. A sparsity percentage answers none of them by itself. The first
deliverable is therefore a target card that connects one model revision and workload to
a measurable deployment gate and a rollback condition.

## Predict before reading the result

1. Predict whether a masked 50% sparse matrix will materially beat its dense copy in ordinary dense GEMM.
2. Predict how the narrower layer changes parameters, FLOPs, and output shape.
3. Write one acceptance gate and one rollback gate for a latency-driven project.

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

## 2. Derive the mechanism

For a dense matrix multiplication, leading work is approximately `2MKN`. Replacing half
of W with zeros leaves M, N, and K unchanged when the operator still dispatches a dense
GEMM. Physically reducing the output width changes N and therefore both arithmetic and
output storage. A valid target card distinguishes logical sparsity, serialized
representation, physical shape, kernel path, and end-to-end metric. This is why a
parameter-count goal and an 80 ms first-frame SLO are related but not interchangeable.

### Mechanism at a glance

```mermaid
flowchart LR
  V["Value state<br/>which entries are zero?"] --> R["Representation state<br/>what is stored?"]
  R --> S["Shape state<br/>which axes changed?"]
  S --> E["Execution state<br/>which operator ran?"]
  E --> P["Product metric<br/>did the target improve?"]
```

### Walk it step by step

1. **Name the product objective.** Choose package size, memory, latency, throughput, energy, or cost and attach a measurable gate.
2. **Locate the structural change.** Distinguish zeros in a tensor from a changed tensor shape or stored representation.
3. **Locate the execution change.** Confirm whether the runtime dispatched a smaller dense operator or a supported sparse operator.
4. **Accept on the original objective.** A candidate succeeds only when quality and the named deployment metric both pass.

## 3. Translate the theory into an experiment

**Experiment:** Compare dense, same-shape masked, and physically narrower CUDA linear layers under one timing protocol.

| Experimental role | Frozen definition |
|---|---|
| Baseline | dense BF16 linear layer at the original shape |
| Candidate | 50% same-shape masking and a 50% physically narrower dense layer |
| Held constant | GPU, input batch, input width, dtype, warm-up, repetitions, and random seed |
| Measurements | logical sparsity, physical parameters, median/p95 latency, and output width |
| Evidence label | `pytorch-gpu` |

### Code walk-through

The lab constructs all three candidates from one base weight tensor. The masked
candidate preserves the dense shape, while the narrow candidate copies a selected subset
of rows. CUDA events bracket only repeated forward calls after warm-up. Reading these
rows together shows which optimization changed values and which changed the work exposed
to the runtime.

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

## 5. Solve the puzzle and make a decision

> Zero weights satisfy a sparsity statistic; only a supported representation and a measured deployment path satisfy a performance objective.

### Acceptance and rollback gate

Accept a pruning route only if the deployment metric improves under the frozen workload
and the quality gate passes; otherwise keep the dense revision as the explicit rollback.

### How this conclusion can fail

A narrow microbenchmark can still mislead if first-frame setup, preprocessing, memory
allocation, or a mobile runtime dominates. Conversely, masking may compress well on disk
even when it does not accelerate the measured dense operator. Never transfer one
objective's success to another objective without new evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/01-pruning-objectives/lab.ipynb
```

## Extend the experiment

Add model serialization and a real deployment runtime, then repeat cold-start and
steady-state tests separately. Record the graph dimensions and operator trace beside the
latency table.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
