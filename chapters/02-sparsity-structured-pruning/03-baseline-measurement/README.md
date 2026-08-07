# Lesson 03 — Baseline Measurement: Parameters, FLOPs, Latency, and Throughput

> **Puzzle:** Which baseline numbers are required before a pruning result can be interpreted?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A pruning percentage without a baseline is not a comparison. Parameters and analytical
FLOPs describe model structure; median and tail latency describe a workload on one
stack; throughput and peak memory answer still different questions. A useful baseline
freezes all of them before changing the model.

For **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput**, the
engineering question is not whether a definition can be repeated; it is whether the
following claim survives a controlled GPU test: *Which baseline numbers are required
before a pruning result can be interpreted?* The lab therefore changes the mechanism
described below, retains its measured state, and names the evidence that would still be
needed for deployment.

## Predict before reading the result

1. Predict how batch 1 and batch 64 change latency and examples per second.
2. Explain why lower FLOPs does not mathematically guarantee lower p95 latency.
3. List every environment field required to compare a later pruned run.

Before opening Lesson 03's retained output, answer the first prompt— *Predict how batch
1 and batch 64 change latency and examples per second.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

The concrete system is a three-layer CUDA MLP, two batch sizes, a known dtype, a fixed
random input, a parameter counter, an analytical linear-FLOP ledger, repeated CUDA-event
samples, and peak allocated memory.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Structural metrics are deterministic for a frozen graph. |
| 2 | Latency and throughput depend on the workload and timing protocol. |
| 3 | Tail statistics require repeated samples, not one synchronized call. |

Lesson 03 tracks three layers through Baseline Measurement: Parameters, FLOPs, Latency,
and Throughput: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

For linear layers, parameters are `in_features × out_features` plus bias and leading
multiply-add work is `2 × batch × in × out`. These values are deterministic properties
of the chosen shape. Latency is a distribution affected by warm-up, synchronization, and
batch; throughput is `batch / elapsed_time` and cannot be inferred from a single-request
timing. Peak allocated memory must be reset and sampled over the same measurement
window.

The inspectable invariant for **Baseline Measurement: Parameters, FLOPs, Latency, and
Throughput** is tested by: Record a complete dense MLP baseline at batch 1 and batch 64
with structural and runtime metrics. Its purpose is to prevent the specific category
error behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Record a complete dense MLP baseline at batch 1 and batch 64 with structural and runtime metrics.

| Experimental role | Frozen definition |
|---|---|
| Baseline | the same dense MLP evaluated at batch 1 |
| Candidate | the same dense MLP evaluated at batch 64 |
| Held constant | model weights, hidden sizes, dtype, GPU, warm-up, repetitions, and input distribution |
| Measurements | parameters, analytical FLOPs, median/p95 latency, throughput, and peak allocated memory |
| Evidence label | `pytorch-gpu` |

This Lesson 03 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **model weights, hidden sizes, dtype, GPU, warm-up, repetitions, and input
distribution**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

The notebook computes the structural ledger directly from module shapes, then uses the
same timing helper for both batches. CUDA synchronization happens after the event pair,
and the result retains every sample so p95 can be recomputed. The batch comparison is
not a candidate victory; it demonstrates why service workload belongs in the baseline
identity.

For **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput**, the
environment cell asserts CUDA and fixes a lesson-specific seed. The experiment cell
implements the same dense MLP evaluated at batch 64 and records parameters, analytical
FLOPs, median/p95 latency, throughput, and peak allocated memory. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Parameters | 4,459,776 |
| Batch-1 FLOPs | 8,912,896 |
| Batch-1 median | 0.056288 ms |
| Batch-1 p95 | 0.059221 ms |
| Batch-64 median | 0.056608 ms |
| Batch-64 throughput | 1,130,582.3/s |
| Peak memory | 41.133 MiB |

### What the numbers mean

The frozen MLP contains 4,459,776 parameters and 8,912,896 leading linear FLOPs at batch
1. Batch 1 measured median/p95 0.056288/0.059221 ms, while batch 64 measured 0.056608 ms
and 1130582.3 examples/s. The batch field therefore changes the meaning of the
performance baseline even though the parameters are identical.

Lesson 03's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Parameters, FLOPs, latency, throughput, and memory are complementary baseline fields, not interchangeable compression scores.

### Acceptance and rollback gate

Reject any pruning comparison that cannot reproduce the dense baseline within a
predefined tolerance on the same hardware and software stack.

The gate for **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput** is
stricter than “the code ran” because it binds this lesson's tensor or model identity,
quality tolerance, workload, runtime path, and rollback evidence. A missing optional
package can settle a compatibility question, but it cannot satisfy the
native-performance decision stated above.

### How this conclusion can fail

Timing before warm-up can include allocator and kernel initialization. Dividing batch by
host wall time without synchronization can overstate throughput. Peak memory from an
earlier operation can contaminate the window. A baseline report should make each of
these failure modes auditable.

## 6. Follow the theory inside the notebook

In Lesson 03's [`lab.ipynb`](lab.ipynb), first identify **the same dense MLP evaluated
at batch 1** and **the same dense MLP evaluated at batch 64** without running them. Next
inspect the dimensions or lifecycle state that implements the derivation. After **Run
All**, verify the RTX 5090 environment and the frozen fields before reconciling the
result table with the artifact.

The reader loop for **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput**
is **predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/03-baseline-measurement/lab.ipynb
```

To reproduce **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput**, use a
PyTorch build compiled for the target GPU and select `Run All`. Compare the measurements
in the frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Add power, cold-start, and operator traces, then repeat across a batch/sequence grid.
Use confidence intervals or repeated runs when the acceptance margin is close to noise.

For Lesson 03, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Baseline Measurement: Parameters, FLOPs, Latency, and Throughput**
observation belongs to Lesson 03's RTX 5090 environment, shapes, seed, and protocol. It
does not establish the unmeasured task quality or platform properties named in the
failure analysis. This independently written tutorial uses the study topic as a
question, without redistributing source HTML, model weights, private paths, or
infrastructure.

## References

- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
