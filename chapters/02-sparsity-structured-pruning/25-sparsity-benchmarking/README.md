# Lesson 25 — Benchmarking Sparsity: Proving a Real Speedup

> **Puzzle:** Which benchmark prevents a lower mean from hiding an unchanged p99 or memory peak?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Sparse acceleration is a runtime claim. The protocol must freeze shapes, batches, dtype,
warm-up, synchronization, sample window, power state, and backend. It should report a
distribution and throughput, plus memory and operator evidence, rather than a single
average.

For **Benchmarking Sparsity: Proving a Real Speedup**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Which benchmark prevents a lower mean from hiding an unchanged p99
or memory peak?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict which candidate changes dense operator dimensions.
2. Explain why 20 samples are weak evidence for p99.
3. Choose a warm-up and sampling protocol before reading results.

Before opening Lesson 25's retained output, answer the first prompt— *Predict which
candidate changes dense operator dimensions.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

Dense, same-shape masked, and physically narrowed linear blocks are timed with retained
per-iteration CUDA-event samples at batch 1 and batch 64. Median, p95, p99, throughput,
and peak memory are computed.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Benchmark identity includes workload, backend, and timing semantics. |
| 2 | Mean, median, and tail latency can rank candidates differently. |
| 3 | Masked dense and physically narrow controls distinguish zeros from less work. |

Lesson 25 tracks three layers through Benchmarking Sparsity: Proving a Real Speedup:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

GPU work is asynchronous, so host timing without synchronization measures enqueue cost.
Warm-up absorbs initialization and algorithm selection. Percentiles require sorted
repeated samples; `p99` is unstable with too few points. Throughput is workload
completed per unit time and should be measured at the serving batch, not derived from
peak FLOPs. Peak memory must be reset around each candidate.

The inspectable invariant for **Benchmarking Sparsity: Proving a Real Speedup** is
tested by: Benchmark dense, masked, and narrowed candidates across latency and
throughput batches with retained samples. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Benchmark dense, masked, and narrowed candidates across latency and throughput batches with retained samples.

| Experimental role | Frozen definition |
|---|---|
| Baseline | dense full-width and same-shape 75%-masked dense execution |
| Candidate | physically quarter-width dense execution |
| Held constant | GPU, clocks as observed, shapes, weights, dtype, batches, warm-up, samples, and synchronization |
| Measurements | p50/p95/p99 latency, throughput, peak memory, shape, and speedup |
| Evidence label | `pytorch-gpu` |

This Lesson 25 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **GPU, clocks as observed, shapes, weights, dtype, batches, warm-up, samples,
and synchronization**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The timing helper allocates tensors before measurement, warms each candidate, and
records individual CUDA-event durations. Summary functions preserve raw samples in the
JSON artifact. A separate large batch prevents single-request latency from masquerading
as service throughput.

For **Benchmarking Sparsity: Proving a Real Speedup**, the environment cell asserts CUDA
and fixes a lesson-specific seed. The experiment cell implements physically
quarter-width dense execution and records p50/p95/p99 latency, throughput, peak memory,
shape, and speedup. The artifact cell serializes those same fields. Only
optional-backend import or API failures become compatibility evidence; an error in the
core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense p50 | 0.017056 ms |
| Dense p99 | 0.022265 ms |
| Masked p50 | 0.017136 ms |
| Narrow p50 | 0.013856 ms |
| Narrow p99 | 0.015994 ms |
| Batch-64 speedup | 0.982x |
| Samples | 80 |

### What the numbers mean

At batch 1, dense p50/p99 were 0.017056/0.022265 ms, the same-shape masked candidate was
0.017136/0.018128 ms, and the physical narrow candidate was 0.013856/0.015994 ms. At
batch 64 the narrow/full median ratio was 0.982x.

Lesson 25's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> A sparsity speedup is a distribution measured on the intended execution path, not a zero count or a best-case sample.

### Acceptance and rollback gate

Accept a speedup only when the matched baseline, tail gate, throughput gate, memory
gate, and operator/shape evidence all meet the frozen protocol.

The gate for **Benchmarking Sparsity: Proving a Real Speedup** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Shared GPU load, dynamic clocks, allocator history, and insufficient samples can move
tails. Microbenchmarks omit data movement and service queues. A narrower toy layer
cannot prove an end-to-end model gain.

## 6. Follow the theory inside the notebook

In Lesson 25's [`lab.ipynb`](lab.ipynb), first identify **dense full-width and
same-shape 75%-masked dense execution** and **physically quarter-width dense execution**
without running them. Next inspect the dimensions or lifecycle state that implements the
derivation. After **Run All**, verify the RTX 5090 environment and the frozen fields
before reconciling the result table with the artifact.

The reader loop for **Benchmarking Sparsity: Proving a Real Speedup** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/25-sparsity-benchmarking/lab.ipynb
```

To reproduce **Benchmarking Sparsity: Proving a Real Speedup**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Repeat in an isolated process, capture Nsight or profiler operator names, add confidence
intervals, and run a representative end-to-end service load with request arrivals.

For Lesson 25, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Benchmarking Sparsity: Proving a Real Speedup** observation belongs to
Lesson 25's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
