# Lesson 28 — Why Edge and Server Deployment Need Different Pruning Strategies

> **Puzzle:** Should one sparse checkpoint be expected to win on both a phone and a GPU service?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Edge devices often prioritize package bytes, cold start, peak memory, energy, and
standard mobile operators. GPU services prioritize batch throughput, tail latency,
concurrency, and kernel support. The same zeros can compress well for one platform and
execute as an unchanged dense operator on another.

For **Why Edge and Server Deployment Need Different Pruning Strategies**, the
engineering question is not whether a definition can be repeated; it is whether the
following claim survives a controlled GPU test: *Should one sparse checkpoint be
expected to win on both a phone and a GPU service?* The lab therefore changes the
mechanism described below, retains its measured state, and names the evidence that would
still be needed for deployment.

## Predict before reading the result

1. Predict which candidate has the smallest compressed weight payload.
2. Predict which candidate changes GPU dense GEMM dimensions.
3. Write separate acceptance gates for an edge app and a batched GPU service.

Before opening Lesson 28's retained output, answer the first prompt— *Predict which
candidate has the smallest compressed weight payload.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

The final lab combines measured RTX 5090 batch-1/batch-64 timing for dense, masked, and
physically narrowed candidates with a transparent storage ledger and a platform decision
matrix. Edge runtime numbers remain unmeasured.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Platform objectives weight storage, latency, throughput, and energy differently. |
| 2 | Compressed bytes do not predict GPU dense-path speed. |
| 3 | Unmeasured edge metrics must remain `not run` in the decision matrix. |

Lesson 28 tracks three layers through Why Edge and Server Deployment Need Different
Pruning Strategies: *value state* says which entries are zero, *shape state* says which
axes physically changed, and *execution state* says which operator actually ran. The
anchors above identify where this lesson's claim lives, so a zero count cannot silently
turn into a latency claim.

## 2. Derive the mechanism

A masked dense matrix can reduce compressed bytes because zeros have low entropy while
retaining M, N, and K on the GPU. A physically narrow model reduces dense arithmetic and
activation width but changes architecture and may need more recovery. On edge, supported
TFLite/OpenVINO operators and cold-start memory may dominate; on server, batching can
amortize launch overhead and expose GEMM efficiency. Each platform therefore has
distinct gates and can select a different candidate.

The inspectable invariant for **Why Edge and Server Deployment Need Different Pruning
Strategies** is tested by: Measure GPU candidates at interactive and throughput batches,
calculate storage representations, and derive platform-specific decisions without
inventing edge benchmarks. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Measure GPU candidates at interactive and throughput batches, calculate storage representations, and derive platform-specific decisions without inventing edge benchmarks.

| Experimental role | Frozen definition |
|---|---|
| Baseline | full-width dense and same-shape 75%-masked weight |
| Candidate | physically quarter-width dense candidate plus separate edge/server decision rows |
| Held constant | source weights, input widths, dtype, compression method, GPU timing, batches, and platform gate definitions |
| Measurements | raw/gzip bytes, batch-1 latency, batch-64 throughput, physical dimensions, edge evidence status, and platform decisions |
| Evidence label | `capacity-model` |

This Lesson 28 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **source weights, input widths, dtype, compression method, GPU timing,
batches, and platform gate definitions**. That frozen condition preserves the dependency
or runtime boundary at issue; the small scale limits transfer to larger models but does
not permit the baseline and candidate to answer different questions.

### Code walk-through

The notebook serializes identical candidate weights into raw in-memory payloads and
gzip-compresses them, then measures the CUDA operators. It populates the edge row with
storage facts but leaves device latency and energy unexecuted. The server row uses only
measured RTX 5090 evidence. This prevents cross-platform projection.

For **Why Edge and Server Deployment Need Different Pruning Strategies**, the
environment cell asserts CUDA and fixes a lesson-specific seed. The experiment cell
implements physically quarter-width dense candidate plus separate edge/server decision
rows and records raw/gzip bytes, batch-1 latency, batch-64 throughput, physical
dimensions, edge evidence status, and platform decisions. The artifact cell serializes
those same fields. Only optional-backend import or API failures become compatibility
evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense gzip bytes | 6,646,281 bytes |
| Masked gzip bytes | 2,789,948 bytes |
| Narrow gzip bytes | 1,661,842 bytes |
| GPU batch-1 dense | 0.018304 ms |
| GPU batch-1 narrow | 0.014224 ms |
| GPU batch-64 speedup | 0.981x |
| Edge runtime measured | no |

### What the numbers mean

Dense/masked/narrow gzip payloads were 6,646,281/2,789,948/1,661,842 bytes. On RTX 5090,
batch-1 medians were 0.018304/0.017760/0.014224 ms and the batch-64 physical-width ratio
was 0.981x. Edge latency and energy remain unmeasured, so the edge decision is
explicitly pending.

Lesson 28's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **capacity-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Sparsity strategy is platform-specific: storage evidence, edge execution, and server execution must remain separate until each is measured.

### Acceptance and rollback gate

Choose a platform candidate only when every metric required by that platform has native
evidence; otherwise leave the decision pending and preserve the dense rollback.

The gate for **Why Edge and Server Deployment Need Different Pruning Strategies** is
stricter than “the code ran” because it binds this lesson's tensor or model identity,
quality tolerance, workload, runtime path, and rollback evidence. A missing optional
package can settle a compatibility question, but it cannot satisfy the
native-performance decision stated above.

### How this conclusion can fail

Gzip is not a TFLite sparse encoding, RTX timing is not phone timing, and one server
batch does not represent concurrency. A physically narrow shape can also be unsupported
by a fixed mobile graph or misaligned on a GPU kernel.

## 6. Follow the theory inside the notebook

In Lesson 28's [`lab.ipynb`](lab.ipynb), first identify **full-width dense and
same-shape 75%-masked weight** and **physically quarter-width dense candidate plus
separate edge/server decision rows** without running them. Next inspect the dimensions
or lifecycle state that implements the derivation. After **Run All**, verify the RTX
5090 environment and the frozen fields before reconciling the result table with the
artifact.

The reader loop for **Why Edge and Server Deployment Need Different Pruning Strategies**
is **predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/28-edge-vs-server/lab.ipynb
```

To reproduce **Why Edge and Server Deployment Need Different Pruning Strategies**, use a
PyTorch build compiled for the target GPU and select `Run All`. Compare the measurements
in the frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Export all candidates to TFLite/OpenVINO and a server backend, benchmark the actual
phone/CPU/GPU targets including energy and concurrency, then compare total cost rather
than transferring proxy results.

For Lesson 28, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

Measured CUDA facts and transparent storage arithmetic feed a decision model; unmeasured
platform rows remain pending.

The checked-in **Why Edge and Server Deployment Need Different Pruning Strategies**
observation belongs to Lesson 28's RTX 5090 environment, shapes, seed, and protocol. It
does not establish the unmeasured task quality or platform properties named in the
failure analysis. This independently written tutorial uses the study topic as a
question, without redistributing source HTML, model weights, private paths, or
infrastructure.

## References

- [TensorFlow Lite model optimization](https://www.tensorflow.org/lite/performance/model_optimization)
- [TensorRT sparsity requirements](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html)
