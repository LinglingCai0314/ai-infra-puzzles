# Lesson 20 — CNN Case Study: ResNet Channel Pruning

> **Puzzle:** Why can a ResNet-like block lose parameters without reaching the expected throughput?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Residual networks tie channel width to additions and projection shortcuts. A safe case
study must rebuild a whole stage-compatible block, preserve the add contract, update the
classifier or downstream consumer, and benchmark several batches. The percentage of
removed channels is only the starting point.

For **CNN Case Study: ResNet Channel Pruning**, the engineering question is not whether
a definition can be repeated; it is whether the following claim survives a controlled
GPU test: *Why can a ResNet-like block lose parameters without reaching the expected
throughput?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict every module dimension changed by halving the stage width.
2. Predict whether the percentage FLOP and latency reductions match exactly.
3. Choose batch-specific gates for interactive and throughput services.

Before opening Lesson 20's retained output, answer the first prompt— *Predict every
module dimension changed by halving the stage width.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A compact ResNet-style stem, residual block with projection, global pooling, and
classifier is built in full and narrow variants. Weights are copied by retained indices
where functions align, and structural FLOPs, parameters, output error, and latency are
measured.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Residual additions impose equal output widths. |
| 2 | Stage pruning propagates into later layers and the classifier. |
| 3 | FLOP reduction and throughput gain need separate measurements. |

Lesson 20 tracks three layers through CNN Case Study: ResNet Channel Pruning: *value
state* says which entries are zero, *shape state* says which axes physically changed,
and *execution state* says which operator actually ran. The anchors above identify where
this lesson's claim lives, so a zero count cannot silently turn into a latency claim.

## 2. Derive the mechanism

Within a basic residual block, both the main path's final convolution and shortcut
projection produce the same channel count. Narrowing the stage changes subsequent
convolutions and classifier input. FLOPs fall roughly with channel products, but latency
also depends on convolution algorithm, memory layout, launch overhead, and batch. A
batch-1 result and a batch-64 throughput result answer different deployment questions.

The inspectable invariant for **CNN Case Study: ResNet Channel Pruning** is tested by:
Construct full and half-width ResNet-like models and compare structure, parity control,
batch-1 latency, and batch-64 throughput. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Construct full and half-width ResNet-like models and compare structure, parity control, batch-1 latency, and batch-64 throughput.

| Experimental role | Frozen definition |
|---|---|
| Baseline | full-width ResNet-style stage |
| Candidate | physically half-width stage with synchronized main, shortcut, and classifier dimensions |
| Held constant | input resolution, stem, depth, retained indices, dtype, GPU, warm-up, repetitions, and batches |
| Measurements | parameters, analytical convolution/linear FLOPs, batch-1 latency, batch-64 throughput, and output drift |
| Evidence label | `pytorch-gpu` |

This Lesson 20 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **input resolution, stem, depth, retained indices, dtype, GPU, warm-up,
repetitions, and batches**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The model is intentionally small enough for a repeatable notebook while preserving the
dependency pattern that makes ResNet pruning nonlocal. Structural counters read actual
module shapes. Timing uses the same CUDA-event helper at both batches; the notebook does
not project ImageNet Top-1 or ResNet-50 speed from this mini-network.

For **CNN Case Study: ResNet Channel Pruning**, the environment cell asserts CUDA and
fixes a lesson-specific seed. The experiment cell implements physically half-width stage
with synchronized main, shortcut, and classifier dimensions and records parameters,
analytical convolution/linear FLOPs, batch-1 latency, batch-64 throughput, and output
drift. The artifact cell serializes those same fields. Only optional-backend import or
API failures become compatibility evidence; an error in the core comparison still fails
the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Full parameters | 20,650 |
| Narrow parameters | 5,466 |
| FLOP reduction | 73.94% |
| Batch-1 full median | 0.121744 ms |
| Batch-1 narrow median | 0.095600 ms |
| Batch-64 speedup | 1.007x |

### What the numbers mean

Halving stage width reduced parameters from 20,650 to 5,466 and analytical work by
73.9%. Batch-1 medians were 0.121744 versus 0.095600 ms; batch-64 measured a 1.007x
ratio. Random weights make this a systems/shape case study, not a Top-1 result.

Lesson 20's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> ResNet channel pruning is a stage-level graph transformation whose benefit must be measured at each target workload.

### Acceptance and rollback gate

Accept a ResNet pruning candidate only when stage dependencies, task quality, target
batches, and end-to-end runtime all pass against the exact baseline revision.

The gate for **CNN Case Study: ResNet Channel Pruning** is stricter than “the code ran”
because it binds this lesson's tensor or model identity, quality tolerance, workload,
runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Toy random weights make output drift a bookkeeping signal rather than a quality score.
Widths can cross library-alignment thresholds, and data loading or post-processing can
dominate a production service. A small block cannot prove ResNet-50 throughput.

## 6. Follow the theory inside the notebook

In Lesson 20's [`lab.ipynb`](lab.ipynb), first identify **full-width ResNet-style
stage** and **physically half-width stage with synchronized main, shortcut, and
classifier dimensions** without running them. Next inspect the dimensions or lifecycle
state that implements the derivation. After **Run All**, verify the RTX 5090 environment
and the frozen fields before reconciling the result table with the artifact.

The reader loop for **CNN Case Study: ResNet Channel Pruning** is **predict → execute →
inspect → explain → decide**. Transferring its final number to another architecture,
workload shape, or backend requires a new run because those variables sit outside this
lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/20-resnet-channel-pruning/lab.ipynb
```

To reproduce **CNN Case Study: ResNet Channel Pruning**, use a PyTorch build compiled
for the target GPU and select `Run All`. Compare the measurements in the frozen protocol
with the checked-in artifact. If this lesson touches an optional toolchain, install that
named backend before claiming native execution; otherwise only the compatibility fields
are valid.

## Extend the experiment

Apply the same ledger to a pretrained torchvision ResNet, calibrate importance on real
data, fine-tune, and profile operator shapes at production batches.

For Lesson 20, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **CNN Case Study: ResNet Channel Pruning** observation belongs to Lesson
20's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [DepGraph paper](https://arxiv.org/abs/2301.12900)
