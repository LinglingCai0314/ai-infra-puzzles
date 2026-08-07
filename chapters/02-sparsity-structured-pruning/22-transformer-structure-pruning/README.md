# Lesson 22 — Pruning Transformer Heads, FFN Neurons, and Layers

> **Puzzle:** Which structural unit changes Transformer compute rather than only masking values?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Attention heads, FFN intermediate neurons, hidden dimensions, and whole layers are
different dependency units. Masking a head preserves the packed QKV and output
projection shapes, while physically reducing FFN width changes dense GEMMs. Whole-layer
removal changes depth and residual composition. Each route needs its own quality and
latency evidence.

For **Pruning Transformer Heads, FFN Neurons, and Layers**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Which structural unit changes Transformer compute rather than only
masking values?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict which candidate changes physical parameter count.
2. Estimate the relative attention and FFN work for the chosen S, D, and D_ff.
3. Predict which route has the largest output drift before recovery.

Before opening Lesson 22's retained output, answer the first prompt— *Predict which
candidate changes physical parameter count.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A compact pre-norm Transformer block exposes head outputs, an FFN intermediate,
residuals, and a two-block stack. The lab compares a head mask, a half-width physical
FFN, and layer skipping under one CUDA workload.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | A head mask is not automatically a narrower attention operator. |
| 2 | FFN width directly controls two dense matrix multiplications. |
| 3 | Layer pruning changes depth and residual transformations. |

Lesson 22 tracks three layers through Pruning Transformer Heads, FFN Neurons, and
Layers: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

For sequence length S and hidden width D, attention projections scale roughly with `S
D²` and score/value work with `S² D`; FFN work scales with `S D D_ff`. Removing a
logical head but retaining packed D-wide projections may leave most work unchanged.
Halving D_ff directly reduces two GEMM dimensions. Removing a block deletes both
attention and FFN work but creates a larger functional perturbation. Structural claims
must specify which dimensions changed.

The inspectable invariant for **Pruning Transformer Heads, FFN Neurons, and Layers** is
tested by: Measure head masking, physical FFN narrowing, and whole-layer skipping in a
compact CUDA Transformer. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Measure head masking, physical FFN narrowing, and whole-layer skipping in a compact CUDA Transformer.

| Experimental role | Frozen definition |
|---|---|
| Baseline | full block/stack and same-shape attention-head masking |
| Candidate | physically narrowed FFN and one-layer-shorter stack |
| Held constant | weights where comparable, input, sequence length, batch, hidden width, dtype, eval mode, and timing |
| Measurements | physical parameters, output RMSE/cosine, median latency, and theoretical work components |
| Evidence label | `pytorch-gpu` |

This Lesson 22 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **weights where comparable, input, sequence length, batch, hidden width,
dtype, eval mode, and timing**. That frozen condition preserves the dependency or
runtime boundary at issue; the small scale limits transfer to larger models but does not
permit the baseline and candidate to answer different questions.

### Code walk-through

The block returns a head-mask path without rewriting packed projections, making its
unchanged physical shape visible. The FFN candidate copies selected intermediate
rows/columns into smaller linear modules. Layer skipping reuses the first block output.
These controls keep three pruning units conceptually separate.

For **Pruning Transformer Heads, FFN Neurons, and Layers**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements physically
narrowed FFN and one-layer-shorter stack and records physical parameters, output
RMSE/cosine, median latency, and theoretical work components. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Full parameters | 49,984 |
| FFN-narrow parameters | 33,472 |
| Head-mask RMSE | 0.037988 |
| FFN-narrow RMSE | 0.146989 |
| Layer-skip RMSE | 0.227471 |
| Full median | 0.153360 ms |
| FFN-narrow median | 0.211440 ms |

### What the numbers mean

Masking two of four heads preserved 49,984 parameters and measured 0.159792 ms versus
0.153360 ms for the full block. Physical FFN narrowing reduced parameters to 33,472,
measured 0.211440 ms, and introduced RMSE 0.146989. Layer skipping had RMSE 0.227471
before recovery.

Lesson 22's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Transformer pruning must name the structural unit and prove its physical compute path; masks alone are insufficient.

### Acceptance and rollback gate

Accept a Transformer structure only after task/perplexity gates and a runtime trace
confirm that the intended dimensions or depth changed.

The gate for **Pruning Transformer Heads, FFN Neurons, and Layers** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

Random weights do not reveal head redundancy. Fused attention kernels may require fixed
head dimensions or grouped-query layouts, and KV-cache shape couples attention structure
to serving memory. Layer removal can shift normalization statistics and generation
behavior.

## 6. Follow the theory inside the notebook

In Lesson 22's [`lab.ipynb`](lab.ipynb), first identify **full block/stack and
same-shape attention-head masking** and **physically narrowed FFN and one-layer-shorter
stack** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Pruning Transformer Heads, FFN Neurons, and Layers** is **predict
→ execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/22-transformer-structure-pruning/lab.ipynb
```

To reproduce **Pruning Transformer Heads, FFN Neurons, and Layers**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Repeat on a pretrained encoder or decoder, evaluate task quality/perplexity and KV-cache
bytes, and use a backend with explicit variable-head or narrowed-FFN support.

For Lesson 22, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Pruning Transformer Heads, FFN Neurons, and Layers** observation
belongs to Lesson 22's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)
- [DepGraph paper](https://arxiv.org/abs/2301.12900)
