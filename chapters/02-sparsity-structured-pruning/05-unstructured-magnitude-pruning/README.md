# Lesson 05 — Unstructured Magnitude Pruning Without Storage Myths

> **Puzzle:** Why can a model contain 80% zeros while its ordinary state_dict grows?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Unstructured magnitude pruning is easy to apply and useful for studying redundancy, but
PyTorch's training-time reparameterization stores the original parameter and a mask.
Logical zeros, raw checkpoint bytes, compressed bytes, and physical sparse storage are
four distinct quantities.

For **Unstructured Magnitude Pruning Without Storage Myths**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *Why can a model contain 80% zeros while its ordinary state_dict
grows?* The lab therefore changes the mechanism described below, retains its measured
state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict the state_dict keys immediately after PyTorch pruning.
2. Predict whether the raw serialized bytes shrink after `prune.remove`.
3. Predict which representation gzip compresses most effectively.

Before opening Lesson 05's retained output, answer the first prompt— *Predict the
state_dict keys immediately after PyTorch pruning.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

The experiment uses one linear module before pruning, after `l1_unstructured`, after
`prune.remove`, and after gzip compression. It inspects parameter names, buffers, zero
rate, forward equivalence, and serialized byte counts.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | A mask changes parameterization before it changes storage format. |
| 2 | Removing the reparameterization materializes zeros but keeps a dense tensor. |
| 3 | Raw and compressed checkpoint sizes answer different deployment questions. |

Lesson 05 tracks three layers through Unstructured Magnitude Pruning Without Storage
Myths: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

PyTorch pruning replaces `weight` with `weight_orig` and computes `weight_orig ×
weight_mask` through a pre-hook. The dense tensors still occupy dense storage, and the
additional mask can make an uncompressed state_dict larger. `prune.remove` materializes
the masked weight and deletes the reparameterization but does not convert it to CSR or
pack nonzeros. General-purpose compression can exploit repeated zero bytes, which
explains why compressed file size may fall while raw tensor storage does not.

The inspectable invariant for **Unstructured Magnitude Pruning Without Storage Myths**
is tested by: Trace an 80% magnitude mask through apply, save, remove, and compressed
serialization stages. Its purpose is to prevent the specific category error behind this
puzzle. An algorithmic change, a stored representation, and a runtime observation remain
separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Trace an 80% magnitude mask through apply, save, remove, and compressed serialization stages.

| Experimental role | Frozen definition |
|---|---|
| Baseline | the original dense linear layer state_dict |
| Candidate | PyTorch pruning reparameterization and the materialized masked weight |
| Held constant | same weight values, pruning amount, serializer, compression level, and module shape |
| Measurements | zero rate, state_dict keys, raw bytes, gzip bytes, and forward equivalence |
| Evidence label | `pytorch-gpu` |

This Lesson 05 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **same weight values, pruning amount, serializer, compression level, and
module shape**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

BytesIO keeps the serialization experiment inside memory and avoids path-dependent
artifacts. The notebook records keys before and after `remove`, evaluates the module
around the transition, and compresses the same byte payload. It does not call the
resulting file a sparse runtime format.

For **Unstructured Magnitude Pruning Without Storage Myths**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements PyTorch
pruning reparameterization and the materialized masked weight and records zero rate,
state_dict keys, raw bytes, gzip bytes, and forward equivalence. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Logical sparsity | 80.00% |
| Dense raw bytes | 4,195,945 bytes |
| Pruned-hook raw bytes | 8,390,501 bytes |
| Removed raw bytes | 4,195,945 bytes |
| Removed gzip bytes | 1,120,583 bytes |
| Remove max output drift | 0.000000 |

### What the numbers mean

The effective weight reached 80.0% sparsity. The hook checkpoint used keys
['weight_mask', 'weight_orig'] and occupied 8,390,501 raw bytes, while `remove` restored
a single key ['weight'] and 4,195,945 raw bytes. Gzip reduced the materialized payload
to 1,120,583 bytes. Forward drift across `remove` was 0.000e+00, proving lifecycle
equivalence but not sparse storage.

Lesson 05's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Magnitude pruning creates zeros; storage compression and runtime acceleration require additional explicit representations.

### Acceptance and rollback gate

Accept the mask lifecycle only when the saved keys, zero rate, load path, and intended
deployment representation are explicitly tested.

The gate for **Unstructured Magnitude Pruning Without Storage Myths** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

File systems and zip serialization can introduce version-dependent overhead, so tiny
tensors exaggerate metadata. Gzip size is not resident GPU memory and says nothing about
kernel speed. A deployment claiming sparse storage must identify the actual sparse
encoding and loader.

## 6. Follow the theory inside the notebook

In Lesson 05's [`lab.ipynb`](lab.ipynb), first identify **the original dense linear
layer state_dict** and **PyTorch pruning reparameterization and the materialized masked
weight** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Unstructured Magnitude Pruning Without Storage Myths** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/05-unstructured-magnitude-pruning/lab.ipynb
```

To reproduce **Unstructured Magnitude Pruning Without Storage Myths**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Convert the materialized matrix to CSR and compare metadata plus supported operations;
then load every saved variant into a fresh process and verify outputs before
benchmarking.

For Lesson 05, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Unstructured Magnitude Pruning Without Storage Myths** observation
belongs to Lesson 05's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Deep Compression](https://arxiv.org/abs/1510.00149)
