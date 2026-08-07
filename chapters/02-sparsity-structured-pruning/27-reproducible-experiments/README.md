# Lesson 27 — Automated Experiment Management and Reproducible Pruning Records

> **Puzzle:** Which fields make a pruning mask reproducible by another person?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A sparsity number cannot identify a run. Reproduction needs data and model revisions,
seed, score rule, tie behavior, target, mask bytes or hash, optimizer/recovery schedule,
software, hardware, export command, and measured artifacts. Tracking systems help only
when these fields are logged.

For **Automated Experiment Management and Reproducible Pruning Records**, the
engineering question is not whether a definition can be repeated; it is whether the
following claim survives a controlled GPU test: *Which fields make a pruning mask
reproducible by another person?* The lab therefore changes the mechanism described
below, retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict which hashes match across identical-seed runs.
2. Predict whether a different seed can preserve sparsity while changing the mask.
3. List the minimum fields another machine needs to repeat the result.

Before opening Lesson 27's retained output, answer the first prompt— *Predict which
hashes match across identical-seed runs.*—and write one observation that would falsify
the answer. If the result is already visible, hide it and make the commitment first;
otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A deterministic pruning function is executed twice with one seed and once with another.
Configuration JSON, weight initialization, mask, output, and SHA-256 digests are
compared in a small run registry.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Sparsity equality is weaker than mask identity. |
| 2 | Canonical configuration and binary artifacts need separate hashes. |
| 3 | A tracking UI cannot compensate for missing provenance fields. |

Lesson 27 tracks three layers through Automated Experiment Management and Reproducible
Pruning Records: *value state* says which entries are zero, *shape state* says which
axes physically changed, and *execution state* says which operator actually ran. The
anchors above identify where this lesson's claim lives, so a zero count cannot silently
turn into a latency claim.

## 2. Derive the mechanism

Random seed controls initialization and sampled data, but deterministic algorithms and
stable ordering also matter. Hashing canonical JSON catches configuration drift; hashing
contiguous mask bytes identifies the exact support. A code commit and environment
complete the provenance. Reproducing the same global sparsity with a different mask is
not the same experiment.

The inspectable invariant for **Automated Experiment Management and Reproducible Pruning
Records** is tested by: Run a pruning pipeline twice identically and once with a changed
seed, then compare config, mask, and output hashes. Its purpose is to prevent the
specific category error behind this puzzle. An algorithmic change, a stored
representation, and a runtime observation remain separate until the candidate and
measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Run a pruning pipeline twice identically and once with a changed seed, then compare config, mask, and output hashes.

| Experimental role | Frozen definition |
|---|---|
| Baseline | two executions with identical canonical configuration and seed |
| Candidate | one execution changing only the seed |
| Held constant | algorithm code, config schema, dimensions, target sparsity, dtype, hash method, and environment capture |
| Measurements | config hash, mask hash, output hash, sparsity, same-seed equality, and changed-seed difference |
| Evidence label | `pytorch-gpu` |

This Lesson 27 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **algorithm code, config schema, dimensions, target sparsity, dtype, hash
method, and environment capture**. That frozen condition preserves the dependency or
runtime boundary at issue; the small scale limits transfer to larger models but does not
permit the baseline and candidate to answer different questions.

### Code walk-through

The notebook serializes configuration with sorted keys and compact separators before
hashing. Masks move to CPU as contiguous bytes for a stable digest. The registry rows
include environment and conclusion fields suitable for MLflow or W&B, but no external
service is required to reproduce the core evidence.

For **Automated Experiment Management and Reproducible Pruning Records**, the
environment cell asserts CUDA and fixes a lesson-specific seed. The experiment cell
implements one execution changing only the seed and records config hash, mask hash,
output hash, sparsity, same-seed equality, and changed-seed difference. The artifact
cell serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Same-seed config match | yes |
| Same-seed mask match | yes |
| Same-seed output match | yes |
| Different-seed mask differs | yes |
| Sparsity | 75.00% |
| Mask SHA-256 | `89bdaa05855b` |

### What the numbers mean

Identical-seed runs matched config/mask/output hashes=True/True/True at 75.0% sparsity.
Changing only the seed changed the mask=True. The recorded support digest begins
`89bdaa05855b`.

Lesson 27's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Reproducible pruning identifies the exact configuration, support, environment, and outputs—not merely the final zero percentage.

### Acceptance and rollback gate

Accept a reproduction claim only when an independent rerun matches the declared
configuration, mask or bounded metrics, and environment-sensitive tolerances.

The gate for **Automated Experiment Management and Reproducible Pruning Records** is
stricter than “the code ran” because it binds this lesson's tensor or model identity,
quality tolerance, workload, runtime path, and rollback evidence. A missing optional
package can settle a compatibility question, but it cannot satisfy the
native-performance decision stated above.

### How this conclusion can fail

Seeds do not guarantee bitwise equality across all devices, library versions, or
nondeterministic kernels. Hashing only a filename or sparsity misses content changes.
Private paths and credentials must never enter a public artifact.

## 6. Follow the theory inside the notebook

In Lesson 27's [`lab.ipynb`](lab.ipynb), first identify **two executions with identical
canonical configuration and seed** and **one execution changing only the seed** without
running them. Next inspect the dimensions or lifecycle state that implements the
derivation. After **Run All**, verify the RTX 5090 environment and the frozen fields
before reconciling the result table with the artifact.

The reader loop for **Automated Experiment Management and Reproducible Pruning Records**
is **predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/27-reproducible-experiments/lab.ipynb
```

To reproduce **Automated Experiment Management and Reproducible Pruning Records**, use a
PyTorch build compiled for the target GPU and select `Run All`. Compare the measurements
in the frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Log the same schema to MLflow or W&B, rerun on a second machine, define which fields
must match exactly versus within tolerance, and add checkpoint/export hashes.

For Lesson 27, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Automated Experiment Management and Reproducible Pruning Records**
observation belongs to Lesson 27's RTX 5090 environment, shapes, seed, and protocol. It
does not establish the unmeasured task quality or platform properties named in the
failure analysis. This independently written tutorial uses the study topic as a
question, without redistributing source HTML, model weights, private paths, or
infrastructure.

## References

- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [MLflow tracking documentation](https://mlflow.org/docs/latest/ml/tracking/)
