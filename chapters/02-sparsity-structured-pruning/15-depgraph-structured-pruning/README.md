# Lesson 15 — Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab

> **Puzzle:** Can a dependency graph identify every tensor coupled to one channel deletion on this environment?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

DepGraph turns a local root operation into a pruning group. That is precisely the
bookkeeping manual structural pruning tends to miss. A credible lab must distinguish the
graph concept, a manual CUDA control, and whether the optional Torch-Pruning package
executed successfully on the recorded stack.

For **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab**, the engineering
question is not whether a definition can be repeated; it is whether the following claim
survives a controlled GPU test: *Can a dependency graph identify every tensor coupled to
one channel deletion on this environment?* The lab therefore changes the mechanism
described below, retains its measured state, and names the evidence that would still be
needed for deployment.

## Predict before reading the result

1. Predict which modules join a group rooted at the first convolution.
2. Predict the result when the optional package is absent.
3. State what must be saved after module shapes are mutated.

Before opening Lesson 15's retained output, answer the first prompt— *Predict which
modules join a group rooted at the first convolution.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

A residual mini-network, one channel index set, a manually synchronized narrow copy, an
import/version probe, and—when available—a real `DependencyGraph` group are the concrete
objects.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | DepGraph groups coupled pruning operations from a root decision. |
| 2 | Example inputs and enabled autograd define the traced dependency path. |
| 3 | Package compatibility evidence is distinct from manual structural correctness. |

Lesson 15 tracks three layers through Torch-Pruning DepGraph: A Structured-Pruning
Compatibility Lab: *value state* says which entries are zero, *shape state* says which
axes physically changed, and *execution state* says which operator actually ran. The
anchors above identify where this lesson's claim lives, so a zero count cannot silently
turn into a latency claim.

## 2. Derive the mechanism

Torch-Pruning traces an example forward with autograd enabled, then maps a root pruning
function through module and tensor dependencies. Group validation prevents deleting an
entire dimension. The package mutates module structure, so saving a plain
dense-definition state_dict is insufficient unless architecture metadata is
reconstructed. The manual control proves expected shape propagation independently of
package availability.

The inspectable invariant for **Torch-Pruning DepGraph: A Structured-Pruning
Compatibility Lab** is tested by: Build a real DepGraph group when available and always
execute a manual CUDA structural-control path. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Build a real DepGraph group when available and always execute a manual CUDA structural-control path.

| Experimental role | Frozen definition |
|---|---|
| Baseline | manual synchronized pruning ledger for a residual mini-network |
| Candidate | Torch-Pruning dependency group and mutation when the package is available |
| Held constant | model, example input, root module, channel indices, eval mode, and GPU |
| Measurements | package availability/version, group validity/size, output shape, parameters, and caught exception |
| Evidence label | `compatibility-probe` |

This Lesson 15 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **model, example input, root module, channel indices, eval mode, and GPU**.
That frozen condition preserves the dependency or runtime boundary at issue; the small
scale limits transfer to larger models but does not permit the baseline and candidate to
answer different questions.

### Code walk-through

The notebook first runs the manual control so the lesson remains informative on a
minimal PyTorch installation. It then probes `torch_pruning`, builds the graph without
`no_grad`, requests a pruning group, validates it, and records group detail instead of
converting import failure into a successful backend claim.

For **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab**, the environment
cell asserts CUDA and fixes a lesson-specific seed. The experiment cell implements
Torch-Pruning dependency group and mutation when the package is available and records
package availability/version, group validity/size, output shape, parameters, and caught
exception. The artifact cell serializes those same fields. Only optional-backend import
or API failures become compatibility evidence; an error in the core comparison still
fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Torch-Pruning available | no |
| Group built | no |
| Group valid | no |
| Manual output channels | 6 |
| Manual parameters | 368 |
| Probe message | torch_pruning not installed |

### What the numbers mean

The manual dependency control reduced the model from 552 to 368 parameters and produced
6 output channels. Torch-Pruning availability was False; group built/valid were
False/False. This is a bounded compatibility result when the optional package is absent.

Lesson 15's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **compatibility-probe** evidence; the printed notebook
payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> A dependency graph is valuable when its real group, mutation, and save/load path are observed—not when its name appears in a plan.

### Acceptance and rollback gate

Accept the automated route only when the group is valid, forward and quality checks
pass, and the mutated architecture has a tested save/load contract.

The gate for **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab** is
stricter than “the code ran” because it binds this lesson's tensor or model identity,
quality tolerance, workload, runtime path, and rollback evidence. A missing optional
package can settle a compatibility question, but it cannot satisfy the
native-performance decision stated above.

### How this conclusion can fail

A successful trace can miss data-dependent control flow or static attributes used
outside tensor operations. A package import proves nothing about a specific model group.
Conversely, package absence does not falsify the DepGraph method; it only leaves that
native path unexecuted.

## 6. Follow the theory inside the notebook

In Lesson 15's [`lab.ipynb`](lab.ipynb), first identify **manual synchronized pruning
ledger for a residual mini-network** and **Torch-Pruning dependency group and mutation
when the package is available** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab**
is **predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/15-depgraph-structured-pruning/lab.ipynb
```

This lesson's optional/native backend path requires:

```bash
pip install torch-pruning
```

To reproduce **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab**, use a
PyTorch build compiled for the target GPU and select `Run All`. Compare the measurements
in the frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Install the pinned Torch-Pruning version, run the notebook again, compare printed group
operations with the manual ledger, and test whole-model serialization and reload.

For Lesson 15, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The notebook records real package/API availability and preserves the native success or
failure state. Missing backend execution remains unmeasured.

The checked-in **Torch-Pruning DepGraph: A Structured-Pruning Compatibility Lab**
observation belongs to Lesson 15's RTX 5090 environment, shapes, seed, and protocol. It
does not establish the unmeasured task quality or platform properties named in the
failure analysis. This independently written tutorial uses the study topic as a
question, without redistributing source HTML, model weights, private paths, or
infrastructure.

## References

- [Torch-Pruning reference implementation](https://github.com/VainF/Torch-Pruning)
- [DepGraph paper](https://arxiv.org/abs/2301.12900)
