# Lesson 14 — PyTorch Pruning API and the Complete Mask Lifecycle

> **Puzzle:** What changes in a module before and after `prune.remove`, and what must a rollback loader know?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

PyTorch pruning is a reparameterization lifecycle: apply a method, combine or update
masks, train while preserving the mask, serialize the expected keys, optionally
materialize with `remove`, and verify loading. Confusing a hook-based checkpoint with a
materialized one breaks reproducibility.

For **PyTorch Pruning API and the Complete Mask Lifecycle**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *What changes in a module before and after `prune.remove`, and what
must a rollback loader know?* The lab therefore changes the mechanism described below,
retains its measured state, and names the evidence that would still be needed for
deployment.

## Predict before reading the result

1. Predict parameter and buffer names after the first mask.
2. Predict sparsity after two iterative 25% pruning calls.
3. Predict whether `remove` restores the deleted weights.

Before opening Lesson 14's retained output, answer the first prompt— *Predict parameter
and buffer names after the first mask.*—and write one observation that would falsify the
answer. If the result is already visible, hide it and make the commitment first;
otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

One CUDA linear layer is inspected at five points: dense, first mask, iterated mask,
optimizer update, and removal. Named parameters, buffers, forward pre-hooks, sparsity,
and output drift are captured.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Parameter and buffer names encode the checkpoint lifecycle stage. |
| 2 | Iterative masks compose rather than reset by default. |
| 3 | Removal makes pruning permanent in a dense parameter. |

Lesson 14 tracks three layers through PyTorch Pruning API and the Complete Mask
Lifecycle: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

After pruning, `weight_orig` is a parameter and `weight_mask` a buffer; the visible
`weight` is computed before forward. Iterative pruning combines masks through a pruning
container. Gradients update `weight_orig`, so effective masked values remain zero at
forward even when underlying values change. `remove` replaces this pair with a
materialized `weight` parameter and deletes the hook; it does not undo pruning.

The inspectable invariant for **PyTorch Pruning API and the Complete Mask Lifecycle** is
tested by: Apply iterative PyTorch masks, take a training step, remove the
reparameterization, and audit every state transition. Its purpose is to prevent the
specific category error behind this puzzle. An algorithmic change, a stored
representation, and a runtime observation remain separate until the candidate and
measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Apply iterative PyTorch masks, take a training step, remove the reparameterization, and audit every state transition.

| Experimental role | Frozen definition |
|---|---|
| Baseline | dense module and single-mask state |
| Candidate | iteratively pruned, trained, and materialized module |
| Held constant | module, optimizer rule, input/target, pruning calls, seed, and inspection points |
| Measurements | effective sparsity, parameter names, buffer names, hook count, loss, and removal drift |
| Evidence label | `pytorch-gpu` |

This Lesson 14 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **module, optimizer rule, input/target, pruning calls, seed, and inspection
points**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The notebook queries public module inspection APIs rather than inferring state from
printed tensors. It records the effective weight before and after an optimizer step,
then compares forward output immediately around `remove`. This produces a
loader-oriented lifecycle trace.

For **PyTorch Pruning API and the Complete Mask Lifecycle**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements
iteratively pruned, trained, and materialized module and records effective sparsity,
parameter names, buffer names, hook count, loss, and removal drift. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| First-mask sparsity | 25.00% |
| Iterated sparsity | 43.75% |
| Parameters before remove | bias,weight_orig |
| Buffers before remove | weight_mask |
| Hooks before remove | 1 |
| Remove max drift | 0.000000 |

### What the numbers mean

The first call produced 25.0% sparsity and the second composed to 43.8%. Before removal,
parameters were `bias,weight_orig`, buffers were `weight_mask`, and 1 pre-hook was
active. `remove` left drift 0.000e+00 and restored a materialized `weight` parameter.

Lesson 14's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> PyTorch masks are auditable training state; `remove` materializes them but does not create a sparse runtime format.

### Acceptance and rollback gate

Accept a checkpoint only when its expected lifecycle stage, key schema, load procedure,
mask policy, and rollback artifact are documented and tested.

The gate for **PyTorch Pruning API and the Complete Mask Lifecycle** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

Loading a hook checkpoint into a plain module yields missing or unexpected keys.
Optimizer state can point at reparameterized objects. Removing too early can allow zeros
to regrow during later unconstrained training. These are state-management failures, not
pruning-score failures.

## 6. Follow the theory inside the notebook

In Lesson 14's [`lab.ipynb`](lab.ipynb), first identify **dense module and single-mask
state** and **iteratively pruned, trained, and materialized module** without running
them. Next inspect the dimensions or lifecycle state that implements the derivation.
After **Run All**, verify the RTX 5090 environment and the frozen fields before
reconciling the result table with the artifact.

The reader loop for **PyTorch Pruning API and the Complete Mask Lifecycle** is **predict
→ execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/14-pytorch-mask-lifecycle/lab.ipynb
```

To reproduce **PyTorch Pruning API and the Complete Mask Lifecycle**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Save and reload both lifecycle variants in fresh modules, test optimizer resume, and add
a schema version to the structured artifact.

For Lesson 14, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **PyTorch Pruning API and the Complete Mask Lifecycle** observation
belongs to Lesson 14's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
