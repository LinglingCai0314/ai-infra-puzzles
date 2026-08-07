# Lesson 06 — Global Sparsity and Layer-wise Budget Allocation

> **Puzzle:** Should a fixed 50% global budget prune every layer by 50%?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Layers transform different signals and have different redundancy. A global threshold
spends zeros where weights are small, while a uniform per-layer target ignores
sensitivity. A budget table should preserve the total constraint and show why protected
or aggressive allocations were assigned.

For **Global Sparsity and Layer-wise Budget Allocation**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Should a fixed 50% global budget prune every layer by 50%?* The
lab therefore changes the mechanism described below, retains its measured state, and
names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict which layer will be protected by the calibration sweep.
2. Predict whether uniform and global magnitude masks use the same per-layer rates.
3. Name the quality metric and total-budget invariant required for fairness.

Before opening Lesson 06's retained output, answer the first prompt— *Predict which
layer will be protected by the calibration sweep.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A three-layer MLP, calibration inputs, one global magnitude mask, a uniform 50% mask,
and a sensitivity-aware allocation are compared at equal total nonzero count using
held-out output reconstruction.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Global sparsity is a constraint across tensors, not a uniform policy. |
| 2 | Layer sensitivity must be measured on representative inputs. |
| 3 | Budget comparisons require equal total nonzeros. |

Lesson 06 tracks three layers through Global Sparsity and Layer-wise Budget Allocation:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

For network output `f(x; W)`, a layer's pruning cost depends on downstream amplification
and the input distribution, not only its weight histogram. A first-order sensitivity
sweep can mask a small fraction in one layer at a time and measure output change.
Budgets can then be allocated inversely to observed sensitivity while solving the global
nonzero constraint. The experiment keeps total zeros equal so quality differences come
from allocation rather than extra capacity.

The inspectable invariant for **Global Sparsity and Layer-wise Budget Allocation** is
tested by: Compare uniform, global-magnitude, and sensitivity-aware masks at the same
50% global zero budget. Its purpose is to prevent the specific category error behind
this puzzle. An algorithmic change, a stored representation, and a runtime observation
remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare uniform, global-magnitude, and sensitivity-aware masks at the same 50% global zero budget.

| Experimental role | Frozen definition |
|---|---|
| Baseline | uniform 50% magnitude pruning in every layer |
| Candidate | global thresholding and sensitivity-aware per-layer allocation |
| Held constant | dense weights, calibration/held-out tensors, global zero count, dtype, and seed |
| Measurements | per-layer sparsity, total sparsity, held-out RMSE, cosine similarity, and calibration sensitivity |
| Evidence label | `numerical-model` |

This Lesson 06 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **dense weights, calibration/held-out tensors, global zero count, dtype, and
seed**. That frozen condition preserves the dependency or runtime boundary at issue; the
small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The notebook first perturbs each layer separately to obtain a small calibration
sensitivity score. It then constructs three cloned models and checks exact total
sparsity before measuring held-out output error. The allocation heuristic is
intentionally simple; the evidence target is the budget principle, not a claim of
optimal pruning.

For **Global Sparsity and Layer-wise Budget Allocation**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements global
thresholding and sensitivity-aware per-layer allocation and records per-layer sparsity,
total sparsity, held-out RMSE, cosine similarity, and calibration sensitivity. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Uniform RMSE | 0.067273 |
| Global RMSE | 0.062327 |
| Aware RMSE | 0.061447 |
| Total sparsity | 50.00% |
| Most sensitive layer | 4.weight |

### What the numbers mean

All candidates used approximately 50.0% global sparsity. Held-out RMSE was 0.067273 for
uniform, 0.062327 for global magnitude, and 0.061447 for the sensitivity-adjusted
allocation. The calibration sweep identified `4.weight` as most sensitive. This
validates the budget experiment, not optimality of the heuristic.

Lesson 06's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> A global target needs a measured allocation rule; uniform layer rates are merely one candidate.

### Acceptance and rollback gate

Accept a layer budget only when the total constraint is exact and the ranking remains
stable on held-out data or multiple calibration slices.

The gate for **Global Sparsity and Layer-wise Budget Allocation** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Using the held-out set to allocate budgets leaks evaluation. Very small calibration
batches make sensitivity noisy, and equal zero counts do not ensure equal metadata or
runtime cost across layers. Hardware-aware costs may need a different budget unit than
parameters.

## 6. Follow the theory inside the notebook

In Lesson 06's [`lab.ipynb`](lab.ipynb), first identify **uniform 50% magnitude pruning
in every layer** and **global thresholding and sensitivity-aware per-layer allocation**
without running them. Next inspect the dimensions or lifecycle state that implements the
derivation. After **Run All**, verify the RTX 5090 environment and the frozen fields
before reconciling the result table with the artifact.

The reader loop for **Global Sparsity and Layer-wise Budget Allocation** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/06-global-layerwise-budgets/lab.ipynb
```

To reproduce **Global Sparsity and Layer-wise Budget Allocation**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Repeat the sweep across domains, optimize budgets in latency-weighted channel units, and
test whether protected early layers remain protected after recovery training.

For Lesson 06, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **Global Sparsity and Layer-wise Budget Allocation** observation belongs
to Lesson 06's RTX 5090 environment, shapes, seed, and protocol. It does not establish
the unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)
- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
