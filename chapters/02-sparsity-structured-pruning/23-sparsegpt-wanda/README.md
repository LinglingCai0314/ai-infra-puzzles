# Lesson 23 — One-shot LLM Pruning: SparseGPT and Wanda Mechanisms

> **Puzzle:** Why should calibration activations change which LLM weights survive?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

One-shot LLM pruning must choose a support without full retraining. Magnitude ignores
input usage; Wanda combines weight magnitude with activation norms; SparseGPT uses
second-order reconstruction information and sequential compensation. A small layer can
expose these objectives without pretending to reproduce a 70B run.

For **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *Why should calibration activations change which LLM weights
survive?* The lab therefore changes the mechanism described below, retains its measured
state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict how rescaling one input feature changes Wanda but not magnitude ranking.
2. Explain what the diagonal-curvature proxy omits from SparseGPT.
3. Choose calibration and held-out splits for a fair one-shot comparison.

Before opening Lesson 23's retained output, answer the first prompt— *Predict how
rescaling one input feature changes Wanda but not magnitude ranking.*—and write one
observation that would falsify the answer. If the result is already visible, hide it and
make the commitment first; otherwise this becomes post-hoc explanation rather than a
pruning experiment.

## 1. Start from concrete tensors and state

A wide linear projection, calibration tokens with deliberately uneven feature scales,
held-out tokens, magnitude scores, Wanda scores, a diagonal-curvature proxy, and
equal-sparsity reconstructed outputs form the lab.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Calibration activations define feature importance for one-shot pruning. |
| 2 | Wanda scoring and SparseGPT compensation are not the same algorithm. |
| 3 | Toy layer reconstruction cannot establish 70B perplexity or speed. |

Lesson 23 tracks three layers through One-shot LLM Pruning: SparseGPT and Wanda
Mechanisms: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

For `Y=XW^T`, Wanda scores weight `w_ij` by `|w_ij| ||X_:j||`, so a modest weight on a
frequently excited feature may outrank a larger unused weight. SparseGPT instead
minimizes layer reconstruction with an approximate Hessian and updates remaining weights
as columns are pruned. A diagonal `X^T X` proxy can illustrate sensitivity but omits the
inverse-Hessian sequential algorithm. Equal sparsity and held-out output error are
required for comparison.

The inspectable invariant for **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms**
is tested by: Compare magnitude, Wanda, and diagonal-curvature masks at identical 50%
sparsity on held-out layer outputs. Its purpose is to prevent the specific category
error behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare magnitude, Wanda, and diagonal-curvature masks at identical 50% sparsity on held-out layer outputs.

| Experimental role | Frozen definition |
|---|---|
| Baseline | plain magnitude one-shot pruning |
| Candidate | Wanda activation-aware scoring and a diagonal-curvature sensitivity proxy |
| Held constant | weights, calibration tokens, held-out tokens, sparsity, grouping policy, and seed |
| Measurements | held-out RMSE, cosine similarity, support overlap, sparsity, and calibration feature scales |
| Evidence label | `numerical-model` |

This Lesson 23 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **weights, calibration tokens, held-out tokens, sparsity, grouping policy,
and seed**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The notebook constructs calibration features with unequal energy so activation-aware
methods have a measurable signal. Each scoring rule retains the same number of weights
per row. Output metrics are computed on separate held-out tokens. The artifact
explicitly labels the curvature route a proxy rather than SparseGPT.

For **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements Wanda
activation-aware scoring and a diagonal-curvature sensitivity proxy and records held-out
RMSE, cosine similarity, support overlap, sparsity, and calibration feature scales. The
artifact cell serializes those same fields. Only optional-backend import or API failures
become compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Magnitude RMSE | 20.097502 |
| Wanda RMSE | 4.029922 |
| Curvature-proxy RMSE | 6.004300 |
| Magnitude cosine | 0.962958 |
| Wanda cosine | 0.998537 |
| Support overlap | 66.47% |

### What the numbers mean

At 50.0% sparsity, magnitude/Wanda/curvature-proxy held-out RMSE values were
20.097502/4.029922/6.004300. Magnitude and Wanda retained-support overlap was 66.5%
under a 100x calibration feature-scale range. The curvature score is a diagonal
OBS-style proxy, not SparseGPT's sequential algorithm.

Lesson 23's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **numerical-model** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Activation-aware support selection can reduce one-shot reconstruction error, but official algorithms and full-model evidence remain separate gates.

### Acceptance and rollback gate

Accept an LLM pruning method only after frozen calibration, full-model
perplexity/zero-shot gates, serialization, and a supported sparse inference path are
measured.

The gate for **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

Calibration domains can bias activation norms, and per-row toy masks omit blockwise
sequential compensation. Lower layer RMSE may not preserve generation, rare
capabilities, or safety. Unstructured zeros may still run dense.

## 6. Follow the theory inside the notebook

In Lesson 23's [`lab.ipynb`](lab.ipynb), first identify **plain magnitude one-shot
pruning** and **Wanda activation-aware scoring and a diagonal-curvature sensitivity
proxy** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/23-sparsegpt-wanda/lab.ipynb
```

To reproduce **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Run official SparseGPT and Wanda implementations on a pinned open model, sweep
calibration domains and sparsity patterns, then benchmark a named sparse runtime
separately from quality.

For Lesson 23, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction,
trained production model, or native sparse-kernel benchmark.

The checked-in **One-shot LLM Pruning: SparseGPT and Wanda Mechanisms** observation
belongs to Lesson 23's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [SparseGPT](https://arxiv.org/abs/2301.00774)
- [Wanda](https://arxiv.org/abs/2306.11695)
