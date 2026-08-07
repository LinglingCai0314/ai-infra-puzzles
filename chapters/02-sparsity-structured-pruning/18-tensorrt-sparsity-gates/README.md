# Lesson 18 — TensorRT Sparse Deployment and Polygraphy Evidence

> **Puzzle:** What must the build log show before `--sparsity=enable` becomes an acceleration claim?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

TensorRT evaluates structured-sparsity eligibility and tactic profitability. A
2:4-compliant ONNX weight plus a sparsity flag makes a layer eligible; the builder can
still select a dense tactic. Polygraphy can help inspect and transform models, but only
engine logs and matched benchmarks establish execution.

For **TensorRT Sparse Deployment and Polygraphy Evidence**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *What must the build log show before `--sparsity=enable` becomes an
acceleration claim?* The lab therefore changes the mechanism described below, retains
its measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict whether a compliant weight alone proves a sparse tactic ran.
2. List the log lines and numerical checks required for acceptance.
3. Explain why forced pruning must be treated as a new model candidate.

Before opening Lesson 18's retained output, answer the first prompt— *Predict whether a
compliant weight alone proves a sparse tactic ran.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

The lab creates a compliant convolution/linear-style weight, checks pattern and dtype
gates, probes TensorRT and Polygraphy packages plus `trtexec`, and produces an
eligibility-versus-selection matrix.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Eligibility and tactic selection are separate log events. |
| 2 | Forcing a pattern is a model mutation that needs quality validation. |
| 3 | The engine version, flags, profiles, and timing cache identify the build. |

Lesson 18 tracks three layers through TensorRT Sparse Deployment and Polygraphy
Evidence: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

TensorRT's structured sparsity requires the documented local weight pattern and
supported FP16 or INT8 execution. The builder reports eligible layers separately from
layers for which sparse tactics are chosen. `--sparsity=force` style mutation changes
weights and therefore quality; enable mode should consume already compliant weights.
Strong typing, shapes, workspace, and version influence tactic search.

The inspectable invariant for **TensorRT Sparse Deployment and Polygraphy Evidence** is
tested by: Build the full pre-engine eligibility ledger and probe native
TensorRT/Polygraphy tools on the RTX 5090 host. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Build the full pre-engine eligibility ledger and probe native TensorRT/Polygraphy tools on the RTX 5090 host.

| Experimental role | Frozen definition |
|---|---|
| Baseline | 2:4-compliant BF16/FP16 weight and dense PyTorch numerical control |
| Candidate | native TensorRT build/tactic path when packages and `trtexec` are available |
| Held constant | weight, grouping axis, dtype gate, environment, package probes, and required build fields |
| Measurements | 2:4 compliance, dtype eligibility, TensorRT/Polygraphy/trtexec availability, and native engine status |
| Evidence label | `compatibility-probe` |

This Lesson 18 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **weight, grouping axis, dtype gate, environment, package probes, and
required build fields**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The notebook can prove the data-side invariant on CUDA and can prove whether native
tools exist. It cannot infer a TensorRT engine from PyTorch timing. The gate dictionary
leaves engine build and sparse-tactic selection false unless those events actually
occur.

For **TensorRT Sparse Deployment and Polygraphy Evidence**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements native TensorRT
build/tactic path when packages and `trtexec` are available and records 2:4 compliance,
dtype eligibility, TensorRT/Polygraphy/trtexec availability, and native engine status.
The artifact cell serializes those same fields. Only optional-backend import or API
failures become compatibility evidence; an error in the core comparison still fails the
notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| 2:4 compliance | 100.00% |
| Dtype eligible | yes |
| TensorRT available | no |
| Polygraphy available | no |
| trtexec available | no |
| Sparse engine built | no |

### What the numbers mean

The weight passed 100.0% of exact 2:4 groups at 50.0% sparsity and used an eligible
dtype=True. TensorRT/Polygraphy/trtexec availability was False/False/False. Because no
engine was built, sparse tactic selection remains false.

Lesson 18's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **compatibility-probe** evidence; the printed notebook
payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> TensorRT sparsity is proven by eligibility, selected tactic, correctness, and benchmark evidence—not by a flag.

### Acceptance and rollback gate

Accept TensorRT sparsity only with a valid engine, explicit eligible and selected
sparse-tactic logs, output parity, and matched dense/sparse engine benchmarks.

The gate for **TensorRT Sparse Deployment and Polygraphy Evidence** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

A builder flag can be ignored by ineligible layers or lose tactic search to a faster
dense kernel. Dynamic profiles may select different tactics, and a successful build on
A100 does not establish behavior on RTX 5090.

## 6. Follow the theory inside the notebook

In Lesson 18's [`lab.ipynb`](lab.ipynb), first identify **2:4-compliant BF16/FP16 weight
and dense PyTorch numerical control** and **native TensorRT build/tactic path when
packages and `trtexec` are available** without running them. Next inspect the dimensions
or lifecycle state that implements the derivation. After **Run All**, verify the RTX
5090 environment and the frozen fields before reconciling the result table with the
artifact.

The reader loop for **TensorRT Sparse Deployment and Polygraphy Evidence** is **predict
→ execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/18-tensorrt-sparsity-gates/lab.ipynb
```

This lesson's optional/native backend path requires:

```bash
pip install tensorrt polygraphy
```

To reproduce **TensorRT Sparse Deployment and Polygraphy Evidence**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Use a TensorRT-enabled container, export the compliant model, retain Polygraphy
inspection plus verbose build logs, and benchmark every production optimization profile.

For Lesson 18, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The notebook records real package/API availability and preserves the native success or
failure state. Missing backend execution remains unmeasured.

The checked-in **TensorRT Sparse Deployment and Polygraphy Evidence** observation
belongs to Lesson 18's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [TensorRT sparsity requirements](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html)
- [NVIDIA cuSPARSELt documentation](https://docs.nvidia.com/cuda/cusparselt/)
