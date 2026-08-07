# Lesson 19 — ONNX Export, Graph Repair, and Shape Consistency

> **Puzzle:** Can a pruned PyTorch model run correctly while its exported graph carries inconsistent channel metadata?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Physical pruning changes dimensions across weights, bias, normalization, reshape,
concat, and post-processing nodes. ONNX export success only serializes the traced path;
checker, shape inference, ONNX Runtime parity, and explicit dimension audits establish a
deployable graph.

For **ONNX Export, Graph Repair, and Shape Consistency**, the engineering question is
not whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Can a pruned PyTorch model run correctly while its exported graph
carries inconsistent channel metadata?* The lab therefore changes the mechanism
described below, retains its measured state, and names the evidence that would still be
needed for deployment.

## Predict before reading the result

1. Predict the output channel dimension after the physical slice.
2. Explain what ONNX shape inference can and cannot establish.
3. Choose a parity tolerance for the independent runtime output.

Before opening Lesson 19's retained output, answer the first prompt— *Predict the output
channel dimension after the physical slice.*—and write one observation that would
falsify the answer. If the result is already visible, hide it and make the commitment
first; otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

A small physically pruned multi-input model is exported in memory/on disk, checked with
ONNX, passed through shape inference, executed with ONNX Runtime when available, and
compared with CUDA/PyTorch output.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Export, checker, inference, and runtime parity are distinct gates. |
| 2 | Physical channel changes must reach initializers and consumer shapes. |
| 3 | Dynamic symbols do not excuse inconsistent known dimensions. |

Lesson 19 tracks three layers through ONNX Export, Graph Repair, and Shape Consistency:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

Static tensor shapes encode known dimensions while dynamic axes use symbolic parameters.
Shape inference propagates what operator schemas can prove but cannot resolve every
data-dependent reshape. `onnx.checker.check_model(..., full_check=True)` validates graph
structure and types; ONNX Runtime supplies an independent execution path. After channel
deletion, every initializer and consumer dimension must agree with the new graph
contract.

The inspectable invariant for **ONNX Export, Graph Repair, and Shape Consistency** is
tested by: Export a physically pruned CUDA model, run ONNX checker and shape inference,
and compare ONNX Runtime output. Its purpose is to prevent the specific category error
behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Export a physically pruned CUDA model, run ONNX checker and shape inference, and compare ONNX Runtime output.

| Experimental role | Frozen definition |
|---|---|
| Baseline | PyTorch output and declared pruned shape ledger |
| Candidate | checked/inferred ONNX graph plus ONNX Runtime execution |
| Held constant | weights, retained indices, inputs, opset, dynamic-axis policy, dtype, and tolerance |
| Measurements | export status, checker status, inferred shapes, initializer dimensions, runtime status, and max error |
| Evidence label | `native-backend` |

This Lesson 19 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **weights, retained indices, inputs, opset, dynamic-axis policy, dtype, and
tolerance**. That frozen condition preserves the dependency or runtime boundary at
issue; the small scale limits transfer to larger models but does not permit the baseline
and candidate to answer different questions.

### Code walk-through

The notebook writes the ONNX model into the lesson artifact directory, invokes full
checking, records inferred value shapes, and runs the same inputs through ONNX Runtime.
Exceptions are caught into structured fields, but success requires every gate and
numerical parity rather than export alone.

For **ONNX Export, Graph Repair, and Shape Consistency**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements checked/inferred
ONNX graph plus ONNX Runtime execution and records export status, checker status,
inferred shapes, initializer dimensions, runtime status, and max error. The artifact
cell serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| ONNX available | yes |
| Export succeeded | yes |
| Checker passed | yes |
| Shape inference passed | yes |
| ORT executed | yes |
| ORT max error | 0.000000 |
| ONNX bytes | 1,722 bytes |

### What the numbers mean

ONNX export/checker/shape-inference gates were True/True/True; ONNX Runtime
executed=True with max error 5.960e-08. The graph occupied 1,722 bytes and retained the
physical width 7 through both input projections and the output consumer.

Lesson 19's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **native-backend** evidence; the printed notebook payload
and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> A pruned ONNX graph is deliverable only after structural checks and independent runtime parity confirm the new shape contract.

### Acceptance and rollback gate

Accept the exported graph only when checker, shape inference audit, initializer
dimensions, and target-runtime parity all pass.

The gate for **ONNX Export, Graph Repair, and Shape Consistency** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Tracer warnings and constant folding can hide data-dependent behavior. Shape inference
may remain partial, and ONNX Runtime success does not guarantee TensorRT support.
Multi-profile production shapes must be tested separately.

## 6. Follow the theory inside the notebook

In Lesson 19's [`lab.ipynb`](lab.ipynb), first identify **PyTorch output and declared
pruned shape ledger** and **checked/inferred ONNX graph plus ONNX Runtime execution**
without running them. Next inspect the dimensions or lifecycle state that implements the
derivation. After **Run All**, verify the RTX 5090 environment and the frozen fields
before reconciling the result table with the artifact.

The reader loop for **ONNX Export, Graph Repair, and Shape Consistency** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/19-onnx-shape-consistency/lab.ipynb
```

This lesson's optional/native backend path requires:

```bash
pip install onnx onnxruntime
```

To reproduce **ONNX Export, Graph Repair, and Shape Consistency**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Add dynamic batch and sequence axes, deliberately corrupt one initializer to verify the
audit fails, then test the repaired graph in the final deployment runtime.

For Lesson 19, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

A named non-PyTorch backend executed and its checker/runtime output is retained. This
still does not transfer to another backend or workload.

The checked-in **ONNX Export, Graph Repair, and Shape Consistency** observation belongs
to Lesson 19's RTX 5090 environment, shapes, seed, and protocol. It does not establish
the unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [ONNX checker API](https://onnx.ai/onnx/api/checker.html)
- [ONNX shape inference](https://onnx.ai/onnx/repo-docs/ShapeInference.html)
