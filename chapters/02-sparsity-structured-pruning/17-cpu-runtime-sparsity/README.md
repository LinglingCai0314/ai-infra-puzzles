# Lesson 17 — OpenVINO, NNCF, and Intel Runtime Sparsity

> **Puzzle:** Why can a generic sparse checkpoint miss the optimized CPU path?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

CPU deployment benefit depends on a pattern, graph transformation, precision, and
operator implementation supported by the target OpenVINO/oneDNN stack. NNCF or Intel
Neural Compressor configuration is part of the executable artifact; a PyTorch zero rate
alone is not.

For **OpenVINO, NNCF, and Intel Runtime Sparsity**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *Why can a generic sparse checkpoint miss the optimized CPU path?*
The lab therefore changes the mechanism described below, retains its measured state, and
names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict which package probes succeed in the recorded GPU environment.
2. Explain why physically narrower shapes remain useful without a sparse CPU kernel.
3. List the CPU-specific fields needed for a fair benchmark.

Before opening Lesson 17's retained output, answer the first prompt— *Predict which
package probes succeed in the recorded GPU environment.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

The notebook probes OpenVINO, NNCF, and Neural Compressor packages, creates unstructured
and filter-pruned controls on CUDA, and records a deployment gate matrix without
asserting CPU speed from GPU evidence.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Framework zeros are not an OpenVINO execution plan. |
| 2 | Filter removal and unstructured encoding expose different CPU opportunities. |
| 3 | GPU control results cannot substitute for CPU runtime measurements. |

Lesson 17 tracks three layers through OpenVINO, NNCF, and Intel Runtime Sparsity: *value
state* says which entries are zero, *shape state* says which axes physically changed,
and *execution state* says which operator actually ran. The anchors above identify where
this lesson's claim lives, so a zero count cannot silently turn into a latency claim.

## 2. Derive the mechanism

Unstructured zeros preserve dense tensor dimensions unless a sparse encoding and sparse
operator are selected. NNCF filter pruning can propagate structural changes and export a
smaller graph, while post-training sparsity tools may target runtime-specific patterns.
CPU SIMD utilization, threading, cache behavior, and quantization interact with width.
Therefore the correct handoff includes model format, pattern, runtime version, thread
settings, and operator log.

The inspectable invariant for **OpenVINO, NNCF, and Intel Runtime Sparsity** is tested
by: Probe Intel compression/runtime packages and contrast value sparsity with physical
width under a bounded evidence label. Its purpose is to prevent the specific category
error behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Probe Intel compression/runtime packages and contrast value sparsity with physical width under a bounded evidence label.

| Experimental role | Frozen definition |
|---|---|
| Baseline | same-shape unstructured zero mask represented as a dense PyTorch tensor |
| Candidate | physically narrower dense control and optional OpenVINO/NNCF native path |
| Held constant | source tensor, zero budget, input, environment, package names, and decision gates |
| Measurements | package availability, logical sparsity, physical width, output drift, and native-run status |
| Evidence label | `compatibility-probe` |

This Lesson 17 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **source tensor, zero budget, input, environment, package names, and decision
gates**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The experiment keeps its CUDA numerical control separate from the package matrix.
Conditional imports record exact availability; the conclusion remains `not_run` for
OpenVINO performance unless a native model conversion and CPU workload execute. This
prevents a generic pruning result from being laundered into an Intel deployment claim.

For **OpenVINO, NNCF, and Intel Runtime Sparsity**, the environment cell asserts CUDA
and fixes a lesson-specific seed. The experiment cell implements physically narrower
dense control and optional OpenVINO/NNCF native path and records package availability,
logical sparsity, physical width, output drift, and native-run status. The artifact cell
serializes those same fields. Only optional-backend import or API failures become
compatibility evidence; an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| OpenVINO available | no |
| NNCF available | no |
| Neural Compressor available | no |
| Logical sparsity | 75.00% |
| Physical width reduction | 75.00% |
| Native CPU run | no |

### What the numbers mean

The dense-value control reached 75.0% logical sparsity without changing its 1024-wide
output; the physical control changed width to 256. OpenVINO/NNCF/Neural Compressor
availability was False/False/False. No CPU latency is reported because no native CPU
path executed.

Lesson 17's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **compatibility-probe** evidence; the printed notebook
payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Intel sparsity is a runtime-specific graph and kernel decision; CUDA zeros provide only a numerical control.

### Acceptance and rollback gate

Accept an Intel deployment only after conversion, graph inspection, CPU thread pinning,
quality parity, and repeated target-CPU latency/throughput evidence.

The gate for **OpenVINO, NNCF, and Intel Runtime Sparsity** is stricter than “the code
ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

Package presence is weaker than operator support, and a laptop CPU result may not
transfer to the production SKU. A narrow channel count can hurt vector alignment, while
unstructured compression can reduce disk size without runtime benefit.

## 6. Follow the theory inside the notebook

In Lesson 17's [`lab.ipynb`](lab.ipynb), first identify **same-shape unstructured zero
mask represented as a dense PyTorch tensor** and **physically narrower dense control and
optional OpenVINO/NNCF native path** without running them. Next inspect the dimensions
or lifecycle state that implements the derivation. After **Run All**, verify the RTX
5090 environment and the frozen fields before reconciling the result table with the
artifact.

The reader loop for **OpenVINO, NNCF, and Intel Runtime Sparsity** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/17-cpu-runtime-sparsity/lab.ipynb
```

This lesson's optional/native backend path requires:

```bash
pip install openvino nncf neural-compressor
```

To reproduce **OpenVINO, NNCF, and Intel Runtime Sparsity**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Create a pinned OpenVINO/NNCF environment, export both candidates, inspect IR dimensions
and operators, then benchmark several thread and batch settings on the actual CPU
target.

For Lesson 17, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The notebook records real package/API availability and preserves the native success or
failure state. Missing backend execution remains unmeasured.

The checked-in **OpenVINO, NNCF, and Intel Runtime Sparsity** observation belongs to
Lesson 17's RTX 5090 environment, shapes, seed, and protocol. It does not establish the
unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [OpenVINO model optimization guide](https://docs.openvino.ai/2023.3/openvino_docs_model_optimization_guide.html)
- [NNCF reference implementation](https://github.com/openvinotoolkit/nncf)
