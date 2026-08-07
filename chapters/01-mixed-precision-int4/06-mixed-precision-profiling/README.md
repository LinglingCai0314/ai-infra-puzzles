# Lesson 06 — Profiling Mixed Precision and Verifying Dispatch

> **Puzzle:** If autocast made an operation faster, does that prove the intended low-precision kernel ran?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Three evidence layers answer different questions: model outputs show semantic effect, framework operators show graph dispatch, and native kernel traces show the implementation actually launched.

Quick mental model:

- A wall-clock delta and an operator trace answer different questions.
- Warm-up removes initialization and compilation from the steady-state sample.
- PyTorch operator names are higher-level evidence than native kernel names; use Nsight when kernel identity matters.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Profiling can expose casts, copies, GEMMs, launch count, and device time. Warm-up is required because lazy initialization, compilation, and allocator growth are not steady-state execution.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

A detailed profiler perturbs runtime and creates large traces; a light CUDA-event benchmark has lower overhead but less attribution. Use the least intrusive tool that can answer the current claim.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Profile an autocast BF16 GEMM with PyTorch Profiler and record the relevant operator events beside CUDA-event timing.

The notebook pairs repeated CUDA-event timing with selected PyTorch profiler events and explicitly stops short of inventing a native kernel name.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`pytorch-gpu`**.

## 5. Inspect, accept, or roll back

Require both repeated timing and trace evidence. This lab deliberately labels PyTorch operators rather than claiming a native kernel name.

First reproduce timing without a profiler, then capture a short aligned trace. Name only the level actually observed: PyTorch operator, CUDA kernel, or end-to-end phase.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Autocast timing and PyTorch operator evidence were captured; native kernel identity was not claimed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Use a two-part proof: controlled timing for effect and profiler evidence for dispatch; escalate to Nsight for native-kernel claims.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/06-mixed-precision-profiling/lab.ipynb
```

Use **Run All**. Optional production backends are intentionally not hidden in
the base requirements; install the version appropriate for your GPU and follow
its official compatibility matrix before attempting a native path.

## Evidence boundary

- The checked-in notebook was executed on the GPU recorded inside the artifact;
  results on another GPU or software release may differ.
- Synthetic tensors isolate the mechanism and keep the lab downloadable. They
  do not establish full-model task quality or service throughput.
- Missing optional packages are recorded as `not_installed`, `failed`, or
  `not_measured`; no substitute backend is presented as native evidence.
- This is independently written tutorial material. It does not redistribute the
  source-course HTML, model weights, or private profiler traces.

## References

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
