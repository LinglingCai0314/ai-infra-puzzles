# Lesson 02 — Tensor Core Constraints for Low-Precision GEMM

> **Puzzle:** A low-precision dtype is available, so will every matrix multiplication automatically become a fast Tensor Core operation?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A GEMM consumes `A[M,K]` and `B[K,N]`. Dtype, strides, transposition, leading dimensions, and the three logical sizes travel together into dispatch; the word *BF16* by itself is not a kernel description.

Quick mental model:

- A dtype is only one dispatch condition; layout, dimensions, alignment, and backend policy also select the kernel.
- Arithmetic intensity separates compute-bound GEMMs from shapes dominated by memory traffic or launch overhead.
- Timing establishes performance for a shape; operator or kernel evidence establishes what ran.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

A useful first model is `FLOPs ≈ 2MKN` and `arithmetic intensity = FLOPs / bytes moved`. Large aligned tiles can amortize loads and feed matrix-multiply hardware; awkward dimensions create edge tiles, padding, or a different implementation. Tensor Core eligibility is therefore a conjunction of hardware, dtype, shape, layout, and library support.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Padding may improve tile utilization but adds work and memory. Small GEMMs may be launch- or memory-dominated, so a lower-precision peak-FLOP number may never become the bottleneck that the application sees.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Time FP32 and BF16 matrix multiplications with aligned and deliberately awkward dimensions on the same GPU.

The lab changes dtype and one alignment condition while keeping the GPU and timing method fixed; the output is shape evidence, not a native-kernel assertion.

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

Compare medians by dtype and shape. The lab does not infer Tensor Core use from speed alone; it records a PyTorch GPU timing baseline for later profiler work.

Keep the exact `M,N,K`, strides, dtype, warm-up, and repeated timing. Use an operator trace to show dispatch and Nsight Compute/System metrics before naming a native Tensor Core kernel.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Observed shape- and dtype-dependent GEMM timing; native Tensor Core identity requires a lower-level profiler.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Low precision creates an opportunity, not a guarantee. Preserve exact shapes and profiler evidence when deciding whether a Tensor Core path was reached.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/02-tensor-core-constraints/lab.ipynb
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

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
