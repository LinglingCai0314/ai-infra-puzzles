# Lesson 03 — PyTorch AMP: autocast and GradScaler

> **Puzzle:** Can mixed-precision training be reduced to wrapping the forward pass in autocast?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

The AMP loop contains FP32 parameters and optimizer state, autocast-selected forward activations, gradients, a scalar loss scale, and an optimizer update. These objects do not all share one dtype or lifetime.

Quick mental model:

- Autocast selects lower precision per eligible operation; it does not permanently convert every tensor.
- GradScaler changes loss magnitude before backward, unscales gradients before the optimizer step, and adapts its scale.
- The optimizer state and usually the master parameters remain higher precision.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

If `g` is the true gradient and `S` is the loss scale, backward first produces `S·g`; unscale restores `g` before clipping or the optimizer step. `GradScaler` skips the step when non-finite gradients are found and adapts `S`. Autocast independently chooses eligible forward-operation dtypes.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

BF16 often does not need scaling because of its exponent range, while FP16 can benefit from it. Scaling adds control logic and cannot repair a forward activation that already overflowed.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Train a small CUDA MLP with BF16 autocast and GradScaler while recording loss, parameter dtype, output dtype, gradient finiteness, and scale history.

The notebook prints parameter and output dtypes, runs the complete update loop, and records gradient finiteness rather than stopping after one autocast forward.

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

A valid loop needs finite gradients and an optimizer update. An autocast dtype printout alone is not a training result.

Verify the order `zero_grad -> autocast forward -> scale(loss).backward -> unscale/step -> update`, record finite gradients and scale history, and keep the loss objective identical to the FP32 baseline.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** The full autocast-scale-backward-step-update loop completed with finite gradients.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

AMP is a control loop across forward, backward, unscale, step, and update—not a global dtype switch.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/03-pytorch-amp/lab.ipynb
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
