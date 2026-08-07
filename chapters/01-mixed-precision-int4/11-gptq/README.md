# Lesson 11 — GPTQ: Second-Order Intuition and Layer Reconstruction

> **Puzzle:** Why should two weights with the same magnitude receive different quantization treatment?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

GPTQ reconstructs one layer at a time using the layer weights and representative input activations. It targets output distortion, not unweighted distance between original and rounded weights.

Quick mental model:

- Layer reconstruction minimizes output error under representative inputs, not raw weight error alone.
- Input covariance approximates which directions are sensitive.
- Production GPTQ uses structured second-order updates; a toy sensitivity model is not the library implementation.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

For weight error `ΔW` and inputs `X`, layer error is approximately `||XΔWᵀ||²`; the input Gram/Hessian approximation `XᵀX` weights sensitive directions. GPTQ uses inverse-Hessian information to compensate remaining weights as columns are quantized.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Block size and damping control memory, numerical stability, and approximation quality. Ordering and calibration data alter the result, and the packed inference kernel is a separate concern.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Compare naive INT4 weight quantization with a GPTQ-inspired sensitivity fallback that preserves columns with large input-weighted error.

The lab is deliberately GPTQ-inspired: it uses input-weighted sensitivity and a fallback to expose the objective, while clearly not claiming GPTQModel execution.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`numerical-model`**.

## 5. Inspect, accept, or roll back

Measure layer-output error on held-out inputs and label the experiment as an intuition model, not a GPTQ kernel benchmark.

Record calibration activations, damping, block/group size, ordering, layer reconstruction loss, end-task regression, and the deployed operator.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Input-weighted sensitivity changed which quantization errors mattered; this is a GPTQ intuition model, not GPTQModel execution.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Second-order information changes the objective from nearest weights to faithful layer outputs.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/11-gptq/lab.ipynb
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

- [GPTQ paper](https://arxiv.org/abs/2210.17323)
