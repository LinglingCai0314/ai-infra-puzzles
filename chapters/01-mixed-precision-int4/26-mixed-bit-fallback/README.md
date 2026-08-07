# Lesson 26 — Mixed-Bit Strategies and Sensitive-Layer Fallback

> **Puzzle:** If only a few layers cause most quantization error, should every layer use more bits?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Mixed-bit design assigns a precision/configuration to each layer or group under a memory, latency, and quality budget.

Quick mental model:

- Layer sensitivity is measured by the downstream objective under representative inputs.
- Mixed-bit allocation trades metadata and kernel diversity against quality.
- Fallback layers need a deterministic rule and a fixed memory budget.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

A sensitivity scan replaces one layer at a time and measures downstream change. A simple allocation then spends extra bits on the largest marginal quality benefit per added byte; interactions require re-evaluating the assembled model.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

More bit variants improve the Pareto frontier but fragment kernels, packing, and deployment. Layerwise rankings can change when several layers are quantized together.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Quantize a six-layer CUDA MLP one layer at a time, rank sensitivity, then construct a budgeted INT4/INT8 mixed-bit candidate.

The six-layer CUDA lab ranks INT4 substitutions, gives two layers INT8, computes average bits, and re-runs end to end.

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

Compare the final end-to-end error and estimated storage, not only isolated layer rankings.

Freeze calibration/evaluation, record isolated sensitivities, budget, chosen fallback layers, final assembled quality, storage, operator coverage, and latency.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** A budgeted mixed-bit candidate spent extra precision on measured sensitive layers and was re-evaluated end to end.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Use sensitivity scans to spend precision where it protects the objective, then re-measure the assembled model.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/26-mixed-bit-fallback/lab.ipynb
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

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
