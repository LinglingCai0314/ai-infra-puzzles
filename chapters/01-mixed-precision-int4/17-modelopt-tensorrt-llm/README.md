# Lesson 17 — ModelOpt to TensorRT-LLM Quantization Pipelines

> **Puzzle:** Which evidence is lost when a quantized checkpoint is handed from one tool to another?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A ModelOpt-to-TensorRT-LLM handoff includes base revision, calibration corpus, recipe, per-layer exclusions, quantized tensor metadata, tokenizer, builder/runtime versions, engine flags, and rollback target.

Quick mental model:

- A pipeline needs immutable model revision, calibration recipe, quantization metadata, build flags, and engine identity.
- FP8, INT4, and FP4 are different recipes, not interchangeable compression levels.
- Package availability is only the first compatibility gate.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Model optimization chooses and serializes a numerical representation; the engine builder lowers it to hardware tactics. Losing group axes, scale dtype, or recipe version at the boundary can change semantics even when files load.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Pre-quantized checkpoints shorten deployment but constrain engine/version choices. Re-quantizing locally offers control but requires calibration reproducibility and more build time.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Generate and validate a quantization handoff manifest seeded by a CUDA numerical probe, while checking ModelOpt and TensorRT-LLM availability independently.

The notebook creates a complete handoff manifest and a CUDA numerical fingerprint while explicitly marking ModelOpt and TensorRT-LLM availability.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`compatibility-probe`**.

## 5. Inspect, accept, or roll back

A valid manifest is a reproducibility result, not an engine throughput result.

Validate a schema and hashes at each handoff, run a deterministic smoke sample, inspect engine layers, and keep quality and performance gates separate.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `compatibility-probe`
- **Recorded outcome:** The handoff contract was validated; absent packages remain explicit and no engine benchmark was claimed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Treat every tool boundary as a versioned artifact handoff with explicit validation and rollback metadata.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/17-modelopt-tensorrt-llm/lab.ipynb
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

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
