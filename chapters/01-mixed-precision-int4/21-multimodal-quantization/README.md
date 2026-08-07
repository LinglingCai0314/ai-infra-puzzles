# Lesson 21 — Quantizing Vision and Multimodal Models

> **Puzzle:** Why can a text-only calibration set miss important failure modes in a vision-language model?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A vision-language system contains a vision encoder, patch/token embedding, projector, cross- or self-attention, language model, and KV cache. Each component sees a different activation distribution.

Quick mental model:

- Vision encoders see patch distributions, image contrast, and positional structure unlike text MLP activations.
- A multimodal pipeline contains encoder, projector, language model, and attention/cache objects.
- Coverage and fallback decisions can differ by component.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Patch projection maps local pixel statistics into tokens; contrast and modality shifts can create channel ranges absent from text calibration. Quantization error can then propagate through normalization and attention.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Quantizing the large language component may save most bytes, while quantizing a sensitive bridge can cause disproportionate quality loss. Component-specific fallback may be cheaper than one global dtype.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Quantize a CUDA patch-projection weight and compare reconstruction error for ordinary and high-contrast synthetic images.

The notebook isolates a patch projection and compares normal versus high-contrast image distributions, carefully avoiding a full-VLM claim.

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

Compare domain-specific errors and keep the experiment scoped to the patch projection, not an entire VLM quality claim.

Stratify calibration/evaluation by modality, resolution, prompt length, OCR/chart cases, and component; measure component error plus end-task multimodal quality.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** Patch-projection error changed with image distribution; no full VLM quality conclusion was made.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Calibrate and regress each modality and bridge component rather than applying a text-only decision globally.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/21-multimodal-quantization/lab.ipynb
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
