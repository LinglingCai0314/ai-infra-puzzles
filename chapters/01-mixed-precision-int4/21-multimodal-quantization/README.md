# Lesson 21 — Quantizing Vision and Multimodal Models

> **Puzzle:** Why can a text-only calibration set miss important failure modes in a vision-language model?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A multimodal model does not have one activation distribution. Patch projection sees
pixels and local contrast, the vision encoder sees image tokens, the connector maps
modalities, and the language decoder sees text-conditioned states. Calibrating only text
can leave the vision path with unobserved ranges and brittle low-bit behavior.

## Predict before reading the result

1. Predict how high-contrast image patches change the output error of one quantized patch projection.
2. Identify the calibration strata needed for a vision-language model rather than a text-only LLM.
3. Explain why a patch-projection result cannot establish full VLM quality.

## 1. Start from concrete tensors and state

A vision-language system contains a vision encoder, patch/token embedding, projector,
cross- or self-attention, language model, and KV cache. Each component sees a different
activation distribution.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Vision encoders see patch distributions, image contrast, and positional structure unlike text MLP activations. |
| 2 | A multimodal pipeline contains encoder, projector, language model, and attention/cache objects. |
| 3 | Coverage and fallback decisions can differ by component. |

## 2. Derive the mechanism

Patch projection maps local pixel statistics into tokens; contrast and modality shifts
can create channel ranges absent from text calibration. Quantization error can then
propagate through normalization and attention.

A ViT patch projection is a convolution with kernel and stride equal to patch size. For
16×16 RGB patches, each output token combines 768 input values. Weight quantization
error is filtered by the image distribution: high-contrast or sparse extreme pixels can
amplify particular columns that ordinary Gaussian-like calibration does not emphasize.

Farther downstream, cross-attention and modality connectors introduce their own outliers
and quality objectives. A sound plan therefore calibrates and evaluates per component
and per modality slice, then rejoins them with end-to-end captioning, VQA, OCR, or
grounding tasks.

## 3. Translate the theory into an experiment

**Experiment:** Quantize a CUDA patch-projection weight and compare reconstruction error for ordinary and high-contrast synthetic images.

| Experimental role | Frozen definition |
|---|---|
| Baseline | floating-point 64-channel, 16×16 patch projection |
| Candidate | group-192 INT4-dequantized projection weights |
| Held constant | same weights, image shape, projection stride, normal/high-contrast paired inputs |
| Measurements | projection-output RMSE/MAE/cosine/max error for each image distribution |
| Evidence label | `pytorch-gpu` |

The notebook isolates a patch projection and compares normal versus high-contrast image
distributions, carefully avoiding a full-VLM claim.

### Code walk-through

The notebook isolates the first vision operation so tensor axes remain readable: weights
have shape `[64,3,16,16]`, then flatten to groups for quantization and return to
convolution layout. It evaluates the same candidate on ordinary random images and images
with periodic contrast spikes.

This component test answers whether input distribution changes local error. It
deliberately excludes transformer blocks, the language decoder, preprocessing, and task
metrics, so its conclusion stops before full-model quality.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Patch weight shape | 64 × 3 × 16 × 16 |
| Normal-image RMSE | 0.040851 |
| High-contrast RMSE | 0.063011 |
| Normal max error | 0.202358 |
| High-contrast max error | 0.358466 |

### What the numbers mean

For normal images, projection RMSE was 0.040851 with cosine 0.997531. High-contrast
patches increased RMSE to 0.063011 and max absolute error from 0.202358 to 0.358466,
while cosine remained 0.997666.

The higher absolute error under contrast shift shows why one calibration distribution is
insufficient even when cosine looks stable. Whether that change affects a VLM answer
depends on downstream normalization and attention, which this lab does not model.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Calibrate and regress each modality and bridge component rather than applying a text-only decision globally.

### Acceptance and rollback gate

Stratify calibration/evaluation by modality, resolution, prompt length, OCR/chart cases,
and component; measure component error plus end-task multimodal quality.

### How this conclusion can fail

A text-only calibration set never exercises the patch projection. Average image
embeddings can also hide OCR, diagrams, dark images, or saturated regions. Another
mistake is to use image reconstruction metrics for a model whose deployment objective is
answer correctness or grounding.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map floating-point 64-channel, 16×16 patch projection
and group-192 INT4-dequantized projection weights back to the derivation. Verify the
printed environment, then check that same weights, image shape, projection stride,
normal/high-contrast paired inputs stayed fixed. Read projection-output
RMSE/MAE/cosine/max error for each image distribution before applying the acceptance
gate; the artifact-writing cell retains the complete structured result from the recorded
run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/21-multimodal-quantization/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Capture activation ranges from photographs, documents, charts, OCR-heavy images, and
high-contrast synthetic cases. Quantize vision encoder, connector, and decoder
separately, then run end-to-end task slices. Use mixed precision when one modality
component is consistently more sensitive.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 21's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
