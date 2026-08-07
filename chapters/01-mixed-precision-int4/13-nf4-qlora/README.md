# Lesson 13 — NF4 and QLoRA: A 4-Bit Fine-Tuning Memory Ledger

> **Puzzle:** If the frozen base model is four-bit, where does fine-tuning memory still go?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

QLoRA freezes a four-bit base, computes through a wider dtype, and trains LoRA matrices. The memory ledger still includes adapters, gradients, optimizer states, activations, temporary dequantization, and allocator reserve.

Quick mental model:

- QLoRA freezes a quantized base and trains small low-rank adapters.
- Optimizer state and gradients apply to trainable adapters, while activations remain a major runtime cost.
- NF4 is a non-uniform codebook designed for normally distributed weights.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

A rank-`r` adapter adds `ΔW = A·B` with roughly `r(in+out)` trainable parameters instead of `in×out`. NF4 provides a non-uniform 16-value codebook suited to normally distributed pretrained weights; double quantization compresses scale metadata.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Lower base storage enables larger models, but sequence length and activation checkpointing often dominate training memory. Adapter rank trades capacity against trainable state and compute.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Build a 7B-class memory ledger and run a CUDA low-rank adapter forward/backward over a frozen fake-quantized base matrix.

The lab combines a 7B-class arithmetic ledger with a real CUDA backward pass where only low-rank adapter tensors receive gradients.

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

Separate frozen base storage, trainable parameters, gradients, optimizer estimate, and activations.

Reconcile theoretical and measured peak memory, confirm the base has no gradients, list compute dtype and optimizer, and validate downstream quality against a frozen baseline.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** The frozen four-bit base reduced weight storage, while adapters, optimizer state, and activations remained separate costs.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Four-bit base weights reduce one ledger line; sequence activations and adapter training state still control feasibility.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/13-nf4-qlora/lab.ipynb
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

- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
