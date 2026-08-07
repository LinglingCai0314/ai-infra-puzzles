# Lesson 14 — bitsandbytes 4-Bit Loading: NF4, Compute Dtype, and Nested Quantization

> **Puzzle:** Does `load_in_4bit=True` specify how the layer computes?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A bitsandbytes 4-bit configuration contains at least storage codebook (`NF4` or FP4), compute dtype, optional double/nested quantization, and the module/backend that consumes it.

Quick mental model:

- Storage type, quantization codebook, and compute dtype are separate choices.
- Nested quantization compresses quantization metadata; it does not turn activation compute into two-bit arithmetic.
- Package presence and device support must be checked before claiming a bitsandbytes run.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

NF4 assigns its 16 codes non-uniformly rather than at equal integer spacing. During a linear operation the packed codes are dequantized or consumed by a fused path while activations use the configured compute dtype.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Nested quantization reduces scale metadata but does not halve activation precision. NF4 can suit normally distributed training weights, while inference latency depends on the installed kernel and shapes.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Compare a reference NF4 codebook with uniform INT4 on normally distributed weights and probe whether bitsandbytes is installed.

The lab isolates codebook reconstruction and separately records package presence, so a numerical NF4 result cannot masquerade as bitsandbytes execution.

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

The numerical comparison explains codebooks. Only an installed bitsandbytes layer would support a native-backend claim.

Capture `BitsAndBytesConfig`, package/CUDA compatibility, actual module class, storage bytes, operator evidence, output regression, and timing.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `numerical-model`
- **Recorded outcome:** Codebook behavior was measured numerically; bitsandbytes native execution is claimed only when installed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Record quantization type, compute dtype, nested-quant setting, and actual module class together.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/lab.ipynb
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

- [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
