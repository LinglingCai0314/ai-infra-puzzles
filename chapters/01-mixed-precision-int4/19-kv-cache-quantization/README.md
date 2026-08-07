# Lesson 19 — KV-Cache Quantization for Long Contexts

> **Puzzle:** When context length doubles, why can KV cache dominate even after weight quantization?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

The KV cache stores keys and values per layer and request. Quantized cache additionally stores scales (and sometimes zero points) at a chosen token/head/block granularity.

Quick mental model:

- KV bytes scale linearly with batch, layers, sequence, KV heads, head dimension, and two tensors.
- Cache quantization needs scales and often changes attention input error.
- More cache capacity may increase concurrency even when single-request latency does not improve.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Cache bytes follow `2LBTHD·bytes`, while attention uses `softmax(QKᵀ/√D)V`; quantization error can perturb both logits through `K` and the weighted sum through `V`.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Fine-grained scales reduce error but add metadata and Q/DQ work. Capacity gains can improve concurrency even if a single request pays extra latency.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Quantize representative KV tensors to INT8 on CUDA, compare bytes and attention-output error, and project capacity across context lengths.

The notebook quantizes real CUDA K/V tensors, includes scale bytes, and compares attention outputs rather than reporting compression alone.

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

Report cache bytes, metadata, attention error, and any quantize/dequantize overhead separately.

Measure actual cache allocation, metadata, context-dependent attention or task error, quant/dequant cost, long-context quality, and end-to-end serving metrics.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `pytorch-gpu`
- **Recorded outcome:** INT8 cache reduced storage in this reference while introducing measurable attention-output error.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

KV quantization is primarily a capacity decision until end-to-end latency and quality are measured.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/19-kv-cache-quantization/lab.ipynb
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

- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
