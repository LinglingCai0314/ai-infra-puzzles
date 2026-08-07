# Lesson 18 — Serving INT4 with vLLM

> **Puzzle:** If a checkpoint says AWQ or GPTQ, will vLLM necessarily run it efficiently on the current GPU?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A vLLM service couples checkpoint format, quantization backend, model runner, scheduler, paged KV cache, CUDA graphs, request batching, and sampling. Linear-kernel latency is only one component.

Quick mental model:

- vLLM selects quantization kernels through a changing model-format and hardware compatibility matrix.
- Serving performance includes scheduling, KV cache, batching, and request distribution—not only linear layers.
- An import probe cannot replace a server benchmark.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Prefill cost grows with prompt work while decode repeatedly processes small token steps and reads KV cache. Continuous batching improves utilization by combining requests, but queueing changes time-to-first-token and tail latency.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

An INT4 backend can save weight memory and allow more concurrency yet be slower for batch-one shapes. Compatibility tables change with GPU generation and release.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Probe vLLM availability and benchmark a small PyTorch W4-dequantized matmul across batch sizes as a backend-independent shape warning.

The lab records vLLM availability and uses PyTorch batch-shape timings only as a warning; it labels vLLM service throughput `not_measured`.

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

The timing is labeled PyTorch GPU evidence. vLLM throughput remains `not_measured` when the package/server is absent.

Pass format/hardware load, operator, quality, TTFT, TPOT/inter-token latency, throughput, p90/p99, peak memory, and sustained-concurrency gates with a frozen request distribution.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `compatibility-probe`
- **Recorded outcome:** PyTorch shape timing was measured separately; vLLM service performance requires an installed server and load test.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Pass checkpoint-format, hardware, load, operator, quality, and service-load gates before adopting a vLLM INT4 path.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/18-vllm-int4-serving/lab.ipynb
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
