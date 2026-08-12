# Lesson 18 — Serving INT4 with vLLM

> **Puzzle:** If a checkpoint says AWQ or GPTQ, will vLLM necessarily run it efficiently on the current GPU?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Serving performance belongs to a runtime, not to a checkpoint label. vLLM combines
quantized linear kernels with scheduling, continuous batching, paged KV cache, prefix
caching, and a request distribution. A PyTorch microbenchmark can warn about shape
sensitivity, but it cannot stand in for requests-per-second or latency percentiles from
a vLLM server.

## Predict before reading the result

1. Separate checkpoint-format support, hardware support, kernel dispatch, and service-load performance.
2. Predict whether the reference W4 dequantized matrix path wins at every tested batch.
3. Design a serving workload that reports TTFT and inter-token latency separately.

## 1. Start from concrete tensors and state

A vLLM service couples checkpoint format, quantization backend, model runner, scheduler,
paged KV cache, CUDA graphs, request batching, and sampling. Linear-kernel latency is
only one component.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | vLLM selects quantization kernels through a changing model-format and hardware compatibility matrix. |
| 2 | Serving performance includes scheduling, KV cache, batching, and request distribution—not only linear layers. |
| 3 | An import probe cannot replace a server benchmark. |

## 2. Derive the mechanism

Prefill cost grows with prompt work while decode repeatedly processes small token steps
and reads KV cache. Continuous batching improves utilization by combining requests, but
queueing changes time-to-first-token and tail latency.

Prefill and Decode produce different matrix shapes and interact differently with
batching. Service throughput also depends on arrival rate, prompt/output lengths,
scheduler policy, cache capacity, and queueing. A weight-only checkpoint that loads
successfully can still fall back to a slow kernel for some layers or lose its memory
benefit to KV cache at long context.

The acceptance chain is format metadata → model load → quantized module/operator trace →
output quality → controlled request workload → latency/throughput/capacity. An import
probe only reaches the first compatibility edge.

## 3. Translate the theory into an experiment

**Experiment:** Probe vLLM availability and benchmark a small PyTorch W4-dequantized matmul across batch sizes as a backend-independent shape warning.

| Experimental role | Frozen definition |
|---|---|
| Baseline | BF16 PyTorch matrix path for batches 1, 8, and 32 |
| Candidate | reference dequantized W4 matrix path at the same shapes |
| Held constant | weight/input shapes, GPU, warm-up, repetitions; no server or scheduler |
| Measurements | operator median/p90 by batch plus vLLM installation and service-benchmark status |
| Evidence label | `compatibility-probe` |

The lab records vLLM availability and uses PyTorch batch-shape timings only as a
warning; it labels vLLM service throughput `not_measured`.

### Code walk-through

The notebook probes vLLM availability, then runs a backend-independent PyTorch shape
experiment. The W4 candidate is a dequantized reference tensor, so it tests how the
resulting matrix shape behaves—not vLLM's AWQ/GPTQ kernel. Results are stored under
`pytorch_shape_warning` to make that boundary visible.

A true service cell would start a server, wait for readiness, issue a frozen request
trace, collect TTFT/ITL/latency percentiles and throughput, then terminate cleanly. None
of that is synthesized here.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| vLLM installed | no |
| Service benchmark | not_measured |
| Batch 1 BF16 median | 0.019520 ms |
| Batch 1 reference W4 median | 0.019424 ms |
| Batch 32 BF16 median | 0.018976 ms |
| Batch 32 reference W4 median | 0.019072 ms |

### What the numbers mean

The tiny matrix probe produced nearly tied medians: at batch 1, BF16 was 0.019520 ms and
the reference W4-dequant tensor 0.019424 ms; at batch 8 they were 0.019168 and 0.018912
ms; at batch 32 the candidate reversed slightly to 0.019072 versus 0.018976 ms. vLLM was
not installed and service performance is explicitly `not_measured`.

Sub-microsecond differences of this kind are not a serving result. They show that shape
can reverse a small operator comparison and reinforce why a full request workload is
needed.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Pass checkpoint-format, hardware, load, operator, quality, and service-load gates before adopting a vLLM INT4 path.

### Acceptance and rollback gate

Pass format/hardware load, operator, quality, TTFT, TPOT/inter-token latency,
throughput, p90/p99, peak memory, and sustained-concurrency gates with a frozen request
distribution.

### How this conclusion can fail

Reporting this table as vLLM speed would mislabel the backend and ignore scheduling.
Other traps are benchmarking one warm cache prompt, mixing different model revisions,
omitting output length, and comparing throughput at unequal latency or quality.
Quantization compatibility matrices also change across versions, so the exact release
must be pinned.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/18-vllm-int4-serving/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Install a supported vLLM release in a separate environment, load one documented AWQ or
GPTQ model, confirm module/operator selection, and run `vllm bench serve` with fixed
prompt/output distributions and concurrency. Report TTFT p50/p95, ITL, end-to-end
latency, tokens/s, GPU memory, and rejected requests.

## Evidence boundary

**Evidence label:** [`compatibility-probe`](../README.md#evidence-labels).

## References

- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
