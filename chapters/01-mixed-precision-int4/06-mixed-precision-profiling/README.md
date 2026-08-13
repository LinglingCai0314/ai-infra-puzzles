<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 06 — Profiling Mixed Precision and Verifying Dispatch

> **Puzzle:** If autocast made an operation faster, does that prove the intended low-precision kernel ran?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Timing and dispatch are different claims. A faster autocast region shows an
application-level effect; a PyTorch profiler event identifies framework operators; only
a lower-level trace can justify a native kernel or Tensor Core utilization claim. Good
profiling keeps those evidence levels separate instead of using one as a shortcut for
another.

## Predict before reading the result

1. Predict which PyTorch operator events should surround a BF16 matrix multiplication under autocast.
2. Explain why warm-up and synchronization are required before comparing CUDA timings.
3. Name the additional evidence needed to claim a particular native Tensor Core kernel.

## 1. Start from concrete tensors and state

Three evidence layers answer different questions: model outputs show semantic effect,
framework operators show graph dispatch, and native kernel traces show the
implementation actually launched.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | A wall-clock delta and an operator trace answer different questions. |
| 2 | Warm-up removes initialization and compilation from the steady-state sample. |
| 3 | PyTorch operator names are higher-level evidence than native kernel names; use Nsight when kernel identity matters. |

## 2. Derive the mechanism

Profiling can expose casts, copies, GEMMs, launch count, and device time. Warm-up is
required because lazy initialization, compilation, and allocator growth are not
steady-state execution.

GPU launches are asynchronous: host elapsed time can measure queue submission rather
than device completion. CUDA events timestamp work in the device stream, but
initialization, allocator growth, lazy library loading, and compilation can still
contaminate early samples. A defensible steady-state number therefore specifies warm-up,
synchronization, sample count, and a distribution statistic.

A trace adds causality. At the framework layer, events such as `aten::matmul`,
`aten::mm`, and casts reveal the operation graph and unexpected conversions. At the
native layer, kernel names and hardware counters reveal tile implementation, tensor-pipe
activity, occupancy, and bandwidth. The layers answer complementary questions; neither
makes the other redundant.

## 3. Translate the theory into an experiment

**Experiment:** Profile an autocast BF16 GEMM with PyTorch Profiler and record the relevant operator events beside CUDA-event timing.

| Experimental role | Frozen definition |
|---|---|
| Baseline | theoretical expectation that autocast selects BF16 for an eligible GEMM |
| Candidate | actual timed autocast region plus captured PyTorch operator events |
| Held constant | 2048×2048 shape, seed, GPU, five warm-ups, fifteen CUDA-event samples |
| Measurements | median/p90 latency and selected framework operator names |
| Evidence label | `pytorch-gpu` |

The notebook pairs repeated CUDA-event timing with selected PyTorch profiler events and
explicitly stops short of inventing a native kernel name.

### Code walk-through

The notebook profiles one BF16 autocast matrix multiplication and records selected
events from the PyTorch profiler. It separately times the same region with CUDA events.
Keeping trace collection outside the timed samples avoids conflating profiler overhead
with normal latency.

The result schema calls the events `pytorch_operator_events`, not `native_kernels`. That
naming is deliberate: a framework trace is sufficient to audit the Python-level path but
not to quantify Tensor Core occupancy.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| GEMM shape | 2048 × 2048 |
| Median | 0.104416 ms |
| p90 | 0.106240 ms |
| Samples | 15 |
| PyTorch operator events | {'count': 6, 'operator': 'aten::matmul'}, {'count': 6, 'operator': 'aten::to'}, {'count': 6, 'operator': 'aten::_to_copy'}, {'count': 6, 'operator': 'aten::copy_'}, {'count': 3, 'operator': 'aten::mm'} |

### What the numbers mean

The saved run measured a 0.104416 ms median and 0.106240 ms p90 over fifteen samples.
Five relevant PyTorch operator events were retained. The tight median-to-p90 spread
suggests a stable microbenchmark after warm-up, while the event list confirms that an
autocast/matmul path was captured.

Nothing in these two fields identifies one SASS kernel or reports hardware utilization.
The bounded conclusion is therefore that the application path and timing were observed;
a native dispatch claim remains open.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Use a two-part proof: controlled timing for effect and profiler evidence for dispatch; escalate to Nsight for native-kernel claims.

### Acceptance and rollback gate

First reproduce timing without a profiler, then capture a short aligned trace. Name only
the level actually observed: PyTorch operator, CUDA kernel, or end-to-end phase.

### How this conclusion can fail

Profiler traces can perturb timing, so reporting a profiled duration as production
latency is risky. Conversely, timing without a trace can reward an unintended fallback
or cached result. Other traps include missing synchronization, timing tensor allocation,
and selecting only the fastest sample.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/06-mixed-precision-profiling/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Capture the same operation in Nsight Systems to connect CPU launch, CUDA API, and kernel
timeline, then use Nsight Compute for the selected kernel's tensor-pipe and memory
metrics. Repeat with autocast disabled and with an awkward shape. Build one table that
keeps wall-clock effect, framework dispatch, native kernel, and hardware counters in
separate columns.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
