# Lesson 28 — GPU Memory, Concurrency, and Cost Estimation

> **Puzzle:** How many requests fit after INT4 weight compression, and which hidden assumptions can invalidate that number?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

Capacity uses total/usable HBM, weight and scale bytes, runtime reserve, workspaces, KV per request, fragmentation, tensor parallelism, and traffic context distribution.

Quick mental model:

- Capacity starts from usable memory after runtime reserve, weights, workspaces, and fragmentation allowance.
- Per-request KV cache depends on context and cache dtype.
- Cost per token also depends on achieved throughput and utilization, not GPU price alone.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

A first bound is `requests = floor((usable - weights - workspace) / KV_per_request)`. Cost per token then depends on hourly price divided by achieved, quality-approved tokens per hour.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

INT4 ideal bytes may make weights fit while leaving no useful KV/concurrency margin. Multi-GPU sharding adds communication and changes both cost and latency.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Read live free memory from the RTX GPU and build BF16 versus INT4 capacity projections for a 70B-class model without allocating the model.

The lab seeds a 70B arithmetic model with live RTX 5090 memory but explicitly does not allocate or benchmark a 70B model.

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`capacity-model`**.

## 5. Inspect, accept, or roll back

Label the result as a capacity model. It cannot establish latency, model quality, or whether a particular 70B engine will load.

Use ranges and safety margins, then validate with the actual engine's measured peak, sustained concurrency, SLO, utilization, and cloud billing unit.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `capacity-model`
- **Recorded outcome:** Arithmetic capacity projections used live GPU memory but did not claim that a 70B engine loaded or met latency SLOs.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Use ranges and safety margins, then validate the chosen point with the actual engine and traffic distribution.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/28-gpu-capacity-cost/lab.ipynb
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
