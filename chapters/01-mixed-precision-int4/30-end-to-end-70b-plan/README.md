# Lesson 30 — End-to-End Project: A Serviceable INT4 Plan for a 70B-Class Model

> **Puzzle:** What evidence is required to move from a four-bit checkpoint to a serviceable 70B deployment plan?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A serviceable 70B plan joins model revision, quantization/calibration, hardware topology, engine, cache policy, quality suite, workload/SLO, capacity/cost, observability, ownership, and rollback.

Quick mental model:

- The plan joins memory feasibility, backend compatibility, quality gates, performance SLOs, observability, and rollback.
- A 70B arithmetic ledger is not a successful model load.
- Every unsupported or unmeasured gate remains explicit rather than being filled with optimism.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

The project is a gate graph rather than one conversion command: memory feasibility enables engine build; engine evidence enables quality/performance tests; only passing all critical gates enables canary.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Ideal INT4 arithmetic can suggest single-GPU fit while metadata, unquantized layers, workspaces, and KV cache invalidate it. A multi-GPU plan may fit but violate latency or cost.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Combine live GPU capacity, a small CUDA mixed-bit quality probe, and a gate matrix to produce a bounded 70B deployment decision.

The final lab combines live capacity arithmetic and a small mixed-bit CUDA probe, then returns `not_ready_for_service` because the 70B engine, quality, and service gates were not executed.

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

The notebook can approve further engineering or reject single-GPU feasibility; it cannot claim a 70B engine benchmark without loading one.

Leave every unexecuted gate visibly false. Require a real 70B load, native operator trace, frozen quality suite, service-load SLO, capacity margin, cost model, canary plan, and tested rollback before deployment.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `capacity-model`
- **Recorded outcome:** The gate matrix kept unexecuted 70B engine, quality, and service tests explicit; arithmetic compression alone was insufficient.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

A defensible plan exposes every gate, owner, artifact, and reversal condition before production optimization begins.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/30-end-to-end-70b-plan/lab.ipynb
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
- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
