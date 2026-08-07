# Lesson 27 — Production Deployment, Versioning, and Rollback

> **Puzzle:** What makes a quantized release safely reversible?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

A release unit includes immutable model/tokenizer/recipe/runtime/container identities, metrics, canary policy, observability, and an already verified rollback target.

Quick mental model:

- Model, tokenizer, quantization recipe, runtime, and GPU compatibility form one release unit.
- Canary gates need quality, latency, error-rate, and capacity thresholds.
- Rollback must reference an already verified immutable baseline.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

Promotion is a state machine: offline gates -> load/smoke -> shadow -> canary -> broader rollout. Every transition consumes fixed evidence and has an automatic stop/rollback condition.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Slow rollout reduces blast radius but delays benefit; aggressive rollout increases risk. Rollback speed depends on keeping the baseline warm and compatible with current traffic.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Evaluate a synthetic candidate against frozen gates and emit a release decision plus rollback manifest from measured CUDA output error and timing.

The notebook converts measured CUDA error and timing into a deterministic synthetic release decision and rollback manifest, without claiming live traffic.

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

The manifest is a deployment-control exercise, not evidence that a real service was canaried.

Version every artifact, define quality/latency/error/capacity thresholds, monitor slices, and test the rollback command before canary traffic.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `capacity-model`
- **Recorded outcome:** Frozen synthetic gates produced a deterministic release or rollback decision; no live service canary was claimed.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Automate the decision and rollback metadata before exposing traffic; never improvise rollback after a regression.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/27-production-rollout/lab.ipynb
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
