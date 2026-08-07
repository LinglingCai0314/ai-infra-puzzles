# Lesson 15 — TorchAO INT4 Weight-Only Quantization

> **Puzzle:** Can a PyTorch-native INT4 conversion reduce storage and still lose on latency?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

TorchAO conversion replaces or wraps eligible `Linear` weights with a packed tensor subclass/configuration. The Python module, packed storage, and selected matmul kernel are three inspectable layers.

Quick mental model:

- TorchAO replaces eligible modules according to a quantization configuration.
- Packed storage and executed operator evidence are distinct from a module label.
- Small batch and shape-specific overhead can outweigh lower memory traffic.

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

INT4 weight-only compute conceptually reads packed codes and group scales while BF16 activations enter the linear operation. Modern TorchAO versions may choose among packing formats and external kernel libraries such as MSLK.

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

Packing reduces persistent bytes, but conversion dependencies, scale handling, small-batch overhead, and unsupported shapes can erase latency gains. Version compatibility is part of the result.

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

Convert a BF16 linear layer with TorchAO INT4, record the resulting module type, compare output error, and time both paths.

The notebook attempts the documented native configuration inside an explicit compatibility boundary and records the exact failure class when the path cannot execute.

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

Require conversion success, storage accounting, output error, and repeated latency. A missing TorchAO install becomes an explicit compatibility result.

Require successful import/conversion, quantized tensor/module identity, storage accounting, operator evidence, output error, and repeated latency. Preserve dependency failure rather than falling back silently.

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

<!-- rtx5090-result:start -->
## Checked-in RTX 5090 result

- **Environment:** NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch 2.12.0, CUDA runtime 13.0
- **Evidence label:** `compatibility-probe`
- **Recorded outcome:** TorchAO was installed, but the native INT4 path did not execute; the dependency failure is preserved as a compatibility result.

The exact shapes, repeated samples, errors, compatibility fields, and units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) and the executed notebook output.
<!-- rtx5090-result:end -->

## Explain

Treat TorchAO INT4 as a measured backend path, not a universal performance property of four-bit weights.

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/15-torchao-int4/lab.ipynb
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

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
