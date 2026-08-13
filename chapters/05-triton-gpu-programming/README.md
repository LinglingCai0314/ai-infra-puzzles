<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Chapter 05 — Triton GPU Programming and CUDA Performance

[Project home](../../README.md) · [中文首页](../../README_ZH.md) · [中文本章](../../chapters-zh/05-triton-gpu-programming/README.md)

Chapter 05 is a 30-lesson path from Triton's blocked programming model to a deliverable custom
kernel. It independently reorganizes Linnea Cai's Triton GPU programming study material around
executable puzzles. Every theory topic gets a prediction, a named CUDA/library control, a
reviewed implementation, a correctness gate, retained samples, and a conclusion that states
where it stops.

The checked-in runs use an NVIDIA GeForce RTX 5090, CUDA runtime 13.0, PyTorch 2.13.0, and
Triton 3.7.1 targeting CUDA architecture 120. The execution host did not provide `nvcc`: Lesson
05 therefore retains equivalent CUDA C++ source and records the toolchain as unavailable instead
of inventing a CUDA timing. PyTorch CUDA, cuBLAS-backed `torch.mm`, SDPA, and custom Triton
paths are named separately throughout the chapter.

```mermaid
flowchart LR
  A["blocked programs + masks"] --> B["memory + benchmark"]
  B --> C["Softmax + reduction + GEMM"]
  C --> D["Norm + Attention + stability"]
  D --> E["compile + paged KV + persistence"]
  E --> F["CI + selection + delivery"]
```

## How to study this chapter

1. Predict correctness and latency before opening retained output.
2. Read the baseline name: custom CUDA source, PyTorch CUDA, cuBLAS, SDPA, and a numerical model are not interchangeable.
3. Inspect full timing samples and the environment in the JSON artifact.
4. Re-run awkward tails and layouts before using a conclusion in another operator.
5. Keep a library or PyTorch rollback until the custom kernel passes its declared gate.

## Evidence labels

| Label | What it establishes |
|---|---|
| `native-backend` | A named Triton or PyTorch CUDA path executed on the recorded RTX 5090 stack |
| `compatibility-probe` | An installed API, backend target, source, or compiler capability was inspected without claiming unexecuted performance |
| `capacity-model` | Measured values feed a transparent traffic or decision model |

## Phase I — Programming and measurement foundations

| Lesson | Puzzle | Lab |
|---:|---|---|
| 01 | [Operator Boundaries and the Cost of Small Kernels](01-operator-boundaries/README.md) | [notebook](01-operator-boundaries/lab.ipynb) |
| 02 | [Blocked Programs versus CUDA Threads](02-programming-models/README.md) | [notebook](02-programming-models/lab.ipynb) |
| 03 | [Version Identity and a Reproducible Baseline](03-reproducible-baseline/README.md) | [notebook](03-reproducible-baseline/lab.ipynb) |
| 04 | [A Tail-Safe Vector Kernel](04-first-vector-kernel/README.md) | [notebook](04-first-vector-kernel/lab.ipynb) |
| 05 | [Explicit CUDA Control and Error Boundaries](05-explicit-cuda-control/README.md) | [notebook](05-explicit-cuda-control/lab.ipynb) |
| 06 | [Diagnosing Coalescing with Strides](06-memory-coalescing/README.md) | [notebook](06-memory-coalescing/lab.ipynb) |
| 07 | [Masks and Reduction Identities](07-mask-semantics/README.md) | [notebook](07-mask-semantics/lab.ipynb) |
| 08 | [Pointer Arithmetic and Tensor Layout](08-pointer-layout/README.md) | [notebook](08-pointer-layout/lab.ipynb) |
| 09 | [Benchmark Protocol and Timing Error](09-benchmark-protocol/README.md) | [notebook](09-benchmark-protocol/lab.ipynb) |
| 10 | [Roofline Reasoning before Tuning](10-roofline-arithmetic-intensity/README.md) | [notebook](10-roofline-arithmetic-intensity/lab.ipynb) |

## Phase II — Core operators and resource trade-offs

| Lesson | Puzzle | Lab |
|---:|---|---|
| 11 | [Fused Softmax](11-fused-softmax/README.md) | [notebook](11-fused-softmax/lab.ipynb) |
| 12 | [Reduction and Scan](12-reduction-and-scan/README.md) | [notebook](12-reduction-and-scan/lab.ipynb) |
| 13 | [Matmul Tiling and the Library Boundary](13-matmul-tiling/README.md) | [notebook](13-matmul-tiling/lab.ipynb) |
| 14 | [Autotune Search and Experiment Budget](14-autotune-budget/README.md) | [notebook](14-autotune-budget/lab.ipynb) |
| 15 | [Registers, Warps, and Occupancy Trade-offs](15-resources-occupancy/README.md) | [notebook](15-resources-occupancy/lab.ipynb) |
| 16 | [Tensor Cores and Dtype Semantics](16-tensor-cores-dtypes/README.md) | [notebook](16-tensor-cores-dtypes/lab.ipynb) |
| 17 | [Fused LayerNorm and RMSNorm](17-fused-rmsnorm/README.md) | [notebook](17-fused-rmsnorm/lab.ipynb) |

## Phase III — Attention, stability, and integration

| Lesson | Puzzle | Lab |
|---:|---|---|
| 18 | [Online Softmax and Fused Attention](18-online-softmax-attention/README.md) | [notebook](18-online-softmax-attention/lab.ipynb) |
| 19 | [Numerical Stability and Cast Boundaries](19-numerical-stability/README.md) | [notebook](19-numerical-stability/lab.ipynb) |
| 20 | [Interpreter, Assertions, and Debugging Tools](20-debugging-tools/README.md) | [notebook](20-debugging-tools/lab.ipynb) |
| 21 | [From Triton Source to IR and PTX](21-ir-ptx-reading/README.md) | [notebook](21-ir-ptx-reading/lab.ipynb) |
| 22 | [PyTorch Integration with torch.compile](22-torch-compile-integration/README.md) | [notebook](22-torch-compile-integration/lab.ipynb) |
| 23 | [Paged KV Cache Addressing](23-paged-kv-gather/README.md) | [notebook](23-paged-kv-gather/lab.ipynb) |

## Phase IV — Portability and advanced scheduling

| Lesson | Puzzle | Lab |
|---:|---|---|
| 24 | [Backend Portability and the ROCm Boundary](24-backend-portability/README.md) | [notebook](24-backend-portability/lab.ipynb) |
| 25 | [Dynamic Shapes and Specialization](25-dynamic-shapes/README.md) | [notebook](25-dynamic-shapes/lab.ipynb) |
| 26 | [Persistent Scheduling and the TMA Boundary](26-persistent-kernels/README.md) | [notebook](26-persistent-kernels/lab.ipynb) |
| 27 | [CUTLASS, cuBLAS, cuDNN, or Triton?](27-library-or-custom/README.md) | [notebook](27-library-or-custom/lab.ipynb) |

## Phase V — CI, selection, and delivery

| Lesson | Puzzle | Lab |
|---:|---|---|
| 28 | [Performance Regression CI](28-performance-regression-ci/README.md) | [notebook](28-performance-regression-ci/lab.ipynb) |
| 29 | [A Triton-versus-CUDA Decision Framework](29-selection-framework/README.md) | [notebook](29-selection-framework/lab.ipynb) |
| 30 | [From Slow Subgraph to Deliverable Kernel](30-deliverable-kernel/README.md) | [notebook](30-deliverable-kernel/lab.ipynb) |

## Shared implementation

The executable kernels live in
[`scripts/chapter05_runtime.py`](../../scripts/chapter05_runtime.py). Keeping one reviewed
source prevents thirty notebooks from drifting while every lesson still has an independent entry
point and canonical result. Lesson 05 also contains
[`vector_affine.cu`](05-explicit-cuda-control/vector_affine.cu), the explicit CUDA control that can be
built when a local CUDA Toolkit is available.

## Reproduce and validate

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start 1 --end 30
python3 scripts/build_chapter05_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 05
python3 scripts/audit_chapter05_delivery.py
```
