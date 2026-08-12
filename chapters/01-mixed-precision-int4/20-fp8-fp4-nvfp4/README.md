# Lesson 20 — FP8, FP4, NVFP4, and Hardware Boundaries

> **Puzzle:** Does Blackwell hardware support mean every framework build exposes the same FP8 or NVFP4 path?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A dtype name can exist at four levels: a mathematical format, a hardware instruction, a
library recipe, and a framework operator. Blackwell support does not guarantee that the
installed PyTorch, Transformer Engine, TensorRT, or ModelOpt build exposes the same FP8
or NVFP4 path. Each layer must be probed independently.

## Predict before reading the result

1. Distinguish E4M3 FP8 from E5M2 and ordinary INT4 from block-scaled NVFP4.
2. Predict whether the installed PyTorch build can execute a scaled FP8 matrix multiply.
3. State what additional evidence is needed before claiming NVFP4 performance.

## 1. Start from concrete tensors and state

Keep four layers distinct: numerical format, hardware instruction, library recipe, and
framework/operator API. `torch.float8_*` existing does not alone prove an FP8 GEMM path.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | A format definition, hardware instruction, library API, and framework kernel are four separate layers. |
| 2 | FP8 variants trade exponent range against fraction precision. |
| 3 | NVFP4 adds block scaling; it is not ordinary uniform INT4. |

## 2. Derive the mechanism

E4M3 favors precision with less range; E5M2 favors range. Scaled FP8 matmul applies
explicit scale factors. Blackwell-specific MXFP8/NVFP4 add block-scale structure and
require matching recipes and kernels.

FP8 E4M3 allocates four exponent and three fraction bits after sign, trading range for
precision; E5M2 spends another bit on range. NVFP4 uses FP4 E2M1 values with block
scaling, so its real representation includes both four-bit data and scale hierarchy.
TensorRT's current scheme uses block size 16 for NVFP4, while framework APIs and
supported axes remain version-specific.

Scaled matrix multiplication also requires choosing input and output scales. A
successful `torch._scaled_mm` call proves one framework-level path for one shape and
format; it does not prove Transformer Engine recipes or TensorRT NVFP4 kernels.

## 3. Translate the theory into an experiment

**Experiment:** Attempt native PyTorch FP8 GEMM on the RTX GPU, record error and timing when supported, and separately probe Transformer Engine and NVFP4 APIs.

| Experimental role | Frozen definition |
|---|---|
| Baseline | higher-precision reference matrix multiplication for error comparison |
| Candidate | PyTorch scaled FP8 E4M3 GEMM on RTX 5090 |
| Held constant | 1024-class matrix shape, scaling procedure, warm-up, fifteen timing samples |
| Measurements | API success, RMSE/cosine, median/p90, library availability, NVFP4 status |
| Evidence label | `pytorch-gpu` |

The lab calls PyTorch scaled FP8 matmul when available and leaves Transformer
Engine/NVFP4 unmeasured rather than equating hardware generation with framework support.

### Code walk-through

The notebook checks for float8 dtype support and calls `torch._scaled_mm` with explicit
scales. It compares the output with a higher-precision reference and times repeated CUDA
execution. Separate probes record Transformer Engine availability and leave NVFP4
`not_measured` when its recipe/operator is unavailable.

This design prevents the real FP8 result from being generalized to a different format.
The JSON names the exact API so a future software change can be detected.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| PyTorch API | torch._scaled_mm |
| FP8 GEMM | success |
| Median | 0.017568 ms |
| FP8 RMSE | 1.208455 |
| Transformer Engine installed | no |
| NVFP4 backend | not_measured |

### What the numbers mean

The scaled FP8 GEMM succeeded through `torch._scaled_mm`, with median 0.017568 ms and
p90 0.018560 ms over fifteen samples. Output cosine was 0.999285 and RMSE 1.208455 for
the tested scale and shape. Transformer Engine was not installed, and NVFP4 remained
`not_measured`.

The measured path is therefore real PyTorch GPU evidence for FP8, not proof of a
Transformer Engine or NVFP4 backend. The absolute error also shows why format support
must be paired with scaling and quality policy.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Publish a format-by-hardware-by-library matrix, not a single `supported` checkbox.

### Acceptance and rollback gate

Record compute capability, dtype/API, scaling recipe, operator success, numerical error,
timing, and library version separately for FP8, MXFP8, and NVFP4.

### How this conclusion can fail

Casting tensors to a float8 dtype without a successful matrix operator proves storage
only. Comparing raw FP8 latency against a different shape or excluding scale computation
can misstate speed. Treating NVFP4 as signed uniform INT4 loses its block-scale
semantics entirely.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Install a matching Transformer Engine or TensorRT stack in isolation, run documented FP8
and NVFP4 recipes, and capture operator identity, scale granularity, end-to-end scale
overhead, error, and latency. Build a matrix with rows for format and columns for
hardware, library, API, operator, and tested status.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [TensorRT DynamicQuantize operator](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/operators/DynamicQuantize.html)
