# Chapter 01 — Mixed Precision and INT4 Quantization

> Learn how numerical formats become storage layouts, GPU operators, memory
> costs, latency changes, quality trade-offs, and production decisions.

[← Project homepage](../../README.md)

## How to study this chapter

Every lesson contains a theory `README.md`, an executable `lab.ipynb` with
saved RTX 5090 outputs, and a compact JSON artifact. Follow the same loop:

```text
Predict → Run → Inspect → Explain
```

The notes use a consistent five-part reasoning path:

```text
Concrete object → Mechanism/equation → Engineering trade-off
               → Reproducible evidence → Acceptance or rollback
```

Lessons 02–30 are full tutorials rather than artifact indexes. Each one derives
its mechanism from concrete tensors, freezes a baseline/candidate protocol,
places selected RTX 5090 measurements directly in the note, explains those
numbers, walks through the notebook code, and ends with a lesson-specific
failure analysis and extension. The notebooks retain the complete original GPU
code and outputs while surrounding them with theory before and after execution.

This retains the useful conceptual path of the study curriculum while replacing
generic prose with lesson-specific formulas, tensor objects, failure modes, and
experiments. A numerical model can explain a mechanism; it cannot stand in for
TensorRT, vLLM, CUTLASS, bitsandbytes, ModelOpt, or Transformer Engine execution.

## Lessons: theory transformed into evidence

This is not only a file index. The middle column states the theoretical idea
that each lab turns into an observable object, invariant, or decision gate.

| # | Lesson | Theory → experiment bridge | Evidence |
|---:|---|---|---|
| 01 | [Precision formats: FP32, TF32, FP16, BF16, FP8, INT8, and INT4](01-precision-formats/README.md) | Separate storage bits, compute dtype, accumulator, executed operator, and speed; test the full ledger on one model. | `native-backend` |
| 02 | [Tensor Core Constraints for Low-Precision GEMM](02-tensor-core-constraints/README.md) | Treat dtype, layout, alignment, and tile shape as joint dispatch conditions; compare aligned and awkward GEMMs. | `pytorch-gpu` |
| 03 | [PyTorch AMP: autocast and GradScaler](03-pytorch-amp/README.md) | Model AMP as a forward–backward–unscale–step–update control loop, not a global dtype switch. | `pytorch-gpu` |
| 04 | [Why BF16 Is Often the First Low-Precision Choice](04-bf16-first/README.md) | Contrast exponent range and fraction precision, then measure overflow, error, and latency separately. | `pytorch-gpu` |
| 05 | [Diagnosing FP16 Overflow and Gradient Scaling Failures](05-fp16-overflow/README.md) | Locate the first non-finite or zero-gradient stage and test what loss scaling can and cannot repair. | `pytorch-gpu` |
| 06 | [Profiling Mixed Precision and Verifying Dispatch](06-mixed-precision-profiling/README.md) | Pair repeated timing with operator evidence; do not infer a kernel merely from a speed change. | `pytorch-gpu` |
| 07 | [Inference Precision Layers: Weights, Activations, and KV Cache](07-inference-precision-layers/README.md) | Build separate memory accounts for weights, activations, accumulators, cache, and workspace. | `pytorch-gpu` |
| 08 | [Quantization Math: Scale, Zero Point, Group Size, and Error](08-quantization-math/README.md) | Derive quantize/dequantize equations and expose the error–metadata trade-off as group size changes. | `numerical-model` |
| 09 | [PTQ Calibration Data: Sampling and Coverage](09-ptq-calibration/README.md) | Freeze ranges on calibration data and judge them on held-out domains, tails, and clipping rates. | `numerical-model` |
| 10 | [INT8 SmoothQuant and Activation Outliers](10-smoothquant/README.md) | Verify reciprocal channel scaling preserves `XWᵀ`, then test whether combined W8A8 error improves. | `numerical-model` |
| 11 | [GPTQ: Second-Order Intuition and Layer Reconstruction](11-gptq/README.md) | Replace raw weight error with input-weighted layer-output error and a sensitivity-aware fallback. | `numerical-model` |
| 12 | [AWQ: Protecting Salient Weights in W4A16](12-awq/README.md) | Use activation evidence to protect salient channels while keeping W4A16 storage and compute distinct. | `numerical-model` |
| 13 | [NF4 and QLoRA: A 4-Bit Fine-Tuning Memory Ledger](13-nf4-qlora/README.md) | Account for the frozen base, LoRA weights, gradients, optimizer state, and activations independently. | `pytorch-gpu` |
| 14 | [bitsandbytes 4-Bit Loading: NF4, Compute Dtype, and Nested Quantization](14-bitsandbytes-4bit/README.md) | Separate codebook, stored representation, compute dtype, and nested metadata before probing the backend. | `numerical-model` |
| 15 | [TorchAO INT4 Weight-Only Quantization](15-torchao-int4/README.md) | Require conversion, packed storage, operator identity, numerical error, and latency as separate gates. | `compatibility-probe` |
| 16 | [TensorRT INT4 Block Quantization: Q/DQ, Packing, and WoQ](16-tensorrt-int4/README.md) | Connect block scales and Q/DQ semantics to exact two-nibble packing without claiming an absent TensorRT engine. | `pytorch-gpu` |
| 17 | [ModelOpt to TensorRT-LLM Quantization Pipelines](17-modelopt-tensorrt-llm/README.md) | Turn every tool boundary into a versioned manifest containing calibration, format, build, and rollback identity. | `compatibility-probe` |
| 18 | [Serving INT4 with vLLM](18-vllm-int4-serving/README.md) | Separate checkpoint compatibility and kernel dispatch from scheduler, KV-cache, batching, and request-load effects. | `compatibility-probe` |
| 19 | [KV-Cache Quantization for Long Contexts](19-kv-cache-quantization/README.md) | Derive cache bytes from layers, sequence, KV heads, head size, batch, and dtype; measure attention error separately. | `pytorch-gpu` |
| 20 | [FP8, FP4, NVFP4, and Hardware Boundaries](20-fp8-fp4-nvfp4/README.md) | Distinguish a format, hardware instruction, library recipe, and framework kernel; execute only the available FP8 path. | `pytorch-gpu` |
| 21 | [Quantizing Vision and Multimodal Models](21-multimodal-quantization/README.md) | Expose modality-specific activation distributions and test why text-only calibration cannot cover the vision path. | `pytorch-gpu` |
| 22 | [Packaging an INT4 Inference Deliverable](22-int4-inference-package/README.md) | Treat weights, scales, tokenizer, configuration, hashes, and load-time validation as one deployable contract. | `pytorch-gpu` |
| 23 | [Accuracy Regression Tests for Quantized Models](23-accuracy-regression/README.md) | Convert average similarity into a suite of task, tail, layer, and deterministic acceptance gates. | `pytorch-gpu` |
| 24 | [Benchmark Design: Throughput, Latency, Concurrency, and Memory](24-benchmark-design/README.md) | Relate latency distributions, arrival load, batching, throughput, and memory to a frozen service SLO. | `pytorch-gpu` |
| 25 | [Failure Modes: Outliers, Long Context, MoE, and Small Batches](25-quantization-failure-modes/README.md) | Stress independent failure axes instead of averaging away outliers, routing imbalance, long context, and tiny batches. | `pytorch-gpu` |
| 26 | [Mixed-Bit Strategies and Sensitive-Layer Fallback](26-mixed-bit-fallback/README.md) | Rank layers by output sensitivity and spend a fixed higher-precision budget where it reduces error most. | `pytorch-gpu` |
| 27 | [Production Deployment, Versioning, and Rollback](27-production-rollout/README.md) | Encode model, quantizer, engine, hardware, canary, observability, and rollback as an immutable release manifest. | `capacity-model` |
| 28 | [GPU Memory, Concurrency, and Cost Estimation](28-gpu-capacity-cost/README.md) | Move from nominal weight bits to a capacity ledger with cache, workspace, reserve, concurrency, and cost assumptions. | `capacity-model` |
| 29 | [Custom Kernels: Packing, Dequantization, and CUTLASS Boundaries](29-custom-int4-kernels/README.md) | Trace pack → load → unpack/dequant → MMA → epilogue and compare fused versus materialized data movement. | `pytorch-gpu` |
| 30 | [End-to-End Project: A Serviceable INT4 Plan for a 70B-Class Model](30-end-to-end-70b-plan/README.md) | Join feasibility, engine, quality, SLO, cost, observability, canary, and rollback into a gate graph. | `capacity-model` |

## Chapter environment policy

Every GPU experiment reports the GPU, compute capability, PyTorch and CUDA
runtime, shapes, warm-up policy, repetitions, units, and a bounded conclusion.
The checked-in reference outputs come from an NVIDIA GeForce RTX 5090, but they
are not universal performance rankings.

Run all lightweight labs from the repository root with:

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 01
python3 scripts/validate_chapter.py 01
python3 scripts/audit_chapter01_delivery.py
```

Lesson 01 is a full Qwen/TorchAO comparison and may download a model. Lessons
02–30 use synthetic tensors so readers can isolate each mechanism without
downloading 70B-class checkpoints.
