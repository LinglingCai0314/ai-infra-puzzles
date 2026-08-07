#!/usr/bin/env python3
"""Build the original notes and executable notebooks for Chapter 01.

The source curriculum determines topic order only. All prose, experiments, and
code emitted here are repository-original material.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "01-mixed-precision-int4"

PYTORCH_AMP = ("PyTorch AMP documentation", "https://docs.pytorch.org/docs/stable/amp.html")
CUDA_GUIDE = ("CUDA Programming Guide", "https://docs.nvidia.com/cuda/cuda-programming-guide/index.html")
TRT_QUANT = ("TensorRT quantization schemes", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html")
VLLM_QUANT = ("vLLM quantization documentation", "https://docs.vllm.ai/en/latest/features/quantization/")
TE_DOCS = ("NVIDIA Transformer Engine documentation", "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html")
BNB_DOCS = ("Transformers bitsandbytes guide", "https://huggingface.co/docs/transformers/main/quantization/bitsandbytes")
TORCHAO_DOCS = ("TorchAO documentation", "https://docs.pytorch.org/ao/stable/index.html")


LESSONS = [
    dict(no=2, slug="tensor-core-constraints", title="Tensor Core Constraints for Low-Precision GEMM",
         puzzle="A low-precision dtype is available, so will every matrix multiplication automatically become a fast Tensor Core operation?",
         concepts=["A dtype is only one dispatch condition; layout, dimensions, alignment, and backend policy also select the kernel.", "Arithmetic intensity separates compute-bound GEMMs from shapes dominated by memory traffic or launch overhead.", "Timing establishes performance for a shape; operator or kernel evidence establishes what ran."],
         experiment="Time FP32 and BF16 matrix multiplications with aligned and deliberately awkward dimensions on the same GPU.",
         inspect="Compare medians by dtype and shape. The lab does not infer Tensor Core use from speed alone; it records a PyTorch GPU timing baseline for later profiler work.",
         conclusion="Low precision creates an opportunity, not a guarantee. Preserve exact shapes and profiler evidence when deciding whether a Tensor Core path was reached.", label="pytorch-gpu", refs=[CUDA_GUIDE]),
    dict(no=3, slug="pytorch-amp", title="PyTorch AMP: autocast and GradScaler",
         puzzle="Can mixed-precision training be reduced to wrapping the forward pass in autocast?",
         concepts=["Autocast selects lower precision per eligible operation; it does not permanently convert every tensor.", "GradScaler changes loss magnitude before backward, unscales gradients before the optimizer step, and adapts its scale.", "The optimizer state and usually the master parameters remain higher precision."],
         experiment="Train a small CUDA MLP with BF16 autocast and GradScaler while recording loss, parameter dtype, output dtype, gradient finiteness, and scale history.",
         inspect="A valid loop needs finite gradients and an optimizer update. An autocast dtype printout alone is not a training result.",
         conclusion="AMP is a control loop across forward, backward, unscale, step, and update—not a global dtype switch.", label="pytorch-gpu", refs=[PYTORCH_AMP]),
    dict(no=4, slug="bf16-first", title="Why BF16 Is Often the First Low-Precision Choice",
         puzzle="FP16 and BF16 both use 16 bits. Why can their numerical behavior differ dramatically?",
         concepts=["BF16 keeps an eight-bit exponent, so its range resembles FP32 while its fraction is shorter.", "FP16 has more fraction bits but a much smaller exponent range.", "A stable dtype is not automatically the fastest dtype; measure the actual workload."],
         experiment="Compare range, matrix-multiplication error, and CUDA timing for FP32, FP16, and BF16.",
         inspect="Look separately at overflow behavior, error against FP32, and latency. No single column decides every workload.",
         conclusion="BF16 is a pragmatic stability-first baseline on supported hardware, but workload-specific error and speed still need measurement.", label="pytorch-gpu", refs=[PYTORCH_AMP, CUDA_GUIDE]),
    dict(no=5, slug="fp16-overflow", title="Diagnosing FP16 Overflow and Gradient Scaling Failures",
         puzzle="When loss becomes NaN, how do we distinguish forward overflow, backward overflow, and gradient underflow?",
         concepts=["Overflow creates Inf before it becomes NaN in later arithmetic.", "Underflow silently rounds small gradients to zero.", "Loss scaling moves gradients into a representable interval but cannot repair an already-overflowed forward pass."],
         experiment="Sweep synthetic gradient magnitudes and loss scales in FP16 on CUDA, counting finite, infinite, and zero gradient values.",
         inspect="The useful evidence is the first stage where finiteness changes. A final NaN without intermediate checks is not a diagnosis.",
         conclusion="Place finiteness and zero-rate probes at forward outputs, scaled gradients, unscaled gradients, and parameters before changing the scaler policy.", label="pytorch-gpu", refs=[PYTORCH_AMP]),
    dict(no=6, slug="mixed-precision-profiling", title="Profiling Mixed Precision and Verifying Dispatch",
         puzzle="If autocast made an operation faster, does that prove the intended low-precision kernel ran?",
         concepts=["A wall-clock delta and an operator trace answer different questions.", "Warm-up removes initialization and compilation from the steady-state sample.", "PyTorch operator names are higher-level evidence than native kernel names; use Nsight when kernel identity matters."],
         experiment="Profile an autocast BF16 GEMM with PyTorch Profiler and record the relevant operator events beside CUDA-event timing.",
         inspect="Require both repeated timing and trace evidence. This lab deliberately labels PyTorch operators rather than claiming a native kernel name.",
         conclusion="Use a two-part proof: controlled timing for effect and profiler evidence for dispatch; escalate to Nsight for native-kernel claims.", label="pytorch-gpu", refs=[PYTORCH_AMP, CUDA_GUIDE]),
    dict(no=7, slug="inference-precision-layers", title="Inference Precision Layers: Weights, Activations, and KV Cache",
         puzzle="When a model is called INT4, which tensors are actually four-bit?",
         concepts=["Weight-only quantization leaves activations and accumulation in a floating-point compute dtype.", "KV cache grows with layers, sequence length, key/value heads, head dimension, batch, and cache dtype.", "Peak memory also includes temporary workspaces and allocator reserve."],
         experiment="Build a memory ledger and allocate representative BF16 and INT8 KV tensors on CUDA to validate element-count arithmetic.",
         inspect="Report each object separately. A checkpoint-size reduction does not establish the same reduction in runtime peak memory.",
         conclusion="Name the object and lifecycle whenever you name a precision: weights, activations, accumulators, or cache.", label="pytorch-gpu", refs=[VLLM_QUANT]),
    dict(no=8, slug="quantization-math", title="Quantization Math: Scale, Zero Point, Group Size, and Error",
         puzzle="Why does changing group size alter both model size and reconstruction error?",
         concepts=["Scale maps a floating-point interval to a finite code range.", "Symmetric quantization fixes zero point at zero; asymmetric quantization can spend codes more efficiently on shifted data.", "Smaller groups adapt to local ranges but require more scale metadata."],
         experiment="Quantize an outlier-containing matrix with INT4 group sizes 16, 64, and 128 and compare error plus metadata overhead.",
         inspect="Check saturation, error, and effective bits per value. Do not report the nominal four bits without scale overhead.",
         conclusion="Group size is an error–metadata–kernel compatibility decision, not a cosmetic configuration value.", label="numerical-model", refs=[TRT_QUANT]),
    dict(no=9, slug="ptq-calibration", title="PTQ Calibration Data: Sampling and Coverage",
         puzzle="Can a small calibration set represent the activation ranges that production traffic will exercise?",
         concepts=["Calibration estimates ranges or statistics; evaluation tests the frozen decision on held-out data.", "Rare domains and long sequences can dominate worst-case activation ranges.", "More samples do not help if sampling repeats the same narrow distribution."],
         experiment="Calibrate INT8 activation scales on narrow, balanced, and outlier-aware synthetic datasets, then evaluate all scales on a mixed held-out distribution.",
         inspect="Compare held-out clipping rate and error, not calibration-set reconstruction error.",
         conclusion="Choose calibration data by coverage of deployment modes, and keep it separate from the regression set.", label="numerical-model", refs=[TRT_QUANT]),
    dict(no=10, slug="smoothquant", title="INT8 SmoothQuant and Activation Outliers",
         puzzle="Can we make activations easier to quantize without changing the floating-point linear layer?",
         concepts=["SmoothQuant applies reciprocal channel scaling to activations and weights, preserving the floating-point product.", "The alpha parameter allocates quantization difficulty between activation and weight channels.", "The best alpha depends on observed activation and weight ranges."],
         experiment="Apply SmoothQuant-style channel scaling to an outlier-heavy linear layer, verify floating-point equivalence, and compare W8A8 reconstruction error over alpha values.",
         inspect="First verify algebraic equivalence; then compare quantized output error. A lower activation range alone is incomplete evidence.",
         conclusion="Outlier migration is useful only when the combined activation-plus-weight quantized path improves under a frozen calibration protocol.", label="numerical-model", refs=[("SmoothQuant paper", "https://arxiv.org/abs/2211.10438")]),
    dict(no=11, slug="gptq", title="GPTQ: Second-Order Intuition and Layer Reconstruction",
         puzzle="Why should two weights with the same magnitude receive different quantization treatment?",
         concepts=["Layer reconstruction minimizes output error under representative inputs, not raw weight error alone.", "Input covariance approximates which directions are sensitive.", "Production GPTQ uses structured second-order updates; a toy sensitivity model is not the library implementation."],
         experiment="Compare naive INT4 weight quantization with a GPTQ-inspired sensitivity fallback that preserves columns with large input-weighted error.",
         inspect="Measure layer-output error on held-out inputs and label the experiment as an intuition model, not a GPTQ kernel benchmark.",
         conclusion="Second-order information changes the objective from nearest weights to faithful layer outputs.", label="numerical-model", refs=[("GPTQ paper", "https://arxiv.org/abs/2210.17323")]),
    dict(no=12, slug="awq", title="AWQ: Protecting Salient Weights in W4A16",
         puzzle="Can activation statistics tell us which weight channels deserve more protection?",
         concepts=["AWQ identifies salient weights through activation-aware evidence.", "Equivalent scaling can move quantization difficulty while leaving the original floating-point function unchanged.", "W4A16 describes weight and activation precision; it does not mean the full graph is four-bit."],
         experiment="Search activation-aware per-channel scaling strengths for a toy W4A16 layer and compare output error with naive INT4.",
         inspect="Use held-out output error and a frozen search set. Weight-only mean error is not the optimization target.",
         conclusion="Activation-aware protection is a model-quality method; deployment speed still requires a compatible W4A16 kernel.", label="numerical-model", refs=[("AWQ paper", "https://arxiv.org/abs/2306.00978")]),
    dict(no=13, slug="nf4-qlora", title="NF4 and QLoRA: A 4-Bit Fine-Tuning Memory Ledger",
         puzzle="If the frozen base model is four-bit, where does fine-tuning memory still go?",
         concepts=["QLoRA freezes a quantized base and trains small low-rank adapters.", "Optimizer state and gradients apply to trainable adapters, while activations remain a major runtime cost.", "NF4 is a non-uniform codebook designed for normally distributed weights."],
         experiment="Build a 7B-class memory ledger and run a CUDA low-rank adapter forward/backward over a frozen fake-quantized base matrix.",
         inspect="Separate frozen base storage, trainable parameters, gradients, optimizer estimate, and activations.",
         conclusion="Four-bit base weights reduce one ledger line; sequence activations and adapter training state still control feasibility.", label="pytorch-gpu", refs=[("QLoRA paper", "https://arxiv.org/abs/2305.14314"), BNB_DOCS]),
    dict(no=14, slug="bitsandbytes-4bit", title="bitsandbytes 4-Bit Loading: NF4, Compute Dtype, and Nested Quantization",
         puzzle="Does `load_in_4bit=True` specify how the layer computes?",
         concepts=["Storage type, quantization codebook, and compute dtype are separate choices.", "Nested quantization compresses quantization metadata; it does not turn activation compute into two-bit arithmetic.", "Package presence and device support must be checked before claiming a bitsandbytes run."],
         experiment="Compare a reference NF4 codebook with uniform INT4 on normally distributed weights and probe whether bitsandbytes is installed.",
         inspect="The numerical comparison explains codebooks. Only an installed bitsandbytes layer would support a native-backend claim.",
         conclusion="Record quantization type, compute dtype, nested-quant setting, and actual module class together.", label="numerical-model", refs=[BNB_DOCS, ("QLoRA paper", "https://arxiv.org/abs/2305.14314")]),
    dict(no=15, slug="torchao-int4", title="TorchAO INT4 Weight-Only Quantization",
         puzzle="Can a PyTorch-native INT4 conversion reduce storage and still lose on latency?",
         concepts=["TorchAO replaces eligible modules according to a quantization configuration.", "Packed storage and executed operator evidence are distinct from a module label.", "Small batch and shape-specific overhead can outweigh lower memory traffic."],
         experiment="Convert a BF16 linear layer with TorchAO INT4, record the resulting module type, compare output error, and time both paths.",
         inspect="Require conversion success, storage accounting, output error, and repeated latency. A missing TorchAO install becomes an explicit compatibility result.",
         conclusion="Treat TorchAO INT4 as a measured backend path, not a universal performance property of four-bit weights.", label="compatibility-probe", refs=[TORCHAO_DOCS]),
    dict(no=16, slug="tensorrt-int4", title="TensorRT INT4 Block Quantization: Q/DQ, Packing, and WoQ",
         puzzle="What must be present in a graph and serialized weight buffer before TensorRT can consume INT4 weights?",
         concepts=["Explicit quantization represents scale decisions with Quantize/Dequantize semantics.", "Signed INT4 codes occupy two nibbles per byte when packed.", "TensorRT support has specific block-size and placement rules that a generic fake-quant experiment cannot prove."],
         experiment="Perform block INT4 Q/DQ and nibble packing on CUDA, verify exact unpacking, and separately probe the TensorRT package.",
         inspect="Packing correctness and Q/DQ error are real; engine build and latency remain unmeasured unless TensorRT is installed and executes.",
         conclusion="Validate graph semantics, packing, scales, engine inspection, and timing as separate gates.", label="pytorch-gpu", refs=[TRT_QUANT]),
    dict(no=17, slug="modelopt-tensorrt-llm", title="ModelOpt to TensorRT-LLM Quantization Pipelines",
         puzzle="Which evidence is lost when a quantized checkpoint is handed from one tool to another?",
         concepts=["A pipeline needs immutable model revision, calibration recipe, quantization metadata, build flags, and engine identity.", "FP8, INT4, and FP4 are different recipes, not interchangeable compression levels.", "Package availability is only the first compatibility gate."],
         experiment="Generate and validate a quantization handoff manifest seeded by a CUDA numerical probe, while checking ModelOpt and TensorRT-LLM availability independently.",
         inspect="A valid manifest is a reproducibility result, not an engine throughput result.",
         conclusion="Treat every tool boundary as a versioned artifact handoff with explicit validation and rollback metadata.", label="compatibility-probe", refs=[TRT_QUANT, TE_DOCS]),
    dict(no=18, slug="vllm-int4-serving", title="Serving INT4 with vLLM",
         puzzle="If a checkpoint says AWQ or GPTQ, will vLLM necessarily run it efficiently on the current GPU?",
         concepts=["vLLM selects quantization kernels through a changing model-format and hardware compatibility matrix.", "Serving performance includes scheduling, KV cache, batching, and request distribution—not only linear layers.", "An import probe cannot replace a server benchmark."],
         experiment="Probe vLLM availability and benchmark a small PyTorch W4-dequantized matmul across batch sizes as a backend-independent shape warning.",
         inspect="The timing is labeled PyTorch GPU evidence. vLLM throughput remains `not_measured` when the package/server is absent.",
         conclusion="Pass checkpoint-format, hardware, load, operator, quality, and service-load gates before adopting a vLLM INT4 path.", label="compatibility-probe", refs=[VLLM_QUANT]),
    dict(no=19, slug="kv-cache-quantization", title="KV-Cache Quantization for Long Contexts",
         puzzle="When context length doubles, why can KV cache dominate even after weight quantization?",
         concepts=["KV bytes scale linearly with batch, layers, sequence, KV heads, head dimension, and two tensors.", "Cache quantization needs scales and often changes attention input error.", "More cache capacity may increase concurrency even when single-request latency does not improve."],
         experiment="Quantize representative KV tensors to INT8 on CUDA, compare bytes and attention-output error, and project capacity across context lengths.",
         inspect="Report cache bytes, metadata, attention error, and any quantize/dequantize overhead separately.",
         conclusion="KV quantization is primarily a capacity decision until end-to-end latency and quality are measured.", label="pytorch-gpu", refs=[VLLM_QUANT]),
    dict(no=20, slug="fp8-fp4-nvfp4", title="FP8, FP4, NVFP4, and Hardware Boundaries",
         puzzle="Does Blackwell hardware support mean every framework build exposes the same FP8 or NVFP4 path?",
         concepts=["A format definition, hardware instruction, library API, and framework kernel are four separate layers.", "FP8 variants trade exponent range against fraction precision.", "NVFP4 adds block scaling; it is not ordinary uniform INT4."],
         experiment="Attempt native PyTorch FP8 GEMM on the RTX GPU, record error and timing when supported, and separately probe Transformer Engine and NVFP4 APIs.",
         inspect="A successful float8 PyTorch GEMM proves that path only. NVFP4 remains unmeasured without its library recipe and operator evidence.",
         conclusion="Publish a format-by-hardware-by-library matrix, not a single `supported` checkbox.", label="pytorch-gpu", refs=[TE_DOCS, TRT_QUANT]),
    dict(no=21, slug="multimodal-quantization", title="Quantizing Vision and Multimodal Models",
         puzzle="Why can a text-only calibration set miss important failure modes in a vision-language model?",
         concepts=["Vision encoders see patch distributions, image contrast, and positional structure unlike text MLP activations.", "A multimodal pipeline contains encoder, projector, language model, and attention/cache objects.", "Coverage and fallback decisions can differ by component."],
         experiment="Quantize a CUDA patch-projection weight and compare reconstruction error for ordinary and high-contrast synthetic images.",
         inspect="Compare domain-specific errors and keep the experiment scoped to the patch projection, not an entire VLM quality claim.",
         conclusion="Calibrate and regress each modality and bridge component rather than applying a text-only decision globally.", label="pytorch-gpu", refs=[TRT_QUANT]),
    dict(no=22, slug="int4-inference-package", title="Packaging an INT4 Inference Deliverable",
         puzzle="What files make a quantized model reproducible rather than merely loadable on one machine?",
         concepts=["A deliverable binds base revision, quantization recipe, tokenizer, tensor shapes, scales, packing, and runtime requirements.", "Checksums detect corruption but do not validate semantics.", "A smoke test and rollback pointer belong beside the artifact."],
         experiment="Create an in-memory synthetic INT4 shard on CUDA, serialize only a tiny temporary payload, verify its checksum and manifest fields, then delete the temporary file.",
         inspect="The lab validates packaging logic; it does not publish a model checkpoint.",
         conclusion="Ship a versioned contract with hashes, schema, compatibility, smoke test, and rollback—not a loose weight file.", label="pytorch-gpu", refs=[TRT_QUANT]),
    dict(no=23, slug="accuracy-regression", title="Accuracy Regression Tests for Quantized Models",
         puzzle="Can one aggregate score hide a serious quantization regression?",
         concepts=["Perplexity measures token likelihood, task accuracy measures decisions, and alignment samples cover product behavior.", "Thresholds should be frozen before examining the candidate.", "Slice-level failures can be hidden by a stable global average."],
         experiment="Run a tiny CUDA language-model head before and after INT4 weight Q/DQ, then compare cross-entropy, perplexity, top-1 agreement, and slice metrics.",
         inspect="Apply predeclared gates to every metric and slice. This synthetic probe is not a benchmark score for a named LLM.",
         conclusion="Use a layered quality gate and retain the baseline outputs needed to explain a regression.", label="pytorch-gpu", refs=[TORCHAO_DOCS]),
    dict(no=24, slug="benchmark-design", title="Benchmark Design: Throughput, Latency, Concurrency, and Memory",
         puzzle="How can the same GPU path improve throughput while worsening latency?",
         concepts=["Latency is per request; throughput is completed work per unit time.", "Batching amortizes overhead but increases queueing and memory demand.", "Median alone hides tail behavior; warm-up and repeated samples must be recorded."],
         experiment="Benchmark a CUDA MLP over several batch sizes, recording median, p90, examples per second, and peak allocated memory.",
         inspect="Compare all axes at the same shape and precision. This is an operator workload, not a vLLM service benchmark.",
         conclusion="Choose a candidate against a service-level objective, not the single largest throughput number.", label="pytorch-gpu", refs=[VLLM_QUANT]),
    dict(no=25, slug="quantization-failure-modes", title="Failure Modes: Outliers, Long Context, MoE, and Small Batches",
         puzzle="Where should a quantized system be expected to fail first?",
         concepts=["Outliers enlarge scale and waste codes on ordinary values.", "Long context expands cache and can expose positional or attention regressions.", "MoE routing and small batches create irregular, overhead-sensitive shapes."],
         experiment="Stress an INT4 linear reference with ordinary inputs, activation outliers, narrow batches, and shifted distributions on CUDA.",
         inspect="Keep a failure matrix by condition. Average error over mixed cases can conceal the exact reversal condition.",
         conclusion="Design negative tests from known mechanisms and preserve a fallback for the slice that fails.", label="pytorch-gpu", refs=[TRT_QUANT]),
    dict(no=26, slug="mixed-bit-fallback", title="Mixed-Bit Strategies and Sensitive-Layer Fallback",
         puzzle="If only a few layers cause most quantization error, should every layer use more bits?",
         concepts=["Layer sensitivity is measured by the downstream objective under representative inputs.", "Mixed-bit allocation trades metadata and kernel diversity against quality.", "Fallback layers need a deterministic rule and a fixed memory budget."],
         experiment="Quantize a six-layer CUDA MLP one layer at a time, rank sensitivity, then construct a budgeted INT4/INT8 mixed-bit candidate.",
         inspect="Compare the final end-to-end error and estimated storage, not only isolated layer rankings.",
         conclusion="Use sensitivity scans to spend precision where it protects the objective, then re-measure the assembled model.", label="pytorch-gpu", refs=[TORCHAO_DOCS]),
    dict(no=27, slug="production-rollout", title="Production Deployment, Versioning, and Rollback",
         puzzle="What makes a quantized release safely reversible?",
         concepts=["Model, tokenizer, quantization recipe, runtime, and GPU compatibility form one release unit.", "Canary gates need quality, latency, error-rate, and capacity thresholds.", "Rollback must reference an already verified immutable baseline."],
         experiment="Evaluate a synthetic candidate against frozen gates and emit a release decision plus rollback manifest from measured CUDA output error and timing.",
         inspect="The manifest is a deployment-control exercise, not evidence that a real service was canaried.",
         conclusion="Automate the decision and rollback metadata before exposing traffic; never improvise rollback after a regression.", label="capacity-model", refs=[VLLM_QUANT]),
    dict(no=28, slug="gpu-capacity-cost", title="GPU Memory, Concurrency, and Cost Estimation",
         puzzle="How many requests fit after INT4 weight compression, and which hidden assumptions can invalidate that number?",
         concepts=["Capacity starts from usable memory after runtime reserve, weights, workspaces, and fragmentation allowance.", "Per-request KV cache depends on context and cache dtype.", "Cost per token also depends on achieved throughput and utilization, not GPU price alone."],
         experiment="Read live free memory from the RTX GPU and build BF16 versus INT4 capacity projections for a 70B-class model without allocating the model.",
         inspect="Label the result as a capacity model. It cannot establish latency, model quality, or whether a particular 70B engine will load.",
         conclusion="Use ranges and safety margins, then validate the chosen point with the actual engine and traffic distribution.", label="capacity-model", refs=[VLLM_QUANT]),
    dict(no=29, slug="custom-int4-kernels", title="Custom Kernels: Packing, Dequantization, and CUTLASS Boundaries",
         puzzle="When is an INT4 pack/dequant kernel worth building instead of using an existing backend?",
         concepts=["End-to-end gain includes unpack, scale loads, dequantization, GEMM, launch overhead, and integration cost.", "A Python or composed PyTorch prototype validates semantics but is not a fused CUTLASS kernel.", "The target shape distribution determines whether specialization pays off."],
         experiment="Validate vectorized INT4 nibble packing/unpacking and time the composed PyTorch dequantize-plus-matmul path against BF16.",
         inspect="Use the result to locate overhead, not to claim CUTLASS performance. A custom-kernel project begins only after a measured gap and stable shapes.",
         conclusion="Build custom code when the existing backend misses an important, repeated shape and the recoverable end-to-end budget exceeds integration cost.", label="pytorch-gpu", refs=[CUDA_GUIDE, TRT_QUANT]),
    dict(no=30, slug="end-to-end-70b-plan", title="End-to-End Project: A Serviceable INT4 Plan for a 70B-Class Model",
         puzzle="What evidence is required to move from a four-bit checkpoint to a serviceable 70B deployment plan?",
         concepts=["The plan joins memory feasibility, backend compatibility, quality gates, performance SLOs, observability, and rollback.", "A 70B arithmetic ledger is not a successful model load.", "Every unsupported or unmeasured gate remains explicit rather than being filled with optimism."],
         experiment="Combine live GPU capacity, a small CUDA mixed-bit quality probe, and a gate matrix to produce a bounded 70B deployment decision.",
         inspect="The notebook can approve further engineering or reject single-GPU feasibility; it cannot claim a 70B engine benchmark without loading one.",
         conclusion="A defensible plan exposes every gate, owner, artifact, and reversal condition before production optimization begins.", label="capacity-model", refs=[TRT_QUANT, VLLM_QUANT, TE_DOCS]),
]


# Each entry transforms the source lesson's recurring five-part structure
# (object -> evidence -> trade-off -> reproducibility -> rollback) into precise,
# independently written theory connected to the runnable experiment.
THEORY: dict[int, dict[str, str]] = {
2: dict(
    objects="A GEMM consumes `A[M,K]` and `B[K,N]`. Dtype, strides, transposition, leading dimensions, and the three logical sizes travel together into dispatch; the word *BF16* by itself is not a kernel description.",
    mechanism="A useful first model is `FLOPs ≈ 2MKN` and `arithmetic intensity = FLOPs / bytes moved`. Large aligned tiles can amortize loads and feed matrix-multiply hardware; awkward dimensions create edge tiles, padding, or a different implementation. Tensor Core eligibility is therefore a conjunction of hardware, dtype, shape, layout, and library support.",
    tradeoff="Padding may improve tile utilization but adds work and memory. Small GEMMs may be launch- or memory-dominated, so a lower-precision peak-FLOP number may never become the bottleneck that the application sees.",
    gate="Keep the exact `M,N,K`, strides, dtype, warm-up, and repeated timing. Use an operator trace to show dispatch and Nsight Compute/System metrics before naming a native Tensor Core kernel.",
    code="The lab changes dtype and one alignment condition while keeping the GPU and timing method fixed; the output is shape evidence, not a native-kernel assertion."),
3: dict(
    objects="The AMP loop contains FP32 parameters and optimizer state, autocast-selected forward activations, gradients, a scalar loss scale, and an optimizer update. These objects do not all share one dtype or lifetime.",
    mechanism="If `g` is the true gradient and `S` is the loss scale, backward first produces `S·g`; unscale restores `g` before clipping or the optimizer step. `GradScaler` skips the step when non-finite gradients are found and adapts `S`. Autocast independently chooses eligible forward-operation dtypes.",
    tradeoff="BF16 often does not need scaling because of its exponent range, while FP16 can benefit from it. Scaling adds control logic and cannot repair a forward activation that already overflowed.",
    gate="Verify the order `zero_grad -> autocast forward -> scale(loss).backward -> unscale/step -> update`, record finite gradients and scale history, and keep the loss objective identical to the FP32 baseline.",
    code="The notebook prints parameter and output dtypes, runs the complete update loop, and records gradient finiteness rather than stopping after one autocast forward."),
4: dict(
    objects="FP16 and BF16 both occupy 16 bits, but FP16 uses 5 exponent and 10 fraction bits whereas BF16 uses 8 exponent and 7 fraction bits. The former offers finer local spacing; the latter offers a much larger dynamic range.",
    mechanism="Rounding error is governed by representable spacing near a value, while overflow is governed by exponent range. Accumulation policy adds a third variable: low-precision inputs may still accumulate into a wider type depending on the operator.",
    tradeoff="BF16 can avoid FP16 overflow but may show larger rounding error on well-scaled values. FP32 is a useful numerical reference, not automatically the production throughput winner.",
    gate="Test both a range probe and workload output error against FP32, then measure latency on the target shape. Keep BF16 only when stability and performance meet the frozen thresholds.",
    code="The lab separates large-value representability, GEMM error, and GEMM latency into three observations so one does not stand in for the others."),
5: dict(
    objects="Diagnose four checkpoints: forward outputs, scaled loss/gradients, unscaled gradients, and post-step parameters. A final NaN has already discarded the location of the first failure.",
    mechanism="FP16 normal values end near `6.55e4`; very small values enter a sparse subnormal region and can become zero. Loss scaling shifts gradient magnitudes upward during storage, but unscaling must happen before clipping and parameter updates.",
    tradeoff="An aggressive scale protects small gradients but increases overflow risk. A conservative scale avoids Inf yet may leave many gradients at zero, so the useful interval is workload-dependent.",
    gate="Log finite/Inf/zero fractions and the current scale. If the forward pass is already non-finite, change the operation or dtype; if only scaled gradients overflow, adjust scale policy.",
    code="The CUDA sweep crosses both tiny and large magnitudes at several scales and records zero and Inf fractions, making the failure stage observable."),
6: dict(
    objects="Three evidence layers answer different questions: model outputs show semantic effect, framework operators show graph dispatch, and native kernel traces show the implementation actually launched.",
    mechanism="Profiling can expose casts, copies, GEMMs, launch count, and device time. Warm-up is required because lazy initialization, compilation, and allocator growth are not steady-state execution.",
    tradeoff="A detailed profiler perturbs runtime and creates large traces; a light CUDA-event benchmark has lower overhead but less attribution. Use the least intrusive tool that can answer the current claim.",
    gate="First reproduce timing without a profiler, then capture a short aligned trace. Name only the level actually observed: PyTorch operator, CUDA kernel, or end-to-end phase.",
    code="The notebook pairs repeated CUDA-event timing with selected PyTorch profiler events and explicitly stops short of inventing a native kernel name."),
7: dict(
    objects="Inference precision belongs to separate ledgers: persistent weights, per-step activations/workspaces, accumulators, and persistent-per-request KV cache. Weight-only INT4 normally leaves activation and accumulation formats wider.",
    mechanism="For a standard cache, `bytes = 2 × layers × batch × sequence × kv_heads × head_dim × bytes_per_element`; the leading two is for keys and values. Grouped-query attention changes `kv_heads`, not the number of query heads.",
    tradeoff="Compressing weights creates room for cache or concurrency but does not shrink every runtime object. Cache quantization may increase capacity while adding Q/DQ work and attention error.",
    gate="Measure allocated/reserved/peak memory separately and reconcile them with object-level arithmetic. A checkpoint byte count is not a runtime memory result.",
    code="The lab validates the KV element-count formula with a live allocation and projects several context lengths without pretending to allocate a full model."),
8: dict(
    objects="Uniform quantization stores integer codes plus scale metadata and, for asymmetric schemes, zero points. Granularity may be per tensor, row/channel, or group/block.",
    mechanism="A common mapping is `q = clamp(round(x/s)+z, qmin, qmax)` and `x_hat = s(q-z)`. Symmetric INT4 typically uses `z=0` and a signed range near `[-8,7]`. Smaller groups estimate local ranges and reduce outlier sharing.",
    tradeoff="Smaller groups add scale loads and metadata and may miss a backend's supported block sizes. Larger groups are cheaper but one outlier can enlarge the step for many ordinary weights.",
    gate="Report nominal bits, scale/zero-point overhead, clipping rate, reconstruction error, group axis, and kernel-compatible group size together.",
    code="The notebook holds the weight matrix fixed, changes only group size, and records both error and effective bits per weight."),
9: dict(
    objects="A PTQ pipeline has a calibration distribution used to freeze quantization parameters and a disjoint evaluation distribution used to test the frozen result.",
    mechanism="Max calibration protects observed extremes but can waste most codes; percentile or learned clipping trades a controlled tail for smaller steps. Either choice fails when the calibration set omits a deployment domain.",
    tradeoff="More examples reduce sampling noise only when they add coverage. Long prompts, code, multilingual text, tool schemas, and rare outliers may need explicit strata rather than random repetition.",
    gate="Publish sampling rules, lengths/domains, seed, statistic, sample count, and held-out clipping/error. Never tune the range on the same examples used for the final quality gate.",
    code="The lab freezes scales from narrow, balanced, and outlier-aware sets and evaluates all three on one mixed held-out tensor."),
10: dict(
    objects="SmoothQuant operates on matching input channels of activation `X` and weight `W` for a linear layer `Y=XWᵀ`.",
    mechanism="For positive channel scales `s`, `(X / s)(W · s)ᵀ = XWᵀ`. Choosing `s_j` from activation and weight maxima moves channel difficulty without changing the floating-point function. The exponent `alpha` decides how much range moves toward weights.",
    tradeoff="Activation ranges become easier for INT8 while weight ranges become harder. The correct objective is combined W8A8 output error and backend performance, not activation amax alone.",
    gate="Verify floating-point equivalence first, freeze calibration statistics, sweep alpha on calibration data, and accept using held-out output/quality plus native W8A8 evidence.",
    code="The notebook checks the algebraic invariant before quantizing both sides and comparing output error across alpha values."),
11: dict(
    objects="GPTQ reconstructs one layer at a time using the layer weights and representative input activations. It targets output distortion, not unweighted distance between original and rounded weights.",
    mechanism="For weight error `ΔW` and inputs `X`, layer error is approximately `||XΔWᵀ||²`; the input Gram/Hessian approximation `XᵀX` weights sensitive directions. GPTQ uses inverse-Hessian information to compensate remaining weights as columns are quantized.",
    tradeoff="Block size and damping control memory, numerical stability, and approximation quality. Ordering and calibration data alter the result, and the packed inference kernel is a separate concern.",
    gate="Record calibration activations, damping, block/group size, ordering, layer reconstruction loss, end-task regression, and the deployed operator.",
    code="The lab is deliberately GPTQ-inspired: it uses input-weighted sensitivity and a fallback to expose the objective, while clearly not claiming GPTQModel execution."),
12: dict(
    objects="AWQ studies which weight channels are salient under observed activations and protects them within a weight-only W4A16 deployment path.",
    mechanism="Channel scaling can preserve the floating-point linear transform while changing how weight ranges are shared before INT4 rounding. Activation statistics guide the scale search because frequently excited channels can amplify small weight errors.",
    tradeoff="Protecting more channels or searching more scales costs calibration time and may reduce compression or kernel regularity. Lower weight MAE does not guarantee lower language-model loss.",
    gate="Separate search/calibration from held-out evaluation, report protected fraction and group size, and prove a W4A16 operator executed before making speed claims.",
    code="The notebook freezes calibration activations, searches scaling strength, and chooses by held-out layer-output error rather than weight error."),
13: dict(
    objects="QLoRA freezes a four-bit base, computes through a wider dtype, and trains LoRA matrices. The memory ledger still includes adapters, gradients, optimizer states, activations, temporary dequantization, and allocator reserve.",
    mechanism="A rank-`r` adapter adds `ΔW = A·B` with roughly `r(in+out)` trainable parameters instead of `in×out`. NF4 provides a non-uniform 16-value codebook suited to normally distributed pretrained weights; double quantization compresses scale metadata.",
    tradeoff="Lower base storage enables larger models, but sequence length and activation checkpointing often dominate training memory. Adapter rank trades capacity against trainable state and compute.",
    gate="Reconcile theoretical and measured peak memory, confirm the base has no gradients, list compute dtype and optimizer, and validate downstream quality against a frozen baseline.",
    code="The lab combines a 7B-class arithmetic ledger with a real CUDA backward pass where only low-rank adapter tensors receive gradients."),
14: dict(
    objects="A bitsandbytes 4-bit configuration contains at least storage codebook (`NF4` or FP4), compute dtype, optional double/nested quantization, and the module/backend that consumes it.",
    mechanism="NF4 assigns its 16 codes non-uniformly rather than at equal integer spacing. During a linear operation the packed codes are dequantized or consumed by a fused path while activations use the configured compute dtype.",
    tradeoff="Nested quantization reduces scale metadata but does not halve activation precision. NF4 can suit normally distributed training weights, while inference latency depends on the installed kernel and shapes.",
    gate="Capture `BitsAndBytesConfig`, package/CUDA compatibility, actual module class, storage bytes, operator evidence, output regression, and timing.",
    code="The lab isolates codebook reconstruction and separately records package presence, so a numerical NF4 result cannot masquerade as bitsandbytes execution."),
15: dict(
    objects="TorchAO conversion replaces or wraps eligible `Linear` weights with a packed tensor subclass/configuration. The Python module, packed storage, and selected matmul kernel are three inspectable layers.",
    mechanism="INT4 weight-only compute conceptually reads packed codes and group scales while BF16 activations enter the linear operation. Modern TorchAO versions may choose among packing formats and external kernel libraries such as MSLK.",
    tradeoff="Packing reduces persistent bytes, but conversion dependencies, scale handling, small-batch overhead, and unsupported shapes can erase latency gains. Version compatibility is part of the result.",
    gate="Require successful import/conversion, quantized tensor/module identity, storage accounting, operator evidence, output error, and repeated latency. Preserve dependency failure rather than falling back silently.",
    code="The notebook attempts the documented native configuration inside an explicit compatibility boundary and records the exact failure class when the path cannot execute."),
16: dict(
    objects="TensorRT explicit quantization represents quantization choices with Q/DQ semantics and consumes packed low-bit weights plus scales under supported block/layout constraints.",
    mechanism="For signed INT4, two 4-bit two's-complement codes occupy one byte. Block Q/DQ applies one scale to a supported group, reconstructing floating-point values for the consuming operation or enabling a fused weight-only implementation.",
    tradeoff="A valid packer can still produce an engine-incompatible graph; a valid graph can still select a slow tactic. Semantics, serialization, build, kernel selection, and runtime are separate gates.",
    gate="Round-trip every packed code, verify scale axis/block size and ONNX Q/DQ placement, inspect the built engine, then benchmark the engine against the same baseline.",
    code="The CUDA lab validates block Q/DQ and exact nibble round-trip while an independent package probe prevents a false TensorRT-engine claim."),
17: dict(
    objects="A ModelOpt-to-TensorRT-LLM handoff includes base revision, calibration corpus, recipe, per-layer exclusions, quantized tensor metadata, tokenizer, builder/runtime versions, engine flags, and rollback target.",
    mechanism="Model optimization chooses and serializes a numerical representation; the engine builder lowers it to hardware tactics. Losing group axes, scale dtype, or recipe version at the boundary can change semantics even when files load.",
    tradeoff="Pre-quantized checkpoints shorten deployment but constrain engine/version choices. Re-quantizing locally offers control but requires calibration reproducibility and more build time.",
    gate="Validate a schema and hashes at each handoff, run a deterministic smoke sample, inspect engine layers, and keep quality and performance gates separate.",
    code="The notebook creates a complete handoff manifest and a CUDA numerical fingerprint while explicitly marking ModelOpt and TensorRT-LLM availability."),
18: dict(
    objects="A vLLM service couples checkpoint format, quantization backend, model runner, scheduler, paged KV cache, CUDA graphs, request batching, and sampling. Linear-kernel latency is only one component.",
    mechanism="Prefill cost grows with prompt work while decode repeatedly processes small token steps and reads KV cache. Continuous batching improves utilization by combining requests, but queueing changes time-to-first-token and tail latency.",
    tradeoff="An INT4 backend can save weight memory and allow more concurrency yet be slower for batch-one shapes. Compatibility tables change with GPU generation and release.",
    gate="Pass format/hardware load, operator, quality, TTFT, TPOT/inter-token latency, throughput, p90/p99, peak memory, and sustained-concurrency gates with a frozen request distribution.",
    code="The lab records vLLM availability and uses PyTorch batch-shape timings only as a warning; it labels vLLM service throughput `not_measured`."),
19: dict(
    objects="The KV cache stores keys and values per layer and request. Quantized cache additionally stores scales (and sometimes zero points) at a chosen token/head/block granularity.",
    mechanism="Cache bytes follow `2LBTHD·bytes`, while attention uses `softmax(QKᵀ/√D)V`; quantization error can perturb both logits through `K` and the weighted sum through `V`.",
    tradeoff="Fine-grained scales reduce error but add metadata and Q/DQ work. Capacity gains can improve concurrency even if a single request pays extra latency.",
    gate="Measure actual cache allocation, metadata, context-dependent attention or task error, quant/dequant cost, long-context quality, and end-to-end serving metrics.",
    code="The notebook quantizes real CUDA K/V tensors, includes scale bytes, and compares attention outputs rather than reporting compression alone."),
20: dict(
    objects="Keep four layers distinct: numerical format, hardware instruction, library recipe, and framework/operator API. `torch.float8_*` existing does not alone prove an FP8 GEMM path.",
    mechanism="E4M3 favors precision with less range; E5M2 favors range. Scaled FP8 matmul applies explicit scale factors. Blackwell-specific MXFP8/NVFP4 add block-scale structure and require matching recipes and kernels.",
    tradeoff="Smaller formats reduce traffic and raise theoretical throughput but add scale selection, saturation risk, metadata, and software compatibility constraints.",
    gate="Record compute capability, dtype/API, scaling recipe, operator success, numerical error, timing, and library version separately for FP8, MXFP8, and NVFP4.",
    code="The lab calls PyTorch scaled FP8 matmul when available and leaves Transformer Engine/NVFP4 unmeasured rather than equating hardware generation with framework support."),
21: dict(
    objects="A vision-language system contains a vision encoder, patch/token embedding, projector, cross- or self-attention, language model, and KV cache. Each component sees a different activation distribution.",
    mechanism="Patch projection maps local pixel statistics into tokens; contrast and modality shifts can create channel ranges absent from text calibration. Quantization error can then propagate through normalization and attention.",
    tradeoff="Quantizing the large language component may save most bytes, while quantizing a sensitive bridge can cause disproportionate quality loss. Component-specific fallback may be cheaper than one global dtype.",
    gate="Stratify calibration/evaluation by modality, resolution, prompt length, OCR/chart cases, and component; measure component error plus end-task multimodal quality.",
    code="The notebook isolates a patch projection and compares normal versus high-contrast image distributions, carefully avoiding a full-VLM claim."),
22: dict(
    objects="A deployable package binds tensor shards, scales/zero points, shapes and packing schema, base/tokenizer revisions, runtime requirements, checksums, smoke vectors, and rollback identity.",
    mechanism="A cryptographic hash verifies bytes, while a schema verifies meaning. Both are needed: identical shapes with the wrong scale axis can be semantically corrupt yet perfectly hash-consistent.",
    tradeoff="More self-description increases package size slightly but removes fragile out-of-band assumptions. Safe serialization and shard size affect loading and distribution, not model accuracy.",
    gate="Test fresh-environment load, hash verification, schema validation, deterministic smoke output, memory budget, native operator, and rollback artifact before release.",
    code="The lab creates a tiny temporary packed payload, hashes and validates its manifest, and deletes it so no model checkpoint enters the repository."),
23: dict(
    objects="Quality evidence spans token likelihood (cross-entropy/perplexity), task metrics, output/logit agreement, safety/alignment cases, and business-specific slices.",
    mechanism="Perplexity is `exp(mean token cross-entropy)`; a small average loss change can coexist with large ranking changes on a rare slice. Top-1 agreement reveals decision changes but not whether either answer is correct.",
    tradeoff="Large suites improve coverage but slow iteration. A tiered gate uses fast deterministic smoke/regression samples first and expensive benchmarks before release.",
    gate="Freeze datasets, prompts, decoding, baseline revision, thresholds, and slice definitions. Fail on a critical slice even if the global average passes.",
    code="The CUDA probe computes loss, perplexity, overall agreement, and two slices from identical hidden states before and after INT4 Q/DQ."),
24: dict(
    objects="Benchmark outputs include latency distribution, throughput, concurrency, queueing, TTFT, token latency, memory, power/cost, and workload shape. They cannot be collapsed into one number.",
    mechanism="For a fixed operator, throughput is `batch / latency`; batching can raise throughput while each item waits longer. In a service, arrival rate and queueing add latency beyond GPU execution.",
    tradeoff="A configuration optimized for batch throughput may violate interactive p99. More concurrency improves utilization until memory pressure or scheduling raises tails.",
    gate="Declare workload distribution, warm-up, repetitions, synchronization, concurrency, percentile method, precision, and SLO before seeing the candidate.",
    code="The lab sweeps batch size and reports median, p90, examples/s, and peak allocated memory; it labels the result as an operator workload, not a server test."),
25: dict(
    objects="Failure modes map to mechanisms: range outliers, distribution shift, long-context cache/attention, MoE routing imbalance, and small irregular GEMMs.",
    mechanism="One outlier can enlarge a group scale; shifted inputs change layer-output sensitivity; batch-one and routed experts reduce matrix sizes and make launch/dequant overhead visible.",
    tradeoff="Optimizing the average case can worsen a rare but critical slice. Global fallback is safe but expensive; targeted fallback needs reliable detection and routing.",
    gate="Maintain a condition-by-metric failure matrix with reversal thresholds and reproduce each failure independently before assigning a fallback.",
    code="The lab holds weights fixed and stresses ordinary, outlier, shifted, and small-batch inputs, preserving each condition instead of averaging them together."),
26: dict(
    objects="Mixed-bit design assigns a precision/configuration to each layer or group under a memory, latency, and quality budget.",
    mechanism="A sensitivity scan replaces one layer at a time and measures downstream change. A simple allocation then spends extra bits on the largest marginal quality benefit per added byte; interactions require re-evaluating the assembled model.",
    tradeoff="More bit variants improve the Pareto frontier but fragment kernels, packing, and deployment. Layerwise rankings can change when several layers are quantized together.",
    gate="Freeze calibration/evaluation, record isolated sensitivities, budget, chosen fallback layers, final assembled quality, storage, operator coverage, and latency.",
    code="The six-layer CUDA lab ranks INT4 substitutions, gives two layers INT8, computes average bits, and re-runs end to end."),
27: dict(
    objects="A release unit includes immutable model/tokenizer/recipe/runtime/container identities, metrics, canary policy, observability, and an already verified rollback target.",
    mechanism="Promotion is a state machine: offline gates -> load/smoke -> shadow -> canary -> broader rollout. Every transition consumes fixed evidence and has an automatic stop/rollback condition.",
    tradeoff="Slow rollout reduces blast radius but delays benefit; aggressive rollout increases risk. Rollback speed depends on keeping the baseline warm and compatible with current traffic.",
    gate="Version every artifact, define quality/latency/error/capacity thresholds, monitor slices, and test the rollback command before canary traffic.",
    code="The notebook converts measured CUDA error and timing into a deterministic synthetic release decision and rollback manifest, without claiming live traffic."),
28: dict(
    objects="Capacity uses total/usable HBM, weight and scale bytes, runtime reserve, workspaces, KV per request, fragmentation, tensor parallelism, and traffic context distribution.",
    mechanism="A first bound is `requests = floor((usable - weights - workspace) / KV_per_request)`. Cost per token then depends on hourly price divided by achieved, quality-approved tokens per hour.",
    tradeoff="INT4 ideal bytes may make weights fit while leaving no useful KV/concurrency margin. Multi-GPU sharding adds communication and changes both cost and latency.",
    gate="Use ranges and safety margins, then validate with the actual engine's measured peak, sustained concurrency, SLO, utilization, and cloud billing unit.",
    code="The lab seeds a 70B arithmetic model with live RTX 5090 memory but explicitly does not allocate or benchmark a 70B model."),
29: dict(
    objects="An INT4 execution path contains pack/storage, scale loads, unpack/dequant, GEMM, epilogue, launches, and integration with framework layouts and streams.",
    mechanism="The end-to-end budget is `T = T_pack/load + T_dequant + T_gemm + T_epilogue + overhead`. Fusing stages can remove intermediate traffic; a composed PyTorch reference intentionally exposes that unfused cost.",
    tradeoff="A specialized CUTLASS/Triton kernel may win on stable shapes but costs engineering, testing, portability, and maintenance. Mature libraries remain the baseline to beat.",
    gate="First locate a repeated shape-level gap, verify pack/dequant semantics, profile roofline and memory traffic, implement, then require end-to-end gain and quality across the target shape distribution.",
    code="The lab validates nibble semantics and times a composed unpack-dequant-matmul reference, labeling it explicitly as non-fused and non-CUTLASS."),
30: dict(
    objects="A serviceable 70B plan joins model revision, quantization/calibration, hardware topology, engine, cache policy, quality suite, workload/SLO, capacity/cost, observability, ownership, and rollback.",
    mechanism="The project is a gate graph rather than one conversion command: memory feasibility enables engine build; engine evidence enables quality/performance tests; only passing all critical gates enables canary.",
    tradeoff="Ideal INT4 arithmetic can suggest single-GPU fit while metadata, unquantized layers, workspaces, and KV cache invalidate it. A multi-GPU plan may fit but violate latency or cost.",
    gate="Leave every unexecuted gate visibly false. Require a real 70B load, native operator trace, frozen quality suite, service-load SLO, capacity margin, cost model, canary plan, and tested rollback before deployment.",
    code="The final lab combines live capacity arithmetic and a small mixed-bit CUDA probe, then returns `not_ready_for_service` because the 70B engine, quality, and service gates were not executed."),
}


EXPERIMENTS: dict[int, str] = {
2: '''
shapes = {"aligned": (2048, 2048, 2048), "awkward": (2048, 2055, 2048)}
timings = {}
for name, (m, k, n) in shapes.items():
    timings[name] = {}
    for dtype in (torch.float32, torch.bfloat16):
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        timings[name][str(dtype).split(".")[-1]] = cuda_benchmark(lambda: a @ b, warmup=4, repeats=12)
result = base_result(2, "pytorch-gpu")
result.update({"shapes_mkn": shapes, "timings": timings,
               "conclusion": "Observed shape- and dtype-dependent GEMM timing; native Tensor Core identity requires a lower-level profiler."})
''',
3: '''
model = torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 64)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(256, 512, device=device); target = torch.randn(256, 64, device=device)
history = []
for step in range(6):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x); loss = torch.nn.functional.mse_loss(pred, target)
    scaler.scale(loss).backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
    history.append({"step": step, "loss": round(loss.item(), 7), "output_dtype": str(pred.dtype),
                    "grads_finite": bool(finite), "scale": float(scale_before)})
result = base_result(3, "pytorch-gpu")
result.update({"parameter_dtype": str(next(model.parameters()).dtype), "history": history,
               "conclusion": "The full autocast-scale-backward-step-update loop completed with finite gradients."})
''',
4: '''
n = 1536; a32 = torch.randn(n, n, device=device); b32 = torch.randn(n, n, device=device)
ref = a32 @ b32; rows = {}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    a, b = a32.to(dtype), b32.to(dtype); out = a @ b
    rows[str(dtype).split(".")[-1]] = {"timing": cuda_benchmark(lambda: a @ b, warmup=4, repeats=12),
                                       "error": error_metrics(ref, out.float())}
range_probe = {"fp16_1e5_finite": bool(torch.isfinite(torch.tensor([1e5], device=device).half()).item()),
               "bf16_1e5_finite": bool(torch.isfinite(torch.tensor([1e5], device=device).bfloat16()).item()),
               "fp16_max": torch.finfo(torch.float16).max, "bf16_max": torch.finfo(torch.bfloat16).max}
result = base_result(4, "pytorch-gpu"); result.update({"matrix_shape": [n, n], "formats": rows,
    "range_probe": range_probe, "conclusion": "BF16 preserved the large-value range while FP16 and BF16 showed different accuracy/performance trade-offs."})
''',
5: '''
rows = []
for magnitude in (1e-8, 1e-5, 1.0, 1e3):
    for scale in (1.0, 256.0, 65536.0):
        p = torch.ones(4096, device=device, dtype=torch.float16, requires_grad=True)
        loss = (p.float() * magnitude).sum() * scale
        loss.backward(); g = p.grad
        rows.append({"magnitude": magnitude, "loss_scale": scale,
                     "zero_fraction": round((g == 0).float().mean().item(), 6),
                     "inf_fraction": round(torch.isinf(g).float().mean().item(), 6),
                     "finite_fraction": round(torch.isfinite(g).float().mean().item(), 6)})
forward_overflow = torch.isinf(torch.tensor([1e5], device=device).half()).item()
result = base_result(5, "pytorch-gpu"); result.update({"gradient_sweep": rows,
    "forward_overflow_at_1e5": bool(forward_overflow),
    "conclusion": "Scaling changed gradient representability but could not repair an FP16 value that had already overflowed."})
''',
6: '''
from torch.profiler import ProfilerActivity, profile
import warnings
a = torch.randn(2048, 2048, device=device); b = torch.randn(2048, 2048, device=device)
def work():
    with torch.autocast("cuda", dtype=torch.bfloat16): return a @ b
timing = cuda_benchmark(work, warmup=5, repeats=15)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3): work()
torch.cuda.synchronize()
events = []
for e in prof.key_averages():
    if any(k in e.key.lower() for k in ("mm", "matmul", "to", "copy")):
        events.append({"operator": e.key, "count": e.count})
result = base_result(6, "pytorch-gpu"); result.update({"shape": [2048, 2048], "timing": timing,
    "pytorch_operator_events": events[:20], "conclusion": "Autocast timing and PyTorch operator evidence were captured; native kernel identity was not claimed."})
''',
7: '''
cfg = {"layers": 32, "kv_heads": 8, "head_dim": 128, "batch": 1}
rows = []
for seq in (2048, 8192, 32768):
    elements = 2 * cfg["layers"] * cfg["batch"] * seq * cfg["kv_heads"] * cfg["head_dim"]
    rows.append({"sequence": seq, "bf16_gib": round(elements*2/2**30, 4), "int8_gib": round(elements/2**30, 4)})
k = torch.empty(2, 4096, 8, 128, device=device, dtype=torch.bfloat16)
actual = k.numel() * k.element_size()
result = base_result(7, "pytorch-gpu"); result.update({"configuration": cfg, "projected_kv": rows,
    "allocation_probe": {"shape": list(k.shape), "bytes": actual, "dtype": str(k.dtype)},
    "conclusion": "Weights, activations, and KV cache require separate precision and memory ledger entries."})
''',
8: '''
w = torch.randn(1024, 1024, device=device); w[:, ::97] *= 12
rows = []
for group in (16, 64, 128):
    q, scales, dq = symmetric_quantize(w, bits=4, group_size=group)
    metadata_bits = scales.numel() * 16
    rows.append({"group_size": group, "error": error_metrics(w, dq), "scale_count": scales.numel(),
                 "effective_bits_per_weight": round(4 + metadata_bits / w.numel(), 5),
                 "saturation_fraction": round((q.abs() == 7).float().mean().item(), 6)})
result = base_result(8, "numerical-model"); result.update({"shape": list(w.shape), "group_results": rows,
    "conclusion": "Smaller groups reduced local range sharing at the cost of more scale metadata."})
''',
9: '''
def sample(n, mode):
    x = torch.randn(n, 512, device=device)
    if mode == "shifted": x = x * 2.5 + 1.5
    if mode == "rare": x[:, ::64] *= 10
    return x
eval_x = torch.cat([sample(1024,"base"), sample(512,"shifted"), sample(128,"rare")])
cal_sets = {"narrow": sample(1024,"base"), "balanced": torch.cat([sample(512,"base"),sample(512,"shifted")]),
            "outlier_aware": torch.cat([sample(448,"base"),sample(448,"shifted"),sample(128,"rare")])}
rows = {}
for name, cal in cal_sets.items():
    scale = cal.abs().max() / 127; q = torch.round(eval_x/scale).clamp(-128,127); dq=q*scale
    rows[name] = {"scale": round(scale.item(),8), "clipping_fraction": round((eval_x.abs()>127*scale).float().mean().item(),8),
                  "error": error_metrics(eval_x,dq)}
result=base_result(9,"numerical-model"); result.update({"evaluation_shape":list(eval_x.shape),"calibration_results":rows,
    "conclusion":"Held-out coverage, not calibration reconstruction, determined clipping and error."})
''',
10: '''
batch,in_f,out_f=512,512,384; x=torch.randn(batch,in_f,device=device); w=torch.randn(out_f,in_f,device=device)
x[:,::64]*=18; reference=x@w.t(); rows=[]
for alpha in (0.0,0.25,0.5,0.75,1.0):
    ax=x.abs().amax(0).clamp_min(1e-6); aw=w.abs().amax(0).clamp_min(1e-6)
    s=(ax.pow(alpha)/aw.pow(1-alpha)).clamp_min(1e-6); xs=x/s; ws=w*s
    equivalence=(reference-xs@ws.t()).abs().max().item()
    sx=xs.abs().max()/127; sw=ws.abs().max()/127
    qx=torch.round(xs/sx).clamp(-128,127)*sx; qw=torch.round(ws/sw).clamp(-128,127)*sw
    rows.append({"alpha":alpha,"float_equivalence_max_abs":round(equivalence,6),"output_error":error_metrics(reference,qx@qw.t())})
result=base_result(10,"numerical-model"); result.update({"shape":[batch,in_f,out_f],"alpha_sweep":rows,
    "conclusion":"Reciprocal scaling preserved the floating-point layer while changing combined W8A8 error."})
''',
11: '''
n,in_f,out_f=1024,256,192; x=torch.randn(n,in_f,device=device); x[:,::31]*=5; w=torch.randn(out_f,in_f,device=device)
ref=x@w.t(); _,_,naive=symmetric_quantize(w,bits=4,group_size=64); naive_out=x@naive.t()
sensitivity=x.square().mean(0)*((w-naive).square().mean(0)); keep=torch.topk(sensitivity,k=in_f//8).indices
aware=naive.clone(); aware[:,keep]=w[:,keep]; aware_out=x@aware.t()
result=base_result(11,"numerical-model"); result.update({"shape":[n,in_f,out_f],"preserved_column_fraction":round(len(keep)/in_f,4),
    "naive_output_error":error_metrics(ref,naive_out),"sensitivity_fallback_error":error_metrics(ref,aware_out),
    "conclusion":"Input-weighted sensitivity changed which quantization errors mattered; this is a GPTQ intuition model, not GPTQModel execution."})
''',
12: '''
cal=torch.randn(1024,256,device=device); cal[:,::29]*=7; test=torch.randn(512,256,device=device); test[:,::29]*=7
w=torch.randn(192,256,device=device); ref=test@w.t(); rows=[]
for alpha in (0.0,0.25,0.5,0.75,1.0):
    importance=cal.abs().mean(0).clamp_min(1e-5).pow(alpha); scaled=w*importance
    _,_,dq=symmetric_quantize(scaled,bits=4,group_size=64); restored=dq/importance
    rows.append({"alpha":alpha,"heldout_error":error_metrics(ref,test@restored.t())})
best=min(rows,key=lambda r:r["heldout_error"]["rmse"])
result=base_result(12,"numerical-model"); result.update({"alpha_sweep":rows,"best_alpha":best["alpha"],
    "conclusion":"Activation-aware scaling changed held-out W4A16 output error; no production AWQ kernel was claimed."})
''',
13: '''
params=7_000_000_000; rank=16; hidden=4096; layers=32
ledger={"bf16_base_gib":round(params*2/2**30,3),"int4_base_ideal_gib":round(params*0.5/2**30,3),
        "lora_trainable_mib":round(layers*2*hidden*rank*2/2**20,3),"adam_states_mib":round(layers*2*hidden*rank*8/2**20,3)}
dim=1024; base=torch.randn(dim,dim,device=device); _,_,base_q=symmetric_quantize(base,bits=4,group_size=128); base_q=base_q.detach()
a=torch.nn.Parameter(torch.randn(dim,rank,device=device)*0.01); b=torch.nn.Parameter(torch.zeros(rank,dim,device=device))
x=torch.randn(64,dim,device=device); target=torch.randn(64,dim,device=device)
loss=torch.nn.functional.mse_loss(x@base_q.t()+(x@a)@b,target); loss.backward()
result=base_result(13,"pytorch-gpu"); result.update({"seven_b_ledger":ledger,"toy_loss":round(loss.item(),7),
    "base_requires_grad":base_q.requires_grad,"adapter_grad_finite":bool(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()),
    "conclusion":"The frozen four-bit base reduced weight storage, while adapters, optimizer state, and activations remained separate costs."})
''',
14: '''
import importlib.util
nf4=torch.tensor([-1.0,-0.6962,-0.5251,-0.3949,-0.2844,-0.1848,-0.0911,0.0,0.0796,0.1609,0.2461,0.3379,0.4407,0.5626,0.7230,1.0],device=device)
w=torch.randn(1_000_000,device=device); scale=w.abs().max(); normalized=(w/scale).clamp(-1,1)
idx=(normalized[:,None]-nf4[None,:]).abs().argmin(1); nf4_dq=nf4[idx]*scale
_,_,uniform=symmetric_quantize(w.reshape(1000,1000),bits=4,group_size=1000); uniform=uniform.reshape(-1)
result=base_result(14,"numerical-model"); result.update({"bitsandbytes_installed":importlib.util.find_spec("bitsandbytes") is not None,
    "nf4_error":error_metrics(w,nf4_dq),"uniform_int4_error":error_metrics(w,uniform),
    "conclusion":"Codebook behavior was measured numerically; bitsandbytes native execution is claimed only when installed."})
''',
15: '''
import copy, importlib.util
available=importlib.util.find_spec("torchao") is not None; details={"torchao_installed":available}
if available:
    try:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        layer=torch.nn.Linear(4096,4096,bias=False,device=device,dtype=torch.bfloat16); candidate=copy.deepcopy(layer)
        x=torch.randn(8,4096,device=device,dtype=torch.bfloat16); ref=layer(x)
        quantize_(candidate,Int4WeightOnlyConfig(group_size=128)); out=candidate(x)
        details.update({"conversion":"success","module_type":type(candidate).__name__,"output_error":error_metrics(ref,out),
                        "bf16_timing":cuda_benchmark(lambda:layer(x),warmup=5,repeats=20),
                        "int4_timing":cuda_benchmark(lambda:candidate(x),warmup=5,repeats=20)})
    except Exception as exc:
        details.update({"conversion":"failed","error_type":type(exc).__name__,"error_message":str(exc)[:240]})
result=base_result(15,"native-backend" if details.get("conversion")=="success" else "compatibility-probe")
outcome=("TorchAO INT4 converted and executed; output error and latency were measured."
         if details.get("conversion")=="success" else
         "TorchAO was installed, but the native INT4 path did not execute; the dependency failure is preserved as a compatibility result.")
result.update({"torchao":details,"conclusion":outcome})
''',
16: '''
import importlib.util
w=torch.randn(512,1024,device=device); q,scales,dq=symmetric_quantize(w,bits=4,group_size=64)
codes=(q.to(torch.int16)&0xF).flatten(); packed=(codes[0::2]|(codes[1::2]<<4)).to(torch.uint8)
lo=(packed.to(torch.int16)&0xF); hi=((packed.to(torch.int16)>>4)&0xF); unpack=torch.stack([lo,hi],1).flatten()
unpack=torch.where(unpack>=8,unpack-16,unpack).to(torch.int8).reshape_as(q)
result=base_result(16,"pytorch-gpu"); result.update({"shape":list(w.shape),"group_size":64,"packed_bytes":packed.numel(),
    "codes_exact_after_unpack":bool(torch.equal(q,unpack)),"qdq_error":error_metrics(w,dq),
    "tensorrt_installed":importlib.util.find_spec("tensorrt") is not None,
    "conclusion":"Block Q/DQ and nibble packing were validated; TensorRT engine execution was not inferred from the reference path."})
''',
17: '''
import importlib.util, hashlib
w=torch.randn(256,256,device=device); _,scales,dq=symmetric_quantize(w,bits=4,group_size=64)
manifest={"base_revision":"example-frozen-revision","recipe":{"format":"INT4","group_size":64,"calibration":"synthetic-v1"},
          "handoff":{"modelopt":importlib.util.find_spec("modelopt") is not None,"tensorrt_llm":importlib.util.find_spec("tensorrt_llm") is not None},
          "scale_sha256":hashlib.sha256(scales.cpu().numpy().tobytes()).hexdigest(),"rollback_revision":"bf16-baseline-v1"}
required=("base_revision","recipe","handoff","scale_sha256","rollback_revision")
result=base_result(17,"compatibility-probe"); result.update({"manifest":manifest,"manifest_complete":all(k in manifest for k in required),
    "numerical_probe":error_metrics(w,dq),"conclusion":"The handoff contract was validated; absent packages remain explicit and no engine benchmark was claimed."})
''',
18: '''
import importlib.util
w=torch.randn(2048,2048,device=device,dtype=torch.bfloat16); _,_,dq=symmetric_quantize(w,bits=4,group_size=128); dq=dq.bfloat16()
rows=[]
for batch in (1,8,32):
    x=torch.randn(batch,2048,device=device,dtype=torch.bfloat16)
    rows.append({"batch":batch,"bf16":cuda_benchmark(lambda:x@w.t(),warmup=4,repeats=15),
                 "reference_w4_dequant":cuda_benchmark(lambda:x@dq.t(),warmup=4,repeats=15)})
installed=importlib.util.find_spec("vllm") is not None
result=base_result(18,"compatibility-probe"); result.update({"vllm_installed":installed,"pytorch_shape_warning":rows,
    "vllm_service_benchmark":"not_measured","conclusion":"PyTorch shape timing was measured separately; vLLM service performance requires an installed server and load test."})
''',
19: '''
batch,heads,seq,dim=1,8,4096,128; k=torch.randn(batch,heads,seq,dim,device=device); v=torch.randn_like(k); q=torch.randn(batch,heads,1,dim,device=device)
def qdq(t):
    scale=t.abs().amax(-1,keepdim=True).clamp_min(1e-8)/127; qt=torch.round(t/scale).clamp(-128,127).to(torch.int8); return qt,scale,qt.float()*scale
qk,sk,kd=qdq(k); qv,sv,vd=qdq(v)
ref=torch.softmax(q@k.transpose(-1,-2)/(dim**0.5),-1)@v; cand=torch.softmax(q@kd.transpose(-1,-2)/(dim**0.5),-1)@vd
bf16_bytes=2*(k.numel()+v.numel()); int8_bytes=qk.numel()+qv.numel()+sk.numel()*sk.element_size()+sv.numel()*sv.element_size()
result=base_result(19,"pytorch-gpu"); result.update({"shape":{"batch":batch,"heads":heads,"sequence":seq,"head_dim":dim},
    "bf16_bytes":bf16_bytes,"int8_plus_scale_bytes":int8_bytes,"memory_reduction_pct":round((1-int8_bytes/bf16_bytes)*100,4),
    "attention_output_error":error_metrics(ref,cand),"conclusion":"INT8 cache reduced storage in this reference while introducing measurable attention-output error."})
''',
20: '''
import importlib.util
n=1024; a=torch.randn(n,n,device=device,dtype=torch.bfloat16); b=torch.randn(n,n,device=device,dtype=torch.bfloat16); ref=(a@b).float()
probe={"torch_float8_dtype":hasattr(torch,"float8_e4m3fn"),"transformer_engine_installed":importlib.util.find_spec("transformer_engine") is not None}
if probe["torch_float8_dtype"]:
    try:
        a8=a.to(torch.float8_e4m3fn); b8=b.to(torch.float8_e4m3fn)
        one=torch.tensor(1.0,device=device)
        def fp8_scaled_mm():
            return torch._scaled_mm(a8,b8,scale_a=one,scale_b=one,out_dtype=torch.bfloat16)
        out=fp8_scaled_mm().float()
        probe.update({"fp8_gemm":"success","api":"torch._scaled_mm","fp8_error":error_metrics(ref,out),
                      "fp8_timing":cuda_benchmark(fp8_scaled_mm,warmup=5,repeats=15)})
    except Exception as exc: probe.update({"fp8_gemm":"failed","error_type":type(exc).__name__,"error_message":str(exc)[:240]})
probe["nvfp4_backend"]="not_measured"
result=base_result(20,"pytorch-gpu" if probe.get("fp8_gemm")=="success" else "compatibility-probe"); result.update({"probe":probe,
    "conclusion":"Framework-level FP8 was tested independently; NVFP4 requires a supported library recipe and operator evidence."})
''',
21: '''
conv=torch.nn.Conv2d(3,64,kernel_size=16,stride=16,bias=False,device=device); w=conv.weight.detach(); _,_,dq=symmetric_quantize(w.reshape(64,-1),bits=4,group_size=192); dq=dq.reshape_as(w)
normal=torch.randn(8,3,224,224,device=device); contrast=normal.clone(); contrast[:,:,::16,::16]*=20
def project(x,weight): return torch.nn.functional.conv2d(x,weight,stride=16)
rows={}
for name,x in {"normal":normal,"high_contrast":contrast}.items(): rows[name]=error_metrics(project(x,w),project(x,dq))
result=base_result(21,"pytorch-gpu"); result.update({"patch_projection":{"weight_shape":list(w.shape),"group_size":192},"domain_errors":rows,
    "conclusion":"Patch-projection error changed with image distribution; no full VLM quality conclusion was made."})
''',
22: '''
import hashlib, tempfile
w=torch.randn(256,512,device=device); q,scales,_=symmetric_quantize(w,bits=4,group_size=64); payload=q.cpu().numpy().tobytes()+scales.cpu().numpy().tobytes()
with tempfile.NamedTemporaryFile() as f:
    f.write(payload); f.flush(); digest=hashlib.sha256(Path(f.name).read_bytes()).hexdigest(); size=Path(f.name).stat().st_size
manifest={"schema":1,"format":"reference-int4","group_size":64,"shape":list(w.shape),"sha256":digest,"bytes":size,
          "base_revision":"example-frozen-revision","runtime":"pytorch-reference","rollback":"bf16-baseline-v1"}
required={"schema","format","group_size","shape","sha256","bytes","base_revision","runtime","rollback"}
result=base_result(22,"pytorch-gpu"); result.update({"manifest":manifest,"manifest_complete":required.issubset(manifest),
    "temporary_payload_deleted_after_check":True,"conclusion":"A small packaging contract and checksum were validated without publishing checkpoint data."})
''',
23: '''
torch.manual_seed(7); vocab,hidden,tokens=256,128,4096; h=torch.randn(tokens,hidden,device=device); w=torch.randn(vocab,hidden,device=device); targets=torch.randint(0,vocab,(tokens,),device=device)
base=h@w.t(); _,_,dq=symmetric_quantize(w,bits=4,group_size=64); cand=h@dq.t()
base_loss=torch.nn.functional.cross_entropy(base,targets); cand_loss=torch.nn.functional.cross_entropy(cand,targets)
slices={"first_half":slice(0,tokens//2),"second_half":slice(tokens//2,None)}; slice_rows={}
for name,s in slices.items(): slice_rows[name]={"top1_agreement":round((base[s].argmax(-1)==cand[s].argmax(-1)).float().mean().item(),6)}
result=base_result(23,"pytorch-gpu"); result.update({"synthetic_probe":{"tokens":tokens,"vocab":vocab,"baseline_loss":round(base_loss.item(),7),
    "candidate_loss":round(cand_loss.item(),7),"baseline_perplexity":round(base_loss.exp().item(),5),"candidate_perplexity":round(cand_loss.exp().item(),5),
    "top1_agreement":round((base.argmax(-1)==cand.argmax(-1)).float().mean().item(),6),"slices":slice_rows},
    "conclusion":"Multiple frozen metrics exposed the synthetic INT4 regression; they are not scores for a named language model."})
''',
24: '''
model=torch.nn.Sequential(torch.nn.Linear(2048,4096,bias=False),torch.nn.GELU(),torch.nn.Linear(4096,2048,bias=False)).to(device).bfloat16(); rows=[]
for batch in (1,8,32,128):
    x=torch.randn(batch,2048,device=device,dtype=torch.bfloat16); torch.cuda.reset_peak_memory_stats(); timing=cuda_benchmark(lambda:model(x),warmup=5,repeats=20)
    rows.append({"batch":batch,"timing":timing,"examples_per_second":round(batch/(timing["median_ms"]/1000),2),
                 "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,3)})
result=base_result(24,"pytorch-gpu"); result.update({"operator_workload":rows,
    "conclusion":"Batching changed throughput, latency, and memory in different directions; no service queueing was modeled."})
''',
25: '''
w=torch.randn(1024,1024,device=device); _,_,dq=symmetric_quantize(w,bits=4,group_size=128); rows=[]
cases={"ordinary":torch.randn(32,1024,device=device),"small_batch":torch.randn(1,1024,device=device),
       "activation_outliers":torch.randn(32,1024,device=device),"shifted_domain":torch.randn(32,1024,device=device)*3+2}
cases["activation_outliers"][:,::73]*=30
for name,x in cases.items(): rows.append({"case":name,"output_error":error_metrics(x@w.t(),x@dq.t()),
    "bf16_timing":cuda_benchmark(lambda:x@w.t(),warmup=3,repeats=12),"reference_w4_timing":cuda_benchmark(lambda:x@dq.t(),warmup=3,repeats=12)})
result=base_result(25,"pytorch-gpu"); result.update({"failure_matrix":rows,
    "conclusion":"Condition-specific tests exposed reversals that an aggregate average could conceal."})
''',
26: '''
dims=[256]*7; weights=[torch.randn(dims[i+1],dims[i],device=device)*0.05 for i in range(6)]; x=torch.randn(512,256,device=device)
def forward(ws):
    y=x
    for i,w in enumerate(ws): y=y@w.t(); y=torch.nn.functional.gelu(y) if i<5 else y
    return y
ref=forward(weights); q4=[]; q8=[]
for w in weights:
    q4.append(symmetric_quantize(w,bits=4,group_size=64)[2]); q8.append(symmetric_quantize(w,bits=8,group_size=64)[2])
sensitivity=[]
for i in range(6):
    ws=list(weights); ws[i]=q4[i]; sensitivity.append({"layer":i,"rmse":error_metrics(ref,forward(ws))["rmse"]})
fallback={r["layer"] for r in sorted(sensitivity,key=lambda z:z["rmse"],reverse=True)[:2]}; mixed=[q8[i] if i in fallback else q4[i] for i in range(6)]
bits=sum((8 if i in fallback else 4)*weights[i].numel() for i in range(6))/sum(w.numel() for w in weights)
result=base_result(26,"pytorch-gpu"); result.update({"layer_sensitivity":sensitivity,"int8_fallback_layers":sorted(fallback),
    "average_weight_bits":round(bits,3),"assembled_output_error":error_metrics(ref,forward(mixed)),
    "conclusion":"A budgeted mixed-bit candidate spent extra precision on measured sensitive layers and was re-evaluated end to end."})
''',
27: '''
w=torch.randn(2048,2048,device=device,dtype=torch.bfloat16); x=torch.randn(16,2048,device=device,dtype=torch.bfloat16); ref=x@w.t(); _,_,dq=symmetric_quantize(w,bits=4,group_size=128); dq=dq.bfloat16(); cand=x@dq.t()
base_t=cuda_benchmark(lambda:x@w.t(),warmup=4,repeats=15); cand_t=cuda_benchmark(lambda:x@dq.t(),warmup=4,repeats=15); err=error_metrics(ref,cand)
gates={"rmse_lte_0_5":err["rmse"]<=0.5,"latency_regression_lte_10pct":cand_t["median_ms"]<=base_t["median_ms"]*1.10}
decision="promote_to_canary" if all(gates.values()) else "rollback"
manifest={"candidate":"reference-int4-v1","baseline":"bf16-v1","decision":decision,"gates":gates,"rollback_target":"bf16-v1"}
result=base_result(27,"capacity-model"); result.update({"baseline_timing":base_t,"candidate_timing":cand_t,"output_error":err,"release_manifest":manifest,
    "conclusion":"Frozen synthetic gates produced a deterministic release or rollback decision; no live service canary was claimed."})
''',
28: '''
free,total=torch.cuda.mem_get_info(); cfg={"parameters":70_000_000_000,"layers":80,"kv_heads":8,"head_dim":128,"context":8192}
def plan(weight_bytes,cache_bytes):
    weights=cfg["parameters"]*weight_bytes; reserve=total*0.10; usable=max(0,total-reserve-weights); kv=2*cfg["layers"]*cfg["context"]*cfg["kv_heads"]*cfg["head_dim"]*cache_bytes
    return {"weight_gib":round(weights/2**30,3),"kv_per_request_gib":round(kv/2**30,3),"projected_requests":max(0,int(usable//kv)),"single_gpu_weight_fit":weights<total-reserve}
plans={"bf16_weights_bf16_kv":plan(2,2),"int4_ideal_weights_bf16_kv":plan(0.5,2),"int4_ideal_weights_int8_kv":plan(0.5,1)}
result=base_result(28,"capacity-model"); result.update({"live_memory":{"free_gib":round(free/2**30,3),"total_gib":round(total/2**30,3)},"assumptions":cfg,"plans":plans,
    "conclusion":"Arithmetic capacity projections used live GPU memory but did not claim that a 70B engine loaded or met latency SLOs."})
''',
29: '''
m,k,n=32,4096,4096; x=torch.randn(m,k,device=device,dtype=torch.bfloat16); w=torch.randn(n,k,device=device,dtype=torch.bfloat16); q,scales,_=symmetric_quantize(w,bits=4,group_size=128)
codes=(q.to(torch.int16)&0xF).flatten(); packed=(codes[0::2]|(codes[1::2]<<4)).to(torch.uint8)
def composed():
    lo=(packed.to(torch.int16)&15); hi=((packed.to(torch.int16)>>4)&15); u=torch.stack([lo,hi],1).flatten(); u=torch.where(u>=8,u-16,u).reshape(n,k).float()
    dq=(u.reshape(n,k//128,128)*scales[...,None]).reshape(n,k).bfloat16(); return x@dq.t()
packed_t=cuda_benchmark(composed,warmup=3,repeats=10); bf16_t=cuda_benchmark(lambda:x@w.t(),warmup=5,repeats=15)
result=base_result(29,"pytorch-gpu"); result.update({"shape_mkn":[m,k,n],"packed_bytes":packed.numel(),"bf16_timing":bf16_t,"composed_unpack_dequant_matmul":packed_t,
    "implementation":"composed PyTorch reference, not fused CUTLASS","conclusion":"The composed reference exposed integration overhead; it is a semantic baseline, not a custom-kernel performance claim."})
''',
30: '''
free,total=torch.cuda.mem_get_info(); params=70_000_000_000; ideal_int4=params*0.5; reserve=total*0.1; fit=ideal_int4 < total-reserve
w=torch.randn(1024,1024,device=device); x=torch.randn(64,1024,device=device); ref=x@w.t(); q4=symmetric_quantize(w,bits=4,group_size=128)[2]; q8=symmetric_quantize(w,bits=8,group_size=128)[2]
sensitive=torch.arange(0,1024,64,device=device); mixed=q4.clone(); mixed[:,sensitive]=q8[:,sensitive]; err=error_metrics(ref,x@mixed.t())
gates={"single_gpu_ideal_weight_fit":fit,"backend_engine_built":False,"quality_suite_passed":False,"service_slo_passed":False,
       "toy_mixed_bit_rmse_lte_2":err["rmse"]<=2.0,"rollback_artifact_defined":True}
decision="not_ready_for_service" if not all(gates.values()) else "ready_for_canary"
result=base_result(30,"capacity-model"); result.update({"live_gpu_total_gib":round(total/2**30,3),"ideal_int4_weight_gib":round(ideal_int4/2**30,3),
    "toy_mixed_bit_error":err,"deployment_gates":gates,"decision":decision,
    "conclusion":"The gate matrix kept unexecuted 70B engine, quality, and service tests explicit; arithmetic compression alone was insufficient."})
'''
}


def notebook_for(lesson: dict) -> dict:
    no = lesson["no"]
    slug = lesson["slug"]
    theory = THEORY[no]
    common = dedent(f'''\
        from pathlib import Path
        import json
        import sys
        import torch

        chapter_rel = Path("chapters/01-mixed-precision-int4")
        repo_root = next(
            p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / chapter_rel / "support" / "lab_common.py").exists()
        )
        sys.path.insert(0, str(repo_root / chapter_rel / "support"))
        from lab_common import (base_result, cuda_benchmark, environment_record,
                                error_metrics, require_cuda, save_result,
                                symmetric_quantize)

        lesson_dir = repo_root / chapter_rel / "{no:02d}-{slug}"
        device = require_cuda()
        torch.manual_seed(2026 + {no})
        environment = environment_record()
        print(json.dumps(environment, indent=2))
    ''')
    save = dedent('''\
        artifact_path = save_result(result, lesson_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Saved: artifacts/rtx5090-result.json")
    ''')
    def md(cell_id: str, source: str) -> dict:
        return {"id":cell_id,"cell_type":"markdown","metadata":{},"source":source}
    def code(cell_id: str, source: str) -> dict:
        return {"id":cell_id,"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}
    cells = [
        md(f"l{no:02d}-title", f"# Lesson {no:02d} Lab — {lesson['title']}\n\n**Puzzle:** {lesson['puzzle']}\n\nThe saved outputs were generated by executing every code cell on the recorded RTX 5090. Run all cells to regenerate the evidence on your own CUDA GPU."),
        md(f"l{no:02d}-predict", "## 0. Predict before running\n\nWrite down: (1) the expected direction, (2) the mechanism, (3) the observation that would reverse your prediction, and (4) the evidence level required for the claim."),
        md(f"l{no:02d}-theory-objects", f"## 1. Theory — objects and data flow\n\n{theory['objects']}\n\n### Core mechanism\n\n{theory['mechanism']}"),
        code(f"l{no:02d}-environment", common),
        md(f"l{no:02d}-theory-map", f"## 2. Connect theory to the experiment\n\n### Engineering trade-off\n\n{theory['tradeoff']}\n\n### What this code tests\n\n{theory['code']}\n\n**Experiment:** {lesson['experiment']}\n\n**Declared evidence label:** `{lesson['label']}`. Check that the shapes, controlled variables, and units match the theoretical question before executing."),
        code(f"l{no:02d}-experiment", dedent(EXPERIMENTS[no]).strip()+"\n"),
        md(f"l{no:02d}-inspect", f"## 3. Inspect the evidence\n\n{lesson['inspect']}\n\n### Acceptance and rollback gate\n\n{theory['gate']}"),
        code(f"l{no:02d}-artifact", save),
        md(f"l{no:02d}-explain", f"## 4. Explain the result\n\n{lesson['conclusion']}\n\nRelate the measured fields back to the mechanism above. Treat the checked-in result as one hardware/software observation, not a universal ranking. The complete derivation, evidence boundary, and primary references are in [`README.md`](README.md)."),
    ]
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}


def readme_for(lesson: dict) -> str:
    no = lesson["no"]
    theory = THEORY[no]
    concepts = "\n".join(f"- {x}" for x in lesson["concepts"])
    refs = "\n".join(f"- [{name}]({url})" for name,url in lesson["refs"])
    return f"""# Lesson {no:02d} — {lesson['title']}

> **Puzzle:** {lesson['puzzle']}

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 artifact](artifacts/rtx5090-result.json)

## Predict

Before opening the saved result, write a falsifiable prediction:

1. Which measured quantity should change, and in which direction?
2. What GPU, numerical, or systems mechanism should cause that change?
3. Which observation would make you keep the baseline or add a fallback?
4. What level of evidence is needed: numerical model, PyTorch GPU path, or named native backend?

## 1. Start from the concrete objects

{theory['objects']}

Quick mental model:

{concepts}

This object-first view prevents storage format, compute format, accumulation,
operator dispatch, latency, memory, and model quality from being treated as one
interchangeable idea.

## 2. Core mechanism

{theory['mechanism']}

The formula or invariant above is the bridge between the theory and the code.
If the implementation does not preserve or test it, the experiment is answering
a different question.

## 3. Engineering trade-off and failure mode

{theory['tradeoff']}

The most important failure mode for this lesson is therefore not simply "the
number is worse." It is a mismatch between the claimed mechanism and the object,
shape, distribution, or backend that actually ran.

## 4. From theory to the notebook

{lesson['experiment']}

{theory['code']}

| Theory question | Notebook evidence |
|---|---|
| What object or tensor changes? | Explicit shapes, dtypes, and configuration |
| What mechanism should cause the effect? | Controlled baseline/candidate code |
| Did the expected path run? | Evidence label and compatibility/operator fields |
| What changed numerically or operationally? | Error, memory, or repeated timing fields |
| When should we stop or roll back? | The acceptance gate below |

The notebook records a sanitized environment and deterministic seed. GPU
timings use CUDA events with synchronization, warm-up iterations, and repeated
samples. The declared evidence label is **`{lesson['label']}`**.

## 5. Inspect, accept, or roll back

{lesson['inspect']}

{theory['gate']}

Open [`lab.ipynb`](lab.ipynb) for the executable derivation and retained output.
The compact [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) is
designed for diffs and automated checks.

## Explain

{lesson['conclusion']}

A useful conclusion states what changed, what did not change, and which backend,
shape, or workload could reverse the result. It never upgrades a compatibility
probe or numerical model into a production-kernel claim.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/{no:02d}-{lesson['slug']}/lab.ipynb
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

{refs}
"""


def chapter_readme() -> str:
    rows = ['| # | Lesson | Evidence | Status |','|---:|---|---|---|',
            '| 01 | [Precision formats: FP32, TF32, FP16, BF16, FP8, INT8, and INT4](01-precision-formats/README.md) | native TorchAO + model benchmark | Published |']
    for x in LESSONS:
        rows.append(f'| {x["no"]:02d} | [{x["title"]}]({x["no"]:02d}-{x["slug"]}/README.md) | `{x["label"]}` | Published |')
    table='\n'.join(rows)
    return f"""# Chapter 01 — Mixed Precision and INT4 Quantization

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

This retains the useful conceptual path of the study curriculum while replacing
generic prose with lesson-specific formulas, tensor objects, failure modes, and
experiments. A numerical model can explain a mechanism; it cannot stand in for
TensorRT, vLLM, CUTLASS, bitsandbytes, ModelOpt, or Transformer Engine execution.

## Lessons

{table}

## Chapter environment policy

Every GPU experiment reports the GPU, compute capability, PyTorch and CUDA
runtime, shapes, warm-up policy, repetitions, units, and a bounded conclusion.
The checked-in reference outputs come from an NVIDIA GeForce RTX 5090, but they
are not universal performance rankings.

Run all lightweight labs from the repository root with:

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 01
python3 scripts/validate_chapter.py 01
```

Lesson 01 is a full Qwen/TorchAO comparison and may download a model. Lessons
02–30 use synthetic tensors so readers can isolate each mechanism without
downloading 70B-class checkpoints.
"""


def main() -> None:
    for lesson in LESSONS:
        directory = CHAPTER / f"{lesson['no']:02d}-{lesson['slug']}"
        (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(readme_for(lesson), encoding="utf-8")
        (directory / "lab.ipynb").write_text(json.dumps(notebook_for(lesson), ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    # The chapter map is hand-maintained so its 30-lesson theory route is not
    # replaced by the compact bootstrap table above.
    from enrich_chapter01_delivery import main as enrich_delivery

    enrich_delivery()
    print(f"Built and enriched {len(LESSONS)} lesson notes and notebooks under {CHAPTER}")


if __name__ == "__main__":
    main()
