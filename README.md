<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

<p align="center">
  <a href="#what-is-ai-infra-puzzles"><strong>Overview</strong></a> ·
  <a href="#start-with-chapter-01"><strong>Start Here</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#how-this-repository-works"><strong>How It Works</strong></a> ·
  <a href="README_ZH.md"><strong>中文教程</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Original_Work-Linnea_Cai-8A2BE2" alt="Original work by Linnea Cai">
  <img src="https://img.shields.io/badge/RTX_5090-verified-76B900" alt="Verified on RTX 5090">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Chapter_01-30_Labs-6C63FF" alt="Chapter 01 has 30 labs">
  <img src="https://img.shields.io/badge/Chapter_02-28_Labs-00A6A6" alt="Chapter 02 has 28 labs">
  <img src="https://img.shields.io/badge/Chapter_03-30_Labs-F59E0B" alt="Chapter 03 has 30 labs">
  <img src="https://img.shields.io/badge/Chapter_04-17_Labs-2563EB" alt="Chapter 04 has 17 labs">
</p>

## What is AI Infra Puzzles?

**AI Infra Puzzles** is a growing, learn-by-doing course repository for CUDA
kernel optimization and LLM inference. It turns technical notes into runnable
experiments that connect precision formats, GPU operators, memory traffic, and
end-to-end model behavior.

Each puzzle starts with a plausible systems claim—such as _"INT4 must be faster
because it uses fewer bits"_—and turns it into an experiment. You predict the
outcome, run the code, inspect memory and operator evidence, and decide what the
optimization actually achieved.

The goal is not to collect isolated benchmark numbers. It is to build the habit
of asking four questions:

1. What is stored?
2. Which kernel runs?
3. What becomes faster or smaller?
4. What evidence would change the conclusion?

## Start with Chapter 01

### [Mixed Precision and INT4 Quantization](chapters/01-mixed-precision-int4/README.md)

This chapter is a complete 30-lesson learning path. Every lesson contains an
original theory `README.md`, an executable `lab.ipynb` with retained RTX 5090
outputs, and compact machine-readable evidence. The path moves from formats and
AMP through PTQ algorithms and production backends to service benchmarking,
custom-kernel boundaries, and a gated 70B deployment plan.

The lesson notes include the derivation, controlled experiment, selected
measured values, interpretation, failure modes, and next experiment—not only a
link to notebook output. The notebooks keep the theory next to the code so a
reader can predict, run, inspect, and explain without switching documents at
every step. High-leverage mechanisms also include a dedicated Mermaid diagram
and a four-step reasoning walkthrough.

See the [full 30-lesson chapter map](chapters/01-mixed-precision-int4/README.md).

#### [Lesson 01 — Precision Formats: INT4, Smaller but Faster?](chapters/01-mixed-precision-int4/01-precision-formats/README.md)

> A 4-bit model should use less memory. Does that also make Prefill and Decode
> faster than BF16?

The first puzzle compares BF16 with TorchAO weight-only INT4 on a Qwen2.5-1.5B
model and an NVIDIA RTX 5090. It follows the quantized modules into the profiler
instead of stopping at a dtype label.

| What we measured | BF16 | INT4 | Observation |
|---|---:|---:|---|
| CUDA allocated after load | 2.876 GiB | 1.336 GiB | 53.54% less |
| Prefill, 512 tokens | 11.318 ms | 43.963 ms | INT4 latency was 288.42% higher |
| Approx. Decode throughput | 101.077 tok/s | 96.966 tok/s | INT4 was 4.07% lower |

The surprising result is the puzzle: **smaller storage did not produce faster
inference for these shapes and this backend.** Read the
[full walkthrough](chapters/01-mixed-precision-int4/01-precision-formats/README.md)
or open the
[executed notebook](chapters/01-mixed-precision-int4/01-precision-formats/lab.ipynb)
to see why.

## Continue with Chapter 02

### [Sparsity and Structured Pruning](chapters/02-sparsity-structured-pruning/README.md)

Chapter 02 is a 28-lesson path from pruning objectives and mask semantics to
physical channel deletion, dependency graphs, 2:4 constraints, framework
lifecycles, CNN/Transformer/LLM pruning, honest acceleration benchmarks, and
edge-versus-server deployment decisions.

Every lesson distinguishes four different claims: values became zero, storage
became smaller, tensor shapes changed, and the runtime became faster. The
checked-in labs were executed on an RTX 5090 with PyTorch 2.12 and CUDA 13.0.
Optional toolchains are reported as compatibility probes when unavailable;
missing native execution is never presented as measured acceleration. Each
lesson now visualizes its pruning, dependency, export, or deployment path and
then explains that path in four concrete steps.

See the [complete 28-lesson map](chapters/02-sparsity-structured-pruning/README.md)
or start with
[Lesson 01 — Pruning Objectives](chapters/02-sparsity-structured-pruning/01-pruning-objectives/README.md).

## Continue with Chapter 03

### [vLLM Inference and Serving](chapters/03-vllm-inference-serving/README.md)

Chapter 03 follows a request from Prefill and KV-cache allocation through continuous
batching, offline and OpenAI-compatible APIs, prefix caching, FP8 KV, speculative
decoding, structured outputs, benchmarking, metrics, containers, Kubernetes, capacity,
security, and a reversible production gate.

Its 30 labs pin vLLM 0.27.1 and retain RTX 5090 evidence. Native vLLM execution,
compatibility probes, scheduler simulations, and capacity models use different evidence
labels. Single-GPU results never stand in for multi-node, Kubernetes, or disaggregated
Prefill/Decode measurements.

See the [complete 30-lesson map](chapters/03-vllm-inference-serving/README.md) or begin
with [Lesson 01 — The Inference Service Bottleneck](chapters/03-vllm-inference-serving/01-inference-service-bottleneck/README.md).

## Continue with Chapter 04

### [GPU Hardware Foundations: From CMOS to Attention](chapters/04-gpu-hardware-foundations/README.md)

Chapter 04 connects transistor and memory-cell intuition to CUDA performance. Its 17
lessons trace data through external memory, controllers, L2 slices, the on-chip network,
SM storage, and execution units before measuring Roofline behavior, attention IO,
coalescing, atomics, reductions, and asynchronous timing.

The chapter preserves the complete visual set from Linnea Cai's GPU hardware study notes
and pairs every theory topic with executable code. Numerical circuit and queue models are
kept separate from native PyTorch/CUDA measurements on the RTX 5090.

See the [complete 17-lesson map](chapters/04-gpu-hardware-foundations/README.md) or begin
with [Lesson 01 — CMOS Switching, State, and Dynamic Power](chapters/04-gpu-hardware-foundations/01-cmos-switching-dynamic-power/README.md).

## Quick Start

### Run the executed GPU notebook

The notebook performs fresh BF16 and INT4 measurements when you use **Run All**.
It requires a compatible NVIDIA GPU and downloads the default model unless you
configure a local checkpoint:

```bash
git clone https://github.com/LinglingCai0314/ai-infra-puzzles.git
cd ai-infra-puzzles

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
pip install -r requirements.txt
jupyter lab chapters/01-mixed-precision-int4/01-precision-formats/lab.ipynb
```

The outputs checked into `lab.ipynb` were produced by executing every cell on
an RTX 5090. Your own run will regenerate them for your GPU.

To execute the lightweight mechanism labs for Lessons 02–30 in order:

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 01 --start 2 --end 30
python3 scripts/validate_chapter.py 01
python3 scripts/audit_chapter01_delivery.py
```

To execute and validate all Chapter 02 pruning labs:

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 02 --start 1 --end 28
python3 scripts/build_chapter02_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 02
python3 scripts/audit_chapter02_delivery.py
```

To execute and validate Chapter 03 in the pinned vLLM environment:

```bash
export CH3_MODEL=/path/to/Qwen2.5-1.5B-Instruct
python3 scripts/execute_chapter_notebooks.py --chapter 03 --start 1 --end 30
python3 scripts/build_chapter03_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 03
python3 scripts/audit_chapter03_delivery.py
```

To execute and validate the GPU hardware foundations labs:

```bash
python3 -m pip install -r requirements-gpu-foundations.txt
python3 scripts/execute_chapter_notebooks.py --chapter 04 --start 1 --end 17
python3 scripts/build_chapter04_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 04
python3 scripts/audit_chapter04_delivery.py
```

### Command-line alternative

#### Prerequisites

- An NVIDIA GPU
- A PyTorch build compatible with that GPU
- Python 3.12
- Enough storage to download `Qwen/Qwen2.5-1.5B-Instruct`

The published measurements were collected on an RTX 5090. Other GPUs, software
versions, batch sizes, and matrix shapes may produce different results.

```bash
pip install -r requirements.txt
./chapters/01-mixed-precision-int4/01-precision-formats/support/run.sh
```

To use an existing local checkpoint:

```bash
CH1_MODEL=/path/to/model CH1_LOCAL_FILES_ONLY=1 \
  ./chapters/01-mixed-precision-int4/01-precision-formats/support/run.sh
```

Generated artifacts are written to the lesson's `outputs/` directory, which is
intentionally ignored by Git.

## How this repository works

Every released puzzle follows the same loop:

```text
Predict → Run → Inspect → Explain
```

- **Predict:** Commit to an expectation before seeing the result.
- **Run:** Use one reproducible script for the baseline and candidate.
- **Inspect:** Check memory, latency, output behavior, and the dispatched kernel.
- **Explain:** State a bounded conclusion and the conditions under which it may
  reverse.

Only completed lessons with runnable code and inspectable evidence are linked.
Chapters 01, 02, 03, and 04 publish 30, 28, 30, and 17 labs respectively; all retain
outputs from their recorded RTX 5090 environments.

```text
ai-infra-puzzles/
├── README.md
├── ATTRIBUTION.md          # Authorship and third-party source policy
├── requirements-notebook.txt
├── assets/branding/        # Project logo
├── chapters/
│   ├── 01-mixed-precision-int4/
│   │   ├── README.md       # 30-lesson chapter map
│   │   ├── support/        # Shared timing and result helpers
│   │   ├── 01-precision-formats/
│   │   │   ├── README.md   # Theory notes
│   │   │   ├── lab.ipynb   # Code with retained outputs
│   │   │   ├── artifacts/  # Small public evidence
│   │   │   └── support/    # Full-model runner
│   │   └── 02-... through 30-.../
│   │       ├── README.md
│   │       ├── lab.ipynb
│   │       └── artifacts/
│   ├── 02-sparsity-structured-pruning/
│       ├── README.md       # 28-lesson chapter map
│       └── 01-... through 28-.../
│           ├── README.md
│           ├── lab.ipynb
│           └── artifacts/
│   ├── 03-vllm-inference-serving/
│       ├── README.md       # 30-lesson chapter map
│       └── 01-... through 30-.../
│           ├── README.md
│           ├── lab.ipynb
│           └── artifacts/
│   └── 04-gpu-hardware-foundations/
│       ├── README.md       # 17-lesson chapter map and visual atlas
│       ├── assets/         # Conceptual diagrams, interactive HTML, and printable PDFs
│       └── 01-... through 17-.../
│           ├── README.md
│           ├── lab.ipynb
│           └── artifacts/
└── scripts/                # Repository-wide public-safety checks
```

## Results and validation

See the structured
[RTX 5090 result](chapters/01-mixed-precision-int4/01-precision-formats/artifacts/rtx5090-result.json)
or run the
public-content check:

```bash
python3 scripts/check_public_safety.py .
```

## Contributing

Corrections, clearer explanations, and reproducibility reports on other GPUs are
welcome. When reporting a result, include the GPU, software versions, model
revision, batch size, sequence length, warm-up policy, and raw repeated samples.

## Acknowledgments

The learn-by-doing presentation is inspired by
[Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles) and
[GPU Puzzles](https://github.com/srush/GPU-Puzzles). AI Infra Puzzles uses an
original visual identity and focuses on evidence-driven LLM inference systems.

## Authorship and Attribution

Unless otherwise noted, the explanations, lesson organization, original
diagrams, benchmark design, notebooks, and repository-specific tutorial code in
this project are original work by **Linnea Cai
([@LinglingCai0314](https://github.com/LinglingCai0314))**.

This is an independent study and tutorial project. It does not reproduce or
redistribute the source course materials, and it is not affiliated with or
endorsed by the course provider, NVIDIA, PyTorch, Hugging Face, Modular, or
Qwen. Third-party concepts, libraries, model checkpoints, APIs, trademarks, and
adapted materials remain the property of their respective owners and are
credited where they are used. See [ATTRIBUTION.md](ATTRIBUTION.md) for the full
authorship boundary and source policy.

Copyright © 2026 Linnea Cai.

## License

No open-source license has been selected yet. The authorship statement above
does not grant permission to copy, modify, or redistribute the work. Until a
license file is added, the repository remains all rights reserved.
