<p align="center">
  <img src="./assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>

<p align="center">
  <a href="#what-is-ai-infra-puzzles"><strong>Overview</strong></a> ·
  <a href="#start-with-chapter-01"><strong>Start Here</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#how-this-repository-works"><strong>How It Works</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Original_Work-Linnea_Cai-8A2BE2" alt="Original work by Linnea Cai">
  <img src="https://img.shields.io/badge/RTX_5090-verified-76B900" alt="Verified on RTX 5090">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Chapter_01-30_Labs-6C63FF" alt="Chapter 01 has 30 labs">
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
Chapter 01 currently has all 30 lessons published and executed on the recorded
RTX 5090 environment.

```text
ai-infra-puzzles/
├── README.md
├── ATTRIBUTION.md          # Authorship and third-party source policy
├── requirements-notebook.txt
├── assets/branding/        # Project logo
├── chapters/
│   └── 01-mixed-precision-int4/
│       ├── README.md       # 30-lesson chapter map
│       ├── support/        # Shared timing and result helpers
│       ├── 01-precision-formats/
│       │   ├── README.md   # Theory notes
│       │   ├── lab.ipynb   # Code with retained outputs
│       │   ├── artifacts/  # Small public evidence
│       │   └── support/    # Full-model runner
│       └── 02-... through 30-.../
│           ├── README.md
│           ├── lab.ipynb
│           └── artifacts/
└── scripts/                # Repository-wide public-safety checks
```

## Evidence, not marketing

The checked-in result is deliberately small and auditable. It records the model
revision, environment, measurement protocol, memory accounting, latency samples,
quality probe, and operator evidence. Large model files, full logits, private
paths, credentials, and raw profiler traces are never committed.

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
