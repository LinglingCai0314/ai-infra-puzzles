<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

<p align="center"> <a href="#what-is-ai-infra-puzzles"><strong>概览</strong></a> · <a href="#start-with-chapter-01"><strong>从这里开始</strong></a> · <a href="#quick-start"><strong>快速开始</strong></a> · <a href="#how-this-repository-works"><strong>它是如何工作的</strong></a> · <a href="README.md"><strong>English</strong></a> </p>

<p align="center"> <img src="https://img.shields.io/badge/Original_Work-Linnea_Cai-8A2BE2" alt="Original work by Linnea Cai"> <img src="https://img.shields.io/badge/RTX_5090-verified-76B900" alt="Verified on RTX 5090"> <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"> <img src="https://img.shields.io/badge/Chapter_01-30_Labs-6C63FF" alt="Chapter 01 has 30 labs"> <img src="https://img.shields.io/badge/Chapter_02-28_Labs-00A6A6" alt="Chapter 02 has 28 labs"> <img src="https://img.shields.io/badge/Chapter_03-30_Labs-F59E0B" alt="Chapter 03 has 30 labs"> </p>

## <a id="what-is-ai-infra-puzzles"></a>什么是 AI Infra Puzzles？

**AI Infra Puzzles**是一个持续扩展、强调动手实践的课程仓库，主题涵盖 CUDA kernel 优化与 LLM 推理。它把技术笔记变成可运行的实验，将精度格式、GPU operator、内存流量和端到端模型行为联系起来。

每个谜题都从一个看似合理的系统判断开始，例如 _“INT4 使用的 bit 更少，所以一定更快”_，再把这个判断转化为实验。你需要预测结果、运行代码、检查内存与 operator 证据，并判断这项优化究竟取得了什么效果。

目标不是收集彼此孤立的 benchmark 数字，而是养成反复追问以下四个问题的习惯：

1. 存储了什么？
2. 运行的是哪个 kernel？
3. 什么变得更快或更小？
4. 什么证据会改变结论？

## <a id="start-with-chapter-01"></a>从第 01 章开始

### [混合精度与 INT4 量化](chapters-zh/01-mixed-precision-int4/README.md)

本章提供一条完整的 30 课学习路径。每课都包含原创理论 `README.md`、保留 RTX 5090 输出的可执行 `lab.ipynb`，以及精简的机器可读证据。内容从数值格式和 AMP 出发，经过 PTQ 算法与生产 backend，最后延伸到服务 benchmark、自定义 kernel 的适用边界，以及带 gate 的 70B 部署方案。

课程笔记不只是链接到 Notebook 输出，还包括推导、受控实验、选定的实测值、结果解释、失效模式和下一步实验。Notebook 把理论放在代码旁边，让读者不必频繁切换文档，就能完成预测、运行、检查和解释。关键机制还配有专门的 Mermaid 图和四步推理导读。

查看[完整的 30 课章节导航](chapters-zh/01-mixed-precision-int4/README.md)。

#### [第 01 课 — 精度格式：INT4，更小就会更快吗？](chapters-zh/01-mixed-precision-int4/01-precision-formats/README.md)

> 4-bit 模型理应占用更少的内存。这是否也意味着它的 Prefill 和 Decode
> 会比 BF16 更快？

第一个谜题在 Qwen2.5-1.5B 模型和 NVIDIA RTX 5090 上，对比 BF16 与 TorchAO weight-only INT4。实验会沿着量化模块继续追踪到 profiler，而不是停留在 dtype 标签上。

| 我们测量的内容 | BF16 | INT4 | 观察 |
|---|---:|---:|---|
| 加载后的 CUDA allocated memory | 2.876 GiB | 1.336 GiB | 减少 53.54% |
| Prefill，512 tokens | 11.318 ms | 43.963 ms | INT4 latency 高 288.42% |
| 近似 Decode throughput | 101.077 tok/s | 96.966 tok/s | INT4 低 4.07% |

令人意外的结果正是本课的谜题：**在这些 shape 和这个 backend 上，更小的存储并没有带来更快的推理。**阅读[完整导读](chapters-zh/01-mixed-precision-int4/01-precision-formats/README.md)，或打开[已执行的 Notebook](chapters/01-mixed-precision-int4/01-precision-formats/lab.ipynb)查看原因。

## 继续学习第 02 章

### [稀疏性与结构化剪枝](chapters-zh/02-sparsity-structured-pruning/README.md)

第 02 章是一条 28 课学习路径，从剪枝目标和 mask 语义讲到物理 channel 删除、依赖图、2:4 约束、框架生命周期、CNN/Transformer/LLM 剪枝、可信的加速 benchmark，以及 edge 与 server 的部署决策。

每课都会区分四种不同的主张：数值变成了零、存储变小、tensor shape 改变，以及 runtime 变快。仓库内的实验使用 RTX 5090、PyTorch 2.12 和 CUDA 13.0 执行。可选工具链不可用时，只会记录为 compatibility probe；没有实际执行的 native 路径绝不会被写成实测加速。每课还会把剪枝、依赖、导出或部署路径画出来，再用四个具体步骤解释。

查看[完整的 28 课章节导航](chapters-zh/02-sparsity-structured-pruning/README.md)，或从[第 01 课 — 剪枝目标](chapters-zh/02-sparsity-structured-pruning/01-pruning-objectives/README.md)开始。

## 继续学习第 03 章

### [vLLM 推理与服务](chapters-zh/03-vllm-inference-serving/README.md)

第 03 章沿着一条请求链路展开：从 Prefill 和 KV-cache 分配，到 continuous batching、离线与 OpenAI-compatible API、prefix caching、FP8 KV、speculative decoding、structured outputs、benchmark、metrics、容器、Kubernetes、容量、安全，以及可逆的 production gate。

本章的 30 个实验固定使用 vLLM 0.27.1，并保留 RTX 5090 证据。native vLLM 执行、compatibility probe、scheduler simulation 和 capacity model 使用不同的证据标签。单 GPU 结果绝不能替代多节点、Kubernetes 或解耦式 Prefill/Decode 的实测结果。

查看[完整的 30 课章节导航](chapters-zh/03-vllm-inference-serving/README.md)，或从[第 01 课 — 推理服务瓶颈](chapters-zh/03-vllm-inference-serving/01-inference-service-bottleneck/README.md)开始。

## <a id="quick-start"></a>快速开始

### 运行包含实测输出的 GPU Notebook

使用 **Run All** 时，Notebook 会重新执行 BF16 和 INT4 测量。你需要一块兼容的 NVIDIA GPU；如果未配置本地 checkpoint，还会下载默认模型。

```bash
git clone https://github.com/LinglingCai0314/ai-infra-puzzles.git
cd ai-infra-puzzles

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
pip install -r requirements.txt
jupyter lab chapters/01-mixed-precision-int4/01-precision-formats/lab.ipynb
```

`lab.ipynb` 中保留的输出来自 RTX 5090 上的逐单元执行。你再次运行时，会针对自己的 GPU 重新生成这些输出。

要执行第 02–30 课的轻量级机制实验，请依次运行：

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 01 --start 2 --end 30
python3 scripts/validate_chapter.py 01
python3 scripts/audit_chapter01_delivery.py
```

执行并验证第 02 章的全部剪枝实验：

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 02 --start 1 --end 28
python3 scripts/build_chapter02_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 02
python3 scripts/audit_chapter02_delivery.py
```

在固定的 vLLM 环境中执行并验证第 03 章：

```bash
export CH3_MODEL=/path/to/Qwen2.5-1.5B-Instruct
python3 scripts/execute_chapter_notebooks.py --chapter 03 --start 1 --end 30
python3 scripts/build_chapter03_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 03
python3 scripts/audit_chapter03_delivery.py
```

### 命令行方式

#### 先决条件

- 一个NVIDIA GPU
- 一个 PyTorch 构建，兼容该GPU
- Python 3.12
- 足够的存储空间下载 `Qwen/Qwen2.5-1.5B-Instruct`

文档中的实测数据采集自 RTX 5090。换用其他 GPU、软件版本、batch size 或矩阵 shape，结果可能不同。

```bash
pip install -r requirements.txt
./chapters/01-mixed-precision-int4/01-precision-formats/support/run.sh
```

使用现有本地检查点：

```bash
CH1_MODEL=/path/to/model CH1_LOCAL_FILES_ONLY=1 \
  ./chapters/01-mixed-precision-int4/01-precision-formats/support/run.sh
```

生成的文件会写入本课的 `outputs/` 目录；Git 会有意忽略该目录。

## <a id="how-this-repository-works"></a>这个仓库如何运作

每个已经发布的谜题都遵循同一个循环：

```text
Predict → Run → Inspect → Explain
```

- **预测：**看到结果前，先写下预期。
- **运行：**用可复现的脚本运行 baseline 与 candidate。
- **检查：**检查内存、latency、输出行为和实际调度的 kernel。
- **解释：**给出有边界的结论，并说明什么条件会使结论反转。

只有已经包含可运行代码和可检查证据的完整课程才会被链接。第 01、02、03 章分别发布了 30、28、30 个实验；所有实验都保留了对应 RTX 5090 环境的运行输出。

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
│   └── 03-vllm-inference-serving/
│       ├── README.md       # 30-lesson chapter map
│       └── 01-... through 30-.../
│           ├── README.md
│           ├── lab.ipynb
│           └── artifacts/
└── scripts/                # Repository-wide public-safety checks
```

## 结果与验证

查看结构化的 [RTX 5090 结果](chapters/01-mixed-precision-int4/01-precision-formats/artifacts/rtx5090-result.json) 或运行公共内容检查：

```bash
python3 scripts/check_public_safety.py .
```

## 贡献

欢迎提交修正、更清晰的解释，以及其他 GPU 上的可复现性报告。报告结果时，请包含 GPU 型号、软件版本、模型 revision、batch size、sequence length、warmup 策略和原始重复样本。

## 致谢

这种“动手学习”的呈现方式受到 [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles) 和 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 启发。AI Infra Puzzles 使用原创视觉标识，内容聚焦于以证据为基础的 LLM 推理系统。

## 作者与归属

除非另有说明，本项目中的解释、课程组织、原始图表、基准设计、笔记本和特定仓库的教程代码均为**Linnea Cai ([@LinglingCai0314](https://github.com/LinglingCai0314))**的原创作品。

这是一个独立的学习与教程项目。它不复制或重新分发源课程材料，也不隶属于课程提供方、NVIDIA、PyTorch、Hugging Face、Modular 或 Qwen，未获得这些组织的背书。第三方概念、库、模型 checkpoint、API、商标和改编材料归各自所有者，并会在使用处注明来源。完整的作者边界与来源政策见 [ATTRIBUTION.md](ATTRIBUTION.md)。

版权所有 © 2026 Linnea Cai。

## 许可证

尚未选择开源许可证。上述作者声明不授予复制、修改或重新分发作品的许可。在添加许可证文件之前，仓库仍保留所有权利。
