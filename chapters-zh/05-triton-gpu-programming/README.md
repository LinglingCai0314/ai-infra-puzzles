<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# 第 05 章 — Triton GPU 编程与 CUDA 性能对比

[← 中文首页](../../README_ZH.md) · [English chapter](../../chapters/05-triton-gpu-programming/README.md)

第 05 章共 30 课，从 Triton blocked programming model 一直讲到可交付自定义 kernel。 课程以 Linnea Cai 的 Triton GPU
编程学习材料为理论底稿，重新组织成可执行 puzzle。 每个主题都包含预测、明确命名的 CUDA/库函数 control、可审阅实现、正确性 gate、完整
计时样本，以及说明适用边界的结论。

仓库结果来自 NVIDIA GeForce RTX 5090、CUDA runtime 13.0、PyTorch 2.13.0 和 Triton 3.7.1，目标是 CUDA
architecture 120。执行环境没有 `nvcc`，因此第 05 课保留等价 CUDA C++ 源码，并把工具链记录为不可用，不虚构 CUDA latency。全章会分别标注
PyTorch CUDA、 cuBLAS-backed `torch.mm`、SDPA 和自定义 Triton 路径，避免把不同 baseline 混成“CUDA”。

```mermaid
flowchart LR
  A["blocked program + mask"] --> B["访存 + benchmark"]
  B --> C["Softmax + reduction + GEMM"]
  C --> D["Norm + Attention + 稳定性"]
  D --> E["compile + paged KV + persistence"]
  E --> F["CI + 选型 + 交付"]
```

## 学习方法

1. 先预测正确性与 latency，再打开保留输出。
2. 先看 baseline 名称：自定义 CUDA 源码、PyTorch CUDA、cuBLAS、SDPA 与数值模型不能互换。
3. 在 JSON artifact 中检查完整计时样本与环境身份。
4. 把结论迁移到其他算子前，先重跑特殊 tail 与布局。
5. 自定义 kernel 通过声明 gate 前，始终保留库函数或 PyTorch rollback。

## 证据标签

| 标签 | 能说明什么 |
|---|---|
| `native-backend` | 指定 Triton 或 PyTorch CUDA 路径在记录的 RTX 5090 软件栈上执行 |
| `compatibility-probe` | 检查 API、backend target、源码或编译器能力，不声称未执行路径的性能 |
| `capacity-model` | 实测值进入透明的流量或决策模型 |

## 阶段 I — 编程与测量基础

| 课 | 问题 | 实验 |
|---:|---|---|
| 01 | [算子边界与小 kernel 的真实成本](01-operator-boundaries/README.md) | [notebook](../../chapters/05-triton-gpu-programming/01-operator-boundaries/lab.ipynb) |
| 02 | [Blocked program 与 CUDA thread](02-programming-models/README.md) | [notebook](../../chapters/05-triton-gpu-programming/02-programming-models/lab.ipynb) |
| 03 | [版本身份与可复现实验基线](03-reproducible-baseline/README.md) | [notebook](../../chapters/05-triton-gpu-programming/03-reproducible-baseline/lab.ipynb) |
| 04 | [第一个 tail-safe 向量 kernel](04-first-vector-kernel/README.md) | [notebook](../../chapters/05-triton-gpu-programming/04-first-vector-kernel/lab.ipynb) |
| 05 | [显式 CUDA 控制与错误边界](05-explicit-cuda-control/README.md) | [notebook](../../chapters/05-triton-gpu-programming/05-explicit-cuda-control/lab.ipynb) |
| 06 | [用 stride 诊断合并访存](06-memory-coalescing/README.md) | [notebook](../../chapters/05-triton-gpu-programming/06-memory-coalescing/lab.ipynb) |
| 07 | [mask 与 reduction 中性元](07-mask-semantics/README.md) | [notebook](../../chapters/05-triton-gpu-programming/07-mask-semantics/lab.ipynb) |
| 08 | [指针算术与 tensor 布局](08-pointer-layout/README.md) | [notebook](../../chapters/05-triton-gpu-programming/08-pointer-layout/lab.ipynb) |
| 09 | [Benchmark 协议与计时误差](09-benchmark-protocol/README.md) | [notebook](../../chapters/05-triton-gpu-programming/09-benchmark-protocol/lab.ipynb) |
| 10 | [调优前先做 Roofline 判断](10-roofline-arithmetic-intensity/README.md) | [notebook](../../chapters/05-triton-gpu-programming/10-roofline-arithmetic-intensity/lab.ipynb) |

## 阶段 II — 核心算子与资源权衡

| 课 | 问题 | 实验 |
|---:|---|---|
| 11 | [融合 Softmax](11-fused-softmax/README.md) | [notebook](../../chapters/05-triton-gpu-programming/11-fused-softmax/lab.ipynb) |
| 12 | [Reduction 与 Scan](12-reduction-and-scan/README.md) | [notebook](../../chapters/05-triton-gpu-programming/12-reduction-and-scan/lab.ipynb) |
| 13 | [Matmul tiling 与库函数边界](13-matmul-tiling/README.md) | [notebook](../../chapters/05-triton-gpu-programming/13-matmul-tiling/lab.ipynb) |
| 14 | [Autotune 搜索与实验预算](14-autotune-budget/README.md) | [notebook](../../chapters/05-triton-gpu-programming/14-autotune-budget/lab.ipynb) |
| 15 | [寄存器、warp 与 occupancy 权衡](15-resources-occupancy/README.md) | [notebook](../../chapters/05-triton-gpu-programming/15-resources-occupancy/lab.ipynb) |
| 16 | [Tensor Core 与 dtype 语义](16-tensor-cores-dtypes/README.md) | [notebook](../../chapters/05-triton-gpu-programming/16-tensor-cores-dtypes/lab.ipynb) |
| 17 | [融合 LayerNorm 与 RMSNorm](17-fused-rmsnorm/README.md) | [notebook](../../chapters/05-triton-gpu-programming/17-fused-rmsnorm/lab.ipynb) |

## 阶段 III — Attention、稳定性与集成

| 课 | 问题 | 实验 |
|---:|---|---|
| 18 | [在线 Softmax 与融合 Attention](18-online-softmax-attention/README.md) | [notebook](../../chapters/05-triton-gpu-programming/18-online-softmax-attention/lab.ipynb) |
| 19 | [数值稳定性与 cast 边界](19-numerical-stability/README.md) | [notebook](../../chapters/05-triton-gpu-programming/19-numerical-stability/lab.ipynb) |
| 20 | [Interpreter、断言与调试工具](20-debugging-tools/README.md) | [notebook](../../chapters/05-triton-gpu-programming/20-debugging-tools/lab.ipynb) |
| 21 | [从 Triton 源码到 IR 与 PTX](21-ir-ptx-reading/README.md) | [notebook](../../chapters/05-triton-gpu-programming/21-ir-ptx-reading/lab.ipynb) |
| 22 | [通过 torch.compile 集成 PyTorch](22-torch-compile-integration/README.md) | [notebook](../../chapters/05-triton-gpu-programming/22-torch-compile-integration/lab.ipynb) |
| 23 | [Paged KV Cache 寻址](23-paged-kv-gather/README.md) | [notebook](../../chapters/05-triton-gpu-programming/23-paged-kv-gather/lab.ipynb) |

## 阶段 IV — 可移植性与高级调度

| 课 | 问题 | 实验 |
|---:|---|---|
| 24 | [Backend 可移植性与 ROCm 边界](24-backend-portability/README.md) | [notebook](../../chapters/05-triton-gpu-programming/24-backend-portability/lab.ipynb) |
| 25 | [动态 shape 与特化](25-dynamic-shapes/README.md) | [notebook](../../chapters/05-triton-gpu-programming/25-dynamic-shapes/lab.ipynb) |
| 26 | [Persistent 调度与 TMA 边界](26-persistent-kernels/README.md) | [notebook](../../chapters/05-triton-gpu-programming/26-persistent-kernels/lab.ipynb) |
| 27 | [CUTLASS、cuBLAS、cuDNN 还是 Triton](27-library-or-custom/README.md) | [notebook](../../chapters/05-triton-gpu-programming/27-library-or-custom/lab.ipynb) |

## 阶段 V — CI、选型与交付

| 课 | 问题 | 实验 |
|---:|---|---|
| 28 | [性能回归 CI](28-performance-regression-ci/README.md) | [notebook](../../chapters/05-triton-gpu-programming/28-performance-regression-ci/lab.ipynb) |
| 29 | [Triton 与 CUDA 选型框架](29-selection-framework/README.md) | [notebook](../../chapters/05-triton-gpu-programming/29-selection-framework/lab.ipynb) |
| 30 | [从慢子图到可交付 kernel](30-deliverable-kernel/README.md) | [notebook](../../chapters/05-triton-gpu-programming/30-deliverable-kernel/lab.ipynb) |

## 共享实现

可执行 kernel 统一保存在 [`scripts/chapter05_runtime.py`](../../scripts/chapter05_runtime.py)。
一份审阅源码可以避免 30 个 Notebook 相互漂移，同时每课仍有独立入口与 canonical result。 第 05 课还提供
[`vector_affine.cu`](../../chapters/05-triton-gpu-programming/05-explicit-cuda-control/vector_affine.cu)，
在本地 CUDA Toolkit 可用时可以构建这条显式 CUDA control。

## 复现与验证

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start 1 --end 30
python3 scripts/build_chapter05_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 05
python3 scripts/audit_chapter05_delivery.py
```
