# 第 02 章 — 稀疏性与结构化剪枝

本章不把模型稀疏性停留在“数一数有多少个零”，而是将它变成一连串可验证的决策。28 课内容涵盖目标、granularity、mask、物理 channel 删除、依赖图、恢复 schedule、N:M 约束、框架生命周期、ONNX/TensorRT 边界、CNN/Transformer/LLM 案例、benchmark、rollback、可复现性和平台相关的部署方案。

[← 中文首页](../../README_ZH.md) · [English chapter](../../chapters/02-sparsity-structured-pruning/README.md)

每一课都遵循一个交付合同：

```text
Concrete tensors/state → mechanism or equation → frozen comparison
                       → retained RTX 5090 evidence → acceptance/rollback
```

这些笔记是独立于学习材料中的想法和工程问题撰写的。源HTML没有被复制到这个仓库中。数值模型、兼容性探针、原生后端和性能运行带有不同的证据标签，因此包检查或零速率计算不能被误认为是加速。

## 交付循环概览

```mermaid
flowchart LR
  A["Define the delivery target"] --> B["Choose a pruning granularity"]
  B --> C["Prune with dependency constraints"]
  C --> D["Recover quality"]
  D --> E["Export a supported representation"]
  E --> F["Prove runtime and product value"]
  F -->|"gate fails"| C
  F -->|"gate passes"| G["Release with rollback evidence"]
```

## 如何阅读一节课

1. 在打开保留结果之前进行预测。
2. 将图表和推导映射到基准和候选`lab.ipynb`。
3. 在比较指标之前，请验证环境和冻结变量。
4. 将笔记本输出与 JSON artifact对齐，然后应用接受门限。

## 证据标签

| 标签 | 它所建立的内容 |
|---|---|
| `pytorch-gpu` | 通过 PyTorch 执行 CUDA，不推断出一个未命名的本地稀疏kernel。 |
| `numerical-model` | 受控机制，而非完整论文或生产复制品 |
| `compatibility-probe` | 包或API的可用性及其确切的成功/失败边界 |
| `native-backend` | 通过命名的后端执行记录的模型和工作负载 |
| `capacity-model` | 基于测量的 CUDA 事实的透明规划算术 |

## 第一阶段——目标和剪枝机制

| 课 | 核心决策 | 实验室 |
|---:|---|---|
| 01 | [剪枝目标、约束和交付边界](01-pruning-objectives/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/01-pruning-objectives/lab.ipynb) |
| 02 | [稀疏粒度谱：权重、通道、块和N:M](02-sparsity-granularity/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/02-sparsity-granularity/lab.ipynb) |
| 03 | [基准测量：参数，FLOPs，延迟和吞吐量](03-baseline-measurement/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/03-baseline-measurement/lab.ipynb) |
| 04 | [闭合回路：训练、剪枝、恢复和重新评估](04-prune-finetune-loop/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/04-prune-finetune-loop/lab.ipynb) |
| 05 | [无结构幅度剪枝，无需存储神话](05-unstructured-magnitude-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/05-unstructured-magnitude-pruning/lab.ipynb) |
| 06 | [全局稀疏性和层间预算分配](06-global-layerwise-budgets/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/06-global-layerwise-budgets/lab.ipynb) |
| 07 | [剪枝滤波器：使卷积物理上变窄](07-filter-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/07-filter-pruning/lab.ipynb) |

## 第二阶段——依赖项、时间表和框架生命周期

| 课 | 核心决策 | 实验室 |
|---:|---|---|
| 08 | [批量归一化缩放因子和网络瘦身](08-network-slimming/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/08-network-slimming/lab.ipynb) |
| 09 | [残差、拼接和依赖图剪枝](09-dependency-graph-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/09-dependency-graph-pruning/lab.ipynb) |
| 10 | [泰勒重要性：按损失变化排名通道](10-taylor-importance/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/10-taylor-importance/lab.ipynb) |
| 11 | [渐进式剪枝计划和恢复训练](11-gradual-pruning-schedule/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/11-gradual-pruning-schedule/lab.ipynb) |
| 12 | [稀疏正则化和可学习结构门](12-sparse-regularization-gates/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/12-sparse-regularization-gates/lab.ipynb) |
| 13 | [N:M 半结构化稀疏性和 2:4 合同](13-nm-2-4-sparsity/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/13-nm-2-4-sparsity/lab.ipynb) |
| 14 | [PyTorch 精简 API 和完整的掩码生命周期](14-pytorch-mask-lifecycle/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/14-pytorch-mask-lifecycle/lab.ipynb) |

## 第三阶段——原生工具链和模型家族

| 课 | 核心决策 | 实验室 |
|---:|---|---|
| 15 | [Torch-Pruning DepGraph: 结构化剪枝兼容性实验室](15-depgraph-structured-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/15-depgraph-structured-pruning/lab.ipynb) |
| 16 | [TensorFlow MOT 和 Keras 裁剪/导出生命周期](16-keras-pruning-lifecycle/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/16-keras-pruning-lifecycle/lab.ipynb) |
| 17 | [OpenVINO, NNCF, 和 Intel 运行时稀疏性](17-cpu-runtime-sparsity/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/17-cpu-runtime-sparsity/lab.ipynb) |
| 18 | [TensorRT稀疏部署和Polygraphy证据](18-tensorrt-sparsity-gates/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/18-tensorrt-sparsity-gates/lab.ipynb) |
| 19 | [ONNX导出、图修复和形状一致性](19-onnx-shape-consistency/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/19-onnx-shape-consistency/lab.ipynb) |
| 20 | [CNN 案例研究：ResNet 通道剪枝](20-resnet-channel-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/20-resnet-channel-pruning/lab.ipynb) |
| 21 | [安全剪枝用于检测和分割](21-detection-segmentation-safety/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/21-detection-segmentation-safety/lab.ipynb) |

## 第四阶段——变压器、生产证据和平台决策

| 课 | 核心决策 | 实验室 |
|---:|---|---|
| 22 | [剪枝Transformer头、FFN神经元和层](22-transformer-structure-pruning/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/22-transformer-structure-pruning/lab.ipynb) |
| 23 | [一次性LLM剪枝：SparseGPT和Wanda机制](23-sparsegpt-wanda/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/23-sparsegpt-wanda/lab.ipynb) |
| 24 | [排序蒸馏、量化和剪枝](24-compression-order/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/24-compression-order/lab.ipynb) |
| 25 | [基准测试稀疏性：证明实际加速](25-sparsity-benchmarking/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/25-sparsity-benchmarking/lab.ipynb) |
| 26 | [准确性恢复、回滚和切片错误分析](26-accuracy-recovery-rollback/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/26-accuracy-recovery-rollback/lab.ipynb) |
| 27 | [自动化实验管理与可重复剪枝记录](27-reproducible-experiments/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/27-reproducible-experiments/lab.ipynb) |
| 28 | [为什么边缘和服务器部署需要不同的剪枝策略](28-edge-vs-server/README.md) | [笔记本](../../chapters/02-sparsity-structured-pruning/28-edge-vs-server/lab.ipynb) |

## 复现和验证

在 CUDA GPU 上从仓库根目录执行所有实验室：

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 02 --start 1 --end 28
python3 scripts/build_chapter02_lessons.py
python3 scripts/validate_chapter.py 02
python3 scripts/audit_chapter02_delivery.py
```

可选框架课程在缺少原生包时保留有限的兼容性结果。安装命名的后端并在做出后端性能声明前重新运行该笔记本。
