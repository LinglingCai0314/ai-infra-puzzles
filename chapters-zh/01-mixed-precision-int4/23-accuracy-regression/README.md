<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 23 — 量化模型的准确度回归测试

> **谜题：**一个综合得分能否掩盖严重的量化回归？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/23-accuracy-regression/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/23-accuracy-regression/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

量化质量不是一个余弦分数。一个发布可以保留平均logits，同时改变top-1决策、稀有领域、长上下文行为、校准敏感层或安全关键输出。回归测试将这些失败模式转化为冻结的门，可以阻止一个数值上很小但行为上很重要的变化。

## 阅读结果前，先做出预测

1. 预测交叉熵、困惑度和 top-1 一致性的相对方向是否一致。
2. 解释为什么合成困惑度大小作为语言模型评分没有意义。
3. 设计至少三个部署切片，使得聚合指标可能隐藏。

## 1. 从具体的张量和状态开始

高质量的证据涵盖了词元似然性（交叉熵/困惑度）、任务指标、输出/logit一致性、安全/对齐案例以及业务特定的切片。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 困惑度衡量了标记的似然性，任务准确性衡量了决策，对齐样本覆盖了产品行为。 |
| 2 | 阈值应在检查候选值之前冻结。 |
| 3 | Slice-level failures can be masked by a stable global average. |

## 2. 推导机制

困惑度是`exp(mean token cross-entropy)`；一个小的平均损失变化可以与罕见切片上的大排名变化共存。Top-1一致性揭示了决策变化，但没有表明任何答案是否正确。

对于目标y和logits z，交叉熵衡量y被赋予的概率；困惑度是`exp(loss)`，可以放大微小的损失变化。Top-1一致性则询问候选方案是否保留了基线决策，无论这两个决策是否正确。logit距离、任务准确率、精确匹配、校准以及人/安全检查回答的仍然是不同的问题。

发布门应该在候选项被评估之前定义基线、数据集、种子、容差和切片策略。否则，阈值会随观察到的回归而漂移。

## 3. 把理论转化为实验**实验：**运行一个微小的 CUDA 语言模型头在之前和之后 INT4 权重 Q/DQ，然后比较交叉熵、困惑度、top-1协议和切片度量。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 浮点合成分类器对4,096标记的逻辑值 |
| 候选方案 | INT4-去量化后的权重对数概率值，对应相同的隐藏状态和目标。 |
| 保持不变 | tokens, vocabulary, targets, 隐藏状态, 权重矩阵, 种子 |
| 测量 | 损失，衍生困惑度，整体和半切片 top-1协议 |
| 证据标签 | `pytorch-gpu` |

CUDA 探针计算损失、困惑度、整体一致性以及在 INT4 Q/DQ 之前和之后的两个相同隐藏状态切片。

### 代码导读

该笔记本生成一个固定的合成分类问题，计算基线和量化后的logits，并评估相同的目标。它报告了完整的集合和两半，这样无法通过一个聚合值隐藏切片分歧。

因为随机的logits会导致巨大的损失和困惑度，因此其绝对值被有意地标记为合成值。这个练习展示了度量关系和门结构，而不是语言建模能力。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| 基准损失 | 32.049492 |
| 候选损失 | 32.212620 |
| Baseline synthetic perplexity | 8.297e+13 |
| 候选合成困惑度 | 9.767e+13 |
| 最高1协议 | 83.6914% |

### 这些数字说明了什么

候选损失从32.049492增加到32.212620。指数运算将这一微小差异转化为合成困惑度约为`8.30e13`和`9.77e13`。总体上，前1一致率为0.836914；两部分分别为0.838379和0.835449。

几乎相等的切片在这组构造的数据集中没有揭示出集中化的失败，但大约 16% 的决策分歧显然是可见的。实际发布需要任务的正确性，而不仅仅是与基准的一致性。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/23-accuracy-regression/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 使用分层质量门限，并保留必要的基线输出以解释回归。

### 验收与回滚门槛

冻结数据集、提示、解码、基准修订、阈值和切片定义。即使全局平均通过，关键切片仍会失败。

### 这个结论可能如何失效

困惑度在极端合成损失下可能会溢出或难以解释。基准一致性可能会保留基准错误，而平均准确率可能会掩盖关键部分。在回归中重用校准提示也会让量化器选择过度拟合门控。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/23-accuracy-regression/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

将合成对数概率替换为一个小型命名模型和一个冻结的、重分布安全的套件：保留文本的困惑度，任务准确性，长上下文切片，多语言/代码/工具使用样本，以及答案/对数概率的一致性。在运行候选模型之前，发布阈值和反转标准。

## 证据边界

**证据标签:** [`pytorch-gpu`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [TorchAO 文档](https://docs.pytorch.org/ao/stable/index.html)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [PyTorch 可再现性注释](https://docs.pytorch.org/docs/stable/notes/randomness.html)
