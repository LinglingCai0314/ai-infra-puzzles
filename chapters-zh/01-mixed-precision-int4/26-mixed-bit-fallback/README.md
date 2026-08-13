<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 26 — 混合位策略与敏感层回退

> **谜题：**如果只有少数几层导致了大部分量化误差，那么每层都应该使用更多的位数吗？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/26-mixed-bit-fallback/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/26-mixed-bit-fallback/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

均匀四比特量化在不同敏感度的层上分配相同的精度。混合比特策略衡量每个层对端到端输出的扰动程度，然后为最严重的影响者分配固定的更高精度预算。预算和重组后的模型结果与排名一样重要。

## 阅读结果前，先做出预测

1. 预测仅允许两个备选方案时，哪些层会接收到 INT8。
2. 计算两个 INT8 和四个 INT4 等大小层的预期平均位宽。
3. 解释为什么孤立层敏感性必须跟随组装模型评估。

## 1. 从具体的张量和状态开始

混合位设计为每个内存、延迟和质量预算下的层或组分配一个精度/配置。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 层敏感性通过代表输入下的下游目标来衡量。 |
| 2 | 混合位分配在元数据和kernel多样性之间权衡质量。 |
| 3 | Fallback layers需要一个确定性的规则和固定的内存预算。 |

## 2. 推导机制

敏感性扫描逐层替换，并测量下游变化。简单的分配则在每增加一个字节时，额外分配比特数给边际质量效益最大的部分；交互作用需要重新评估组装的模型。

让候选位分配 b_l 最小化模型误差，同时满足 `Σ n_l b_l / Σ n_l ≤ B` 条件，其中 n_l 是层大小，B 是平均位预算。一个简单的贪婪策略衡量每次量化一层所引起的输出 RMSE，并将额外精度分配给最大的分数。交互使得这只是一个启发式方法：两个单独安全的层在共同量化时可能会相互放大。

因此，该过程分为两个阶段——在固定探针下排序，然后组装并重新测试完整的分配。存储、kernel兼容性和延迟也必须重新计算，因为混合格式可能会增加调度边界。

## 3. 把理论转化为实验

**实验：**量化一个六层的 CUDA 逐层构建MLP，逐层计算敏感性，然后构建预算。INT4/INT8 混合位候选。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 六层浮点MLP和一个 INT4 候选方案 |
| 候选方案 | INT8 对于两个最敏感的层，INT4 对于剩下的四个 |
| 保持不变 | 等层大小，校准输入，量化器，两层后备预算 |
| 测量 | 逐层隔离 RMSE，选择层，平均权重位数，组装输出误差 |
| 证据标签 | `pytorch-gpu` |

六层 CUDA 实验室对 INT4 进行替换排序，分配两层 INT8，计算平均比特数，并重新进行端到端运行。

### 代码导读

该笔记本将六个等大小的矩阵分别量化为 INT4 和 INT8。它一次替换一层，以测量对全精度网络的敏感性，选择前两层，构建混合模型，并重新评估端到端。

因为各层大小相等，预算透明：`(2×8 + 4×4)/6 = 5.333` 比特每权重。一个真实的变压器会按参数数量和后端兼容的分组来加权各层。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| INT8 备选层 | 0, 1 |
| 平均重量位 | 5.333比特/权重 |
| 层 0 隔离 RMSE | 0.001379 |
| 层 1 隔离 RMSE | 0.001304 |
| 组装 RMSE | 0.002484 |
| 组装余弦 | 0.976198 |

### 这些数字说明了什么

层0和1的独立 RMSE、0.0013786和0.00130424最高，因此它们接收到了 INT8。混合分配使用了5.333平均比特，并产生了使用余弦0.976198组装的 RMSE 0.00248394。

组装的错误比任何孤立的分数都大，这表明各层之间存在交互作用。排名仍然提供了一个可重复的预算候选方案，但是否优于所有-INT4 或其他分配必须通过冻结质量目标和实际存储/运行时测量来判断。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/26-mixed-bit-fallback/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 使用敏感性扫描来优化保护目标的精度，然后重新测量组装的模型。

### 验收与回滚门槛

冻结校准/评估，记录孤立灵敏度、预算、选定的备用层、最终组装的质量、存储、operator覆盖范围和延迟。

### 这个结论可能如何失效

在最终任务集中选择备用层会导致部署评估过度拟合。不报告平均位数而比较混合位数质量是不公平的。后端碎片化也可能抹杀理论上的好处，如果 INT4 和 INT8 层使用不兼容的打包方式或强制同步/实例化。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/26-mixed-bit-fallback/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

添加所有 INT4 和所有 INT8 组装基线，搜索几个预算，并绘制质量与有效字节的关系图。在多个领域和sequence length上重复敏感性测试。然后运行一个支持混合格式的后端，并测量操作边界、内存和延迟。

## 证据边界

**证据标签:** [`pytorch-gpu`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [TorchAO 文档](https://docs.pytorch.org/ao/stable/index.html)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
