<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# 第 09 课 — PTQ 校准数据：采样与覆盖

> **谜题：**一个小的校准集是否能够代表生产流量将使用的激活范围？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/09-ptq-calibration/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/09-ptq-calibration/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

后训练量化冻结了从示例中获得的缩放值。如果这些示例省略了长提示、罕见领域或激活离群值，量化器在校准数据上可能看起来很好，但在生产流量上会截断。因此，校准质量是一个覆盖问题，而不是样本数量问题。

## 阅读结果前，先做出预测

1. 预测哪个校准集能最小化裁剪，哪个能最小化混合留出集上的平均舍入误差。
2. 解释为什么在选择缩放后必须保持评估数据的独立性。
3. 列出随机抽样可能低估的部署层次。

## 1. 从具体的张量和状态开始

一个PTQ管道使用一个校准分布来冻结量化参数，并使用一个独立的评估分布来测试冻结的结果。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 校准估计范围或统计；评估测试在保留数据上冻结的决策。 |
| 2 | 罕见的域和长序列可以主导最坏情况下的激活范围。 |
| 3 | 更多的样本无助于如果采样重复相同的狭窄分布。 |

## 2. 推导机制

最大校准保护观察到的极端值，但可能会浪费大多数代码；百分位数或学习裁剪以可控的尾部交换较小的步骤。这两种选择在校准集不包含部署域时都会失败。

一个最大范围校准器选择`s=max(|x_cal|)/qmax`；一个百分位校准器故意剪枝尾部以缩小步长。两者都估计校准分布的某个属性。当部署分布的尾部更大或位置不同，泛化会失败。更多相同窄提示的副本可以减少估计器噪声，但不会减少分布偏差。

保留出来的裁剪分数衡量超出冻结表示范围的值。RMSE 衡量裁剪尾部和量化步骤的综合成本。这些目标可能会不一致：一个考虑异常值的缩放因子可以避免裁剪，但浪费分辨率在大多数普通值上。

## 3. 把理论转化为实验**实验：**校准 INT8 在狭窄、平衡和异常值感知的合成数据集上调整激活尺度，然后在混合保留分布上评估所有尺度。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 从狭窄的合成校准分布冻结比例尺 |
| 候选方案 | 平衡且明确地考虑异常值的校准集 |
| 保持不变 | INT8 公式，保留混合张量，评估指标，种子 |
| 测量 | 冻结尺度，保留的裁剪分数，RMSE，MAE，余弦，最大误差 |
| 证据标签 | `numerical-model` |

实验室冻结了来自狭窄、平衡和异常值感知的集合的尺度，并在单一混合保留张量上评估了这三种方法。

### 代码导读

该笔记本创建了三个校准group，冻结了每个group中的一个比例尺，并在相同的混合保留张量上评估它们。它在评估数据上从未重新计算过比例尺。这使得比较成为一种小的分布偏移测试，而不是重建演示。

这些示例是合成的，因此域标签是可控的。模型研究将用分层提示和层激活捕获替换它们，同时保持相同的校准/评估分离。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| 窄剪辑 | 2.6478% |
| Narrow RMSE | 0.317395 |
| 平衡裁剪 | 0.0250% |
| 平衡的 RMSE | 0.085046 |
| 异常值感知裁剪 | 0.0000% |
| 基于异常检测的 RMSE | 0.077629 |

### 这些数字说明了什么

窄尺度裁剪了 2.647752% 的保留值，并产生了 RMSE 0.317395，最大误差为 27.9039。平衡校准将裁剪减少到 0.025001% 并将 RMSE 减少到 0.085046。异常值感知校准消除了裁剪，但其更大的尺度提高了 MAE 到 0.067231；其 RMSE，0.077629 仍然稍微更好，因为它避免了灾难性的尾部误差。

没有普遍最佳的行，除非有部署目标。如果尾部失败不可接受，那么感知异常的缩放策略会赢得这次探测。如果平均小值解决占据主导地位，那么截断或混合策略可能更优。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/09-ptq-calibration/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 选择覆盖部署模式的校准数据，并将其与回归集分开。

### 验收与回滚门槛

发布采样规则、长度/域、种子、统计量、样本数量以及保留的剪切/错误。永远不要在用于最终质量门的示例上调整范围。

### 这个结论可能如何失效

调整最终回归集的百分位数会泄露评估到校准中。仅报告平均误差可能会隐藏罕见的灾难性裁剪，而仅报告最大误差可能会让一个异常值消耗整个代码范围。覆盖率元数据——领域、长度、语言、工具使用和频率——是量化产物的一部分。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/09-ptq-calibration/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

构建分层校准表单以适应真实提示，并在固定样本数量下比较随机、平衡和尾部过采样的选择。评估每层裁剪和任务切片在不相交数据集上的表现，然后测试所选缩放策略在模型修订中是否保持稳定。

## 证据边界

**证据标签:** [`numerical-model`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [TensorRT量化方案](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [NVIDIA Model Optimizer PTQ 文档](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
- [TensorRT量化工作流](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)
