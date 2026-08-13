<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# 第 14 课 — bitsandbytes 4-Bit 载入：NF4，计算 Dtype 和嵌套量化

> **谜题：**Does 是否`load_in_4bit=True`指定该层如何计算？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

`load_in_4bit=True` 不是一个完整的数值规范。bitsandbytes 配置还会选择一个codebook，如 NF4，用于去量化矩阵操作的计算dtype，并可选地嵌套量化元数据。加载的模块类和后端可用性决定了这些设置是否成为实际操作符，还是保持为配置文本。

## 阅读结果前，先做出预测

1. 区分量化码书、压缩存储dtype、计算dtype以及嵌套量化。
2. 预测在正态分布的权重下，NF4 或均匀分布的 INT4 哪个会给出更低的 RMSE。
3. 要将结果标记为本地bitsandbytes运行，需要哪些证据？

## 1. 从具体的张量和状态开始

bitsandbytes 4-bit 配置至少包含存储代码本（`NF4` 或 FP4）、计算dtype、可选的双精度/嵌套量化以及消费它的模块/后端。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 存储类型、量化码本和计算dtype是独立的选择。 |
| 2 | 嵌套量化压缩量化元数据；它不将激活计算转换为二进制算术。 |
| 3 | 在声称bitsandbytes运行之前，必须检查包存在和设备支持。 |

## 2. 推导机制

NF4 为 16 代码分配时，不是以等间距的整数间隔分配，而是非均匀分配。在进行线性操作时，压缩后的代码会被解量化或由融合路径消耗，而激活值则使用配置的计算dtype。

均匀的 INT4 在选定的范围内放置均匀间隔的重建级别。NF4 则使用一个非均匀的codebook，其级别在正态分布概率质量更大的地方分配更多的分辨率。存储的代码选择一个级别；矩阵乘法仍然需要去量化/缩放和浮点计算路径。双量化或嵌套量化减少了量化常数的成本，而不是将激活算术减少到两位。

codebook的质量取决于权重分布和归一化规则。NF4可以在保持平均误差的同时降低钟形权重的平均误差，但在尾部产生比范围拟合均匀网格更大的最坏情况误差。部署决策还包括kernel支持和计算dtype稳定性。

## 3. 把理论转化为实验

**实验：**比较参考NF4codebook，具有统一 INT4 在正态分布的权重上进行操作，并验证是否bitsandbytes已安装。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 均匀对称 INT4 重构正态分布权重 |
| 候选方案 | 参考 NF4 代码库重建相同的权重 |
| 保持不变 | 权重张量，归一化，代码数量，错误参考，种子 |
| 测量 | RMSE/MAE/cosine/max error andbitsandbytes安装探针 |
| 证据标签 | `numerical-model` |

实验室隔离了codebook重建，并分别记录了包的存在，因此一个数值的NF4结果不能伪装成bitsandbytes执行。

### 代码导读

该笔记本通过一个参考 NF4 代码本和一个均匀的 INT4 量化器，映射相同的随机权重，然后与原始张量进行比较。它分别检查 bitsandbytes 是否可导入。将这些分支分开，可以防止数值代码本实验伪装成库基准测试。

没有加载变压器模型，没有实例化`Linear4bit`模块，也没有在仓库记录的环境中的bitsandbyteskernel进行计时。因此，证据标签仍然保持为`numerical-model`。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| NF4 RMSE | 0.127836 |
| 统一 INT4 RMSE | 0.142396 |
| NF4最大误差 | 0.719360 |
| 均匀的 INT4 最大误差 | 0.339060 |
| bitsandbytes 已安装 | 否 |

### 这些数字说明了什么

NF4实现 RMSE 0.127836和 MAE0.109566, 低于均匀 INT4 在 RMSE 0.142396和 MAE0.122676对于这个常规张量。均匀 INT4 有一个较小的最大误差。0.339059与...对比NF4's 0.719360, 显示平均值和尾部目标可能不一致。环境探针报告了`bitsandbytes_installed=false`.

结果仅支持基于分布的codebook直觉。它没有提到本地层内存、吞吐量、嵌套量化开销或此RTX堆栈上的任务质量。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 记录量化类型、计算dtype、嵌套量化设置以及实际模块类。

### 验收与回滚门槛

捕获 `BitsAndBytesConfig`，包/CUDA 兼容性，实际模块类，存储字节，operator 证据，输出回归，和时间。

### 这个结论可能如何失效

一个参考代码本可能与库规范化、块大小、打包和缩放dtype不同。声称bitsandbytes的速度来自它也是错误的。另一个陷阱是选择NF4，而下游层对稀有尾部错误敏感时，平均 RMSE。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

安装与当前 PyTorch/CUDA 系统兼容的版本，加载一个 `Linear4bit` 层，并记录其实际模块、存储张量、计算dtype、输出错误和操作追踪。重复进行有嵌套量化和无嵌套量化的情况，然后使用小型模型质量套件。

## 证据边界

**证据标签:** [`numerical-model`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [Transformer bitsandbytes 指南](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes/main/en/index)
