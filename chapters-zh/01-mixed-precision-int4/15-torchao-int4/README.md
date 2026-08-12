# 第 15 课 — TorchAO INT4 仅权重量化

> **谜题：**Can a PyTorch-nativeINT4 转换减少存储量但仍会损失延迟吗？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/15-torchao-int4/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/15-torchao-int4/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

一个 PyTorch 自带的量化 API 仍然依赖于精确的包、ABI、硬件和kernel组合。转换成功、压缩存储、数值一致性和延迟是不同的门槛。保留失败的兼容性尝试比默默地替换一个假量化器并称之为 TorchAO 更有用。

## 阅读结果前，先做出预测

1. 预测比较前所需的证据序列。TorchAO INT4 延迟 BF16.
2. 决定已安装的`torchao`包是否足以声称本地执行。
3. 解释一个ABI或辅助kernel依赖如何阻止一个原本支持的GPU。

## 1. 从具体的张量和状态开始

TorchAO转换替换或包装符合条件的`Linear`权重使用了一个压缩张量子类/配置。Python模块、压缩存储和选择的矩阵乘法kernel是三个可检查的层。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | TorchAO 根据量化配置替换符合条件的模块。 |
| 2 | 压缩存储和执行operator 证据与模块标签是不同的。 |
| 3 | 小批量和特定形状的开销可能超过较低的内存流量。 |

## 2. 推导机制

INT4 仅权重计算概念上读取压缩代码和分组缩放，而 BF16 激活进入线性操作。现代TorchAO 版本可以选择压缩格式和外部kernel库，如MSLK。

仅权重转换将符合条件的线性模块替换为存储压缩低比特权重的表示，并为浮点输入分配兼容的操作符。只有在转换成功、压缩保留，并且运行时避免生成完整的去量化权重时，理论上才会减少带宽。仅凭包元数据无法证明这些条件。

兼容性链是`Python package → PyTorch ABI → auxiliary kernel package → GPU architecture → quantization config → converted module → executed operator`。链开始处的中断会阻止在链的下部进行有意义的内存、错误或延迟比较。

## 3. 把理论转化为实验

**实验：**转换一个 BF16 线性层与TorchAO INT4, 记录生成的模块类型，比较输出错误，并记录两条路径的时间。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | BF16 线性模块，保留作为备用路径 |
| 候选方案 | TorchAO `Int4WeightOnlyConfig`相同转换 CUDA 堆栈 |
| 保持不变 | PyTorch 2.12/CUDA 13 环境，RTX 5090，层/配置意图 |
| 测量 | 包状态，转换状态，确切的异常类/消息；仅在成功时显示下游指标 |
| 证据标签 | `compatibility-probe` |

该笔记本尝试在明确的兼容性边界内执行文档中记录的原生配置，并在路径无法执行时记录确切的失败类。

### 代码导读

该笔记本导入TorchAO，构建预期的转换，并捕获确切的失败，而不是替换候选值。JSON结果记录了`torchao_installed=true`和`conversion=failed`，区分了包发现与后端就绪。

因为转换在量化模块存在之前就已经停止，所以笔记本正确地省略了候选者的存储、输出错误、操作符和延迟数字。从参考量化器中捏造这些数字会回答不同的问题。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| TorchAO 已安装 | 是的 |
| 转换状态 | 失败 |
| 故障类型 | ImportError |
| 失败信息 | 需要 mslk >= 1.0.0 |

### 这些数字说明了什么

TorchAO 存在，但转换引发了 `ImportError: Requires mslk >= 1.0.0`。本地 INT4 操作符未执行，因此证据标签是 `compatibility-probe`，而非 `native-backend`。这个负面结果确立了保存环境的确切边界，并且可以重现下一步操作。

本课01使用了完整的TorchAO路径，该路径在测试配置下确实执行了。这种对比很有价值：后端支持可能依赖于API/配置和依赖版本，即使在同一GPU上也是如此，因此结果必须保持与其确切路径的关联。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/15-torchao-int4/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 将TorchAO和 INT4 视为测量后的后端路径，而非四比特权重的通用性能属性。

### 验收与回滚门槛

要求成功导入/转换、量化张量/模块身份、存储计费、操作证据、输出错误以及重复延迟。保留依赖失败而非静默回退。

### 这个结论可能如何失效

最糟糕的反应是捕获错误，运行手工编写的假量化器，并留下标题 'TorchAO benchmark'。另一种失败是安装任意的夜间轮子，直到导入成功为止，而没有检查ABI兼容性或环境是否因其他课程而改变。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/15-torchao-int4/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

使用TorchAO兼容性矩阵创建一个隔离环境，安装匹配的MSLK/PyTorch 构建，并重新运行转换。只有在成功后，实验室才能添加模块类型、压缩存储、输出错误、operator跟踪、warmup、重复延迟以及与 BF16 基线的比较。

## 证据边界

**证据标签:** [`compatibility-probe`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [TorchAO 文档](https://docs.pytorch.org/ao/stable/index.html)
- [TorchAO量化 API](https://docs.pytorch.org/ao/stable/api_reference/index.html)
- [TorchAO 仓库](https://github.com/pytorch/ao)
