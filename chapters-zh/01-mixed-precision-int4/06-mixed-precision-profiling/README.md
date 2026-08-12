# 第 06 课 — 混合精度分析与调度验证

> **谜题：**如果自动广播操作变得更快了，这是否证明了预期的低精度kernel运行了？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/06-mixed-precision-profiling/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/06-mixed-precision-profiling/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

定时和调度是不同的声明。更快的自动重播区域显示了应用程序级别的效果；PyTorch 性能分析器事件标识了框架操作符；只有较低级别的跟踪才能证明原生kernel或Tensor Core的使用声明。良好的性能分析将这些证据级别分开，而不是用一个作为另一个的捷径。

## 阅读结果前，先做出预测

1. 预测哪些 PyTorch 操作事件应该围绕 BF16 矩阵乘法进行自动混合精度计算。
2. 解释为什么在比较 CUDA 计时之前需要进行warmup和同步。
3. 需要哪些额外的证据来声称特定的本地Tensor Corekernel？

## 1. 从具体的张量和状态开始

三层证据回答不同的问题：模型输出展示语义效果，框架操作展示图调度，原生kernel跟踪展示实际启动的实现。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 时钟差值和operator跟踪回答不同的问题。 |
| 2 | 热身移除了稳态样本的初始化和编译。 |
| 3 | PyTorch 操作符名称是比原生kernel名称更高层次的证据；当kernel身份重要时，请使用Nsight。 |

## 2. 推导机制

性能分析可以揭示转换、复制、GEMMs、启动次数和设备时间。热身是必要的，因为懒惰初始化、编译和分配器增长不是稳态执行。

GPU 启动是异步的：主机执行时间可以测量队列提交而不是设备完成。CUDA 事件时间戳在设备流中工作，但初始化、分配器增长、懒加载库加载和编译仍然可能污染早期样本。因此，一个可辩护的稳态数字需要指定warmup、同步、样本数量和分布统计。

追踪增加了因果关系。在框架层，事件如`aten::matmul`、`aten::mm`和转换揭示了操作图和意外转换。在原生层，kernel名称和硬件计数器揭示了片上实现、张量管道活动、占用率和带宽。各层回答互补问题；两者互不冗余。

## 3. 把理论转化为实验**实验：**分析自动转换 BF16GEMM withPyTorchProfiler and record the relevant operator events beside CUDA-event timing.

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 理论预期是autocast选择 BF16 用于一个合格的 GEMM。 |
| 候选方案 | 实际时间自动转换区域加上捕获的 PyTorch 操作事件 |
| 保持不变 | 2048×2048 形状，种子，GPU，五个warmup，十五个 CUDA 事件样本 |
| 测量 | 中位数/第90百分位延迟和选定框架operator名称 |
| 证据标签 | `pytorch-gpu` |

该笔记本将重复的 CUDA 事件计时与选定的 PyTorch 性能分析器事件配对，并明确不自行发明一个原生kernel名称。

### 代码导读

该笔记本展示了 BF16 的自动混合精度矩阵乘法，并记录了 PyTorch 性能分析器选择的事件。它单独测量了同一区域的 CUDA 事件。将跟踪收集置于计时样本之外，避免了将性能分析器的开销与正常延迟混淆。

结果模式调用事件`pytorch_operator_events`，而非`native_kernels`。这种命名是故意的：框架跟踪足以审计Python级别的路径，但不足以量化Tensor Core的占用情况。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| GEMM 形状 | 2048 × 2048 |
| 中位数 | 0.104416 ms |
| p90 | 0.106240 ms |
| 样本 | 15 |
| PyTorch operator事件 | {'count': 6, 'operator': 'aten::matmul'}, {'count': 6, 'operator': 'aten::to'}, {'count': 6, 'operator': 'aten::_to_copy'}, {'count': 6, 'operator': 'aten::copy_'}, {'count': 3, 'operator': 'aten::mm'} |

### 这些数字说明了什么

保存的运行测量了0.104416毫秒的中位数和0.106240毫秒的p90，共十五个样本。保留了五个相关的 PyTorch 操作事件。紧致的中位数到p90的差距表明在热身之后微基准测试是稳定的，而事件列表确认捕捉到了一个autocast/matmul路径。

这两个字段中没有任何信息标识一个SASSkernel或报告硬件利用率。因此，可以得出一个有限的结论，即应用路径和时间观察到了；原生调度声明仍然悬而未决。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/06-mixed-precision-profiling/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 使用两部分证明：控制时间以影响效果和使用性能分析器证据进行调度；在本地kernel声明时升级到Nsight。

### 验收与回滚门槛

首先不使用性能分析器进行时间测量，然后捕获一个短的对齐跟踪。仅命名实际观察到的级别：PyTorch 操作符，CUDA 核心，或端到端阶段。

### 这个结论可能如何失效

Profiler traces 可能会干扰时间，因此将经过分析的持续时间报告为生产延迟是风险的。相反，没有跟踪的时间可能会奖励一个意外的回退或缓存结果。其他陷阱包括缺少同步、时间张量分配以及仅选择最快样本。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/06-mixed-precision-profiling/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

在Nsight Systems中捕捉相同的操作，连接CPU启动、CUDA API和kernel时间线，然后使用Nsight Compute为选定的kernel的张量管道和内存指标。重复此操作，禁用自动混合精度，并使用不寻常的形状。构建一张表格，分别在列中记录墙钟效果、框架调度、原生kernel和硬件计数器。

## 证据边界

**证据标签:** [`pytorch-gpu`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [PyTorchAMP文档](https://docs.pytorch.org/docs/stable/amp.html)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [PyTorch 采样器文档](https://docs.pytorch.org/docs/stable/profiler.html)
- [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute 诊断指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
