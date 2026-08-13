<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# Lesson 29 — 自定义kernel：打包、去量化和CUTLASS边界

> **谜题：**何时是 INT4 打包/去量化kernel值得构建，而不是使用现有的后端吗？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/29-custom-int4-kernels/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/29-custom-int4-kernels/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

自定义 INT4kernel只有在从端到端路径中移除工作时才会变得复杂。压缩权重是有帮助的，但在调用之前将一个完整的去量化矩阵显式化是不必要的。BF16GEMM 添加读取、写入、转换和启动。目标是融合加载-解包-缩放-MMA-尾部路径，支持特定的tile布局。

## 阅读结果前，先做出预测

1. 计算4096×4096 INT4 权重矩阵的压缩字节数。
2. 预测由解包/去量化/GEMM 路径组成的延迟相对于直接的 BF16 GEMM 在M=32时的延迟。
3. 在调用CUTLASS或自定义kernel结果之前，请命名所需的证据。

## 1. 从具体的张量和状态开始

INT4 执行路径包含打包/存储、缩放加载、解包/去量化、GEMM、尾部处理、启动以及与框架布局和流的集成。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 端到端增益包括解包、缩放加载、去量化、GEMM、启动开销和集成成本。 |
| 2 | 一个Python或由 PyTorch 组成的原型验证语义，但不是融合的CUTLASSkernel。 |
| 3 | 目标形状分布决定了专业化是否值得。 |

## 2. 推导机制

端到端预算为`T = T_pack/load + T_dequant + T_gemm + T_epilogue + overhead`。融合阶段可以去除中间流量；复合 PyTorch 参考故意暴露未融合的成本。

逻辑管道包含以下步骤：全局加载 → 字节提取/扩展符号 → scale 加载 → 去量化片段 → 矩阵乘法/累积 → 尾部处理。如果去量化将一个完整的 BF16 矩阵写入全局内存，路径将支付打包读取和一个大的材料化写入/读取，然后进行 GEMM。融合将重建值保存在寄存器/片段中，并在tile上摊销缩放工作。

kernel盈利能力取决于 M、N、K、组大小、内存对齐、寄存器压力、占用率和尾部融合。语义 PyTorch 组合是一个正确性基准和上限警告，而不是自定义kernel。

## 3. 把理论转化为实验

**实验：**验证向量化 INT4 字节打包/解包并测量组合时间 PyTorch 去量化加矩阵乘法路径对比 BF16.

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 直接使用 BF16 形式的 GEMM 对于形状 M=32, K=N=4096。 |
| 候选方案 | PyTorch-composed 解包，符号恢复，去量化和 GEMM |
| 保持不变 | 相同的 X/W 值，组大小 128，压缩布局，GPU 时序辅助 |
| 测量 | 压缩字节，BF16 中位数/p90，合成中位数/p90，实现身份 |
| 证据标签 | `pytorch-gpu` |

实验室验证了字节语义，并对一个由解码-去量化-矩阵乘法组成的参考进行了计时，明确标记为非融合和非CUTLASS。

### 代码导读

该笔记本每字节包含两个代码，重建带符号代码，应用块缩放，实现 BF16 权重，并进行乘法。它测量整个复合函数的时间，而不是仅测量最终的 GEMM。结果字段明确显示`not fused CUTLASS`。

这提供了一个可读的语义参考，用于测试未来的 CUDA/Triton/CUTLASS实现。未来的kernel必须匹配其输出，同时消除材料化并减少启动次数。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| 形状 M×K×N | 32 × 4096 × 4096 |
| 压缩代码字节 | 8,388,608 字节 |
| BF16 中位数 | 0.027136 ms |
| 组成的路径中位数 | 0.328720 ms |
| 实施 | 组成 PyTorch 参考，而非融合 CUTLASS。 |

### 这些数字说明了什么

压缩存储为8,388,608字节，用于16,777,216权重，每个代码在缩放前为0.5字节。直接的 BF16 GEMM 的中位数时间为0.027136毫秒。组合的解压缩/去量化/矩阵乘法路径耗时0.328720毫秒——大约是12.1倍慢。

减速不是 INT4 硬件慢的证据。它是未融合参考执行过多整合工作和内存流量的证据。它确定了优化目标和正确性验证器。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/29-custom-int4-kernels/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 当现有后端缺少一个重要的、重复的形状，且可恢复的端到端预算超过集成成本时，构建自定义代码。

### 验收与回滚门槛

首先找到重复的形状级间隙，验证打包/解量化语义，分析屋顶线和内存流量，然后实现，最后要求在目标形状分布上实现端到端的收益和质量。

### 这个结论可能如何失效

只在测量之外将预去量化后的 GEMM 操作延迟到时间上隐藏了主要成本。将Python组合称为自定义kernel是不正确的。融合kernel也可能退化，如果寄存器压力降低导致占用率下降或不支持的形状回退，因此形状覆盖和调度必须进行审计。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/29-custom-int4-kernels/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

在CUTLASS、CUDA 或Triton中实现一个最小的融合kernel，针对一个冻结形状。验证压缩布局兼容性和数值一致性，然后在M值上评估指令混合、全局字节数、占用率、张量管道利用率和端到端延迟。为不支持的形状添加一个安全的后备方案。

## 证据边界

**证据标签:** [`pytorch-gpu`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [TensorRT量化方案](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [CUTLASS 文档](https://docs.nvidia.com/cutlass/latest/overview.html)
- [CUTLASS 仓库](https://github.com/NVIDIA/cutlass)
