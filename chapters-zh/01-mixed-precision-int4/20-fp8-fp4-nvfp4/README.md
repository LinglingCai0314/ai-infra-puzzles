# 第 20 课 — FP8, FP4, NVFP4 和硬件边界

> **谜题：**Blackwell硬件支持是否意味着每个框架构建都暴露相同的内容？FP8 或者NVFP4路径?

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

一个dtype名称可以在四个层次存在：数学格式、硬件指令、库配方和框架操作符。Blackwell 支持不保证安装的 PyTorch、Transformer Engine、TensorRT 或 ModelOpt 构建会暴露相同的 FP8 或 NVFP4 路径。每一层必须独立探测。

## 阅读结果前，先做出预测

1. 区分 E4M3 FP8 与 E5M2，并将普通 INT4 与块级 NVFP4 区分开来。
2. 预测已安装的 PyTorch 构建是否可以执行缩放的 FP8 矩阵乘法。
3. 在声称NVFP4性能之前，请说明还需要什么额外的证据。

## 1. 从具体的张量和状态开始

保持四个层次的独立：数值格式、硬件指令、库食谱和框架/operatorAPI。`torch.float8_*`现有的并不单独证明 FP8 GEMM 路径。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 格式定义、硬件指令、库API和框架kernel是四个独立的层次。 |
| 2 | FP8 变体交易指数范围与分数精度。 |
| 3 | NVFP4添加了块缩放；这不是普通的均匀缩放。INT4. |

## 2. 推导机制

E4M3 更注重精度而范围较小；E5M2 更注重范围。缩放 FP8 矩阵乘法应用显式缩放因子。针对Blackwell的MXFP8/NVFP4加法块结构化并需要匹配的食谱和kernel。

FP8E4M3 分配了四个指数位和三个分数位在符号位之后，以换取范围；E5M2 另外花费了一个位在范围上。NVFP4使用FP4E2M1值采用块缩放，因此其真实表示包括四比特数据和缩放层次结构。TensorRT's current scheme uses block size16forNVFP4, 而框架API和受支持的轴仍然保持版本特定。

缩放矩阵乘法也需要选择输入和输出缩放。一个成功的`torch._scaled_mm`调用证明了一个框架级别的路径，适用于一个特定的形状和格式；但它不能证明Transformer Engine食谱或TensorRT NVFP4kernel。

## 3. 把理论转化为实验

**实验：**尝试原生 PyTorch FP8GEMM 在 RTX GPU 上运行时，记录支持时的误差和时间，并分别探测 Transformer Engine 和NVFP4APIs.

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 高精度参考矩阵乘法用于误差比较 |
| 候选方案 | PyTorch 缩放 FP8E4M3 GEMM on RTX 5090 |
| 保持不变 | 1024 类矩阵形状，缩放程序，热身，十五个定时样本 |
| 测量 | API 成功，RMSE/cosine，中位数/p90，库可用性，NVFP4 状态 |
| 证据标签 | `pytorch-gpu` |

实验室调用 PyTorch 缩放的 FP8 matmul，当可用时，并将Transformer Engine/NVFP4留作未测量，而不是将硬件生成与框架支持等同。

### 代码导读

该笔记本检查float8dtype支持，并调用`torch._scaled_mm`，带有明确的缩放因子。它将输出与更高精度的参考值进行比较，并测量重复执行 CUDA 的时间。单独的探针记录Transformer Engine的可用性，并在其配方/操作不可用时保留NVFP4和`not_measured`。

该设计防止真实 FP8 结果被泛化为不同的格式。JSON指定了确切的API，以便将来可以检测到软件变更。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| PyTorch API | torch._scaled_mm |
| FP8 GEMM | 成功 |
| 中位数 | 0.017568 ms |
| FP8 RMSE | 1.208455 |
| Transformer Engine 已安装 | 否 |
| NVFP4后端 | not_measured |

### 这些数字说明了什么

通过`torch._scaled_mm`，缩放的 FP8 GEMM 成功完成，中位数为0.017568 ms，p90为0.018560 ms，共十五个样本。输出余弦值为0.999285，缩放和形状的测试值为 RMSE 1.208455。未安装Transformer Engine，NVFP4 保持为`not_measured`。

因此，测量路径是 PyTorch GPU的实测证据，而不是 FP8 的Transformer Engine或NVFP4后端的证明。绝对误差也说明了格式支持必须与缩放和质量政策相匹配的原因。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 发布一个硬件-库格式矩阵，而不是单个`supported`复选框。

### 验收与回滚门槛

记录 FP8、MXFP8 和 NVFP4 的计算能力、dtype/API、缩放食谱、操作符成功、数值误差、时间以及库版本分别。

### 这个结论可能如何失效

将张量转换为float8dtype而不成功地进行矩阵操作仅证明了存储。将原始 FP8 延迟与不同形状进行比较或排除缩放计算可能会错误地表示速度。将NVFP4视为带符号的均匀 INT4 会完全失去其块级语义。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/20-fp8-fp4-nvfp4/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

在隔离环境下安装匹配的Transformer Engine或TensorRT堆栈，运行文档中描述的 FP8 和NVFP4食谱，并捕获operator身份、缩放粒度、端到端缩放开销、错误和延迟。构建一个矩阵，行代表格式，列代表硬件、库、API、operator和测试状态。

## 证据边界

**证据标签:** [`pytorch-gpu`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [NVIDIA Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TensorRT量化方案](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [TensorRT动态量化操作符](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/operators/DynamicQuantize.html)
