<!-- ai-infra-puzzles-header:start -->
<p align="center">
  <img src="../../../assets/branding/logo.png" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
<!-- ai-infra-puzzles-header:end -->

# 第 28 课 — GPU 内存、并发和成本估算

> **谜题：**有多少请求符合 INT4 权重压缩，以及哪些隐含假设会使得这个数字无效？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/28-gpu-capacity-cost/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/28-gpu-capacity-cost/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

云成本始于内存可行性账本，但不能止于此。理想的权重位、未量化层、缩放元数据、每请求的KV缓存、工作区、碎片化、张量并行、吞吐量、利用率以及每小时的价格都决定了一个GPU是否可用且经济。

## 阅读结果前，先做出预测

1. 估计70B参数的理想 BF16 和 INT4 的重量（以GiB为单位）。
2. 为80层、8 KV头部、维度128以及8K上下文计算一个请求的KV缓存。
3. 预测理想 INT4 权重是否适合32,607 MiB RTX 5090 在10%预留后。

## 1. 从具体的张量和状态开始

容量使用总/可用HBM、重量和缩放字节、运行时预留、工作区、每请求的KV、碎片化、张量并行化以及流量上下文分布。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | 容量从运行时预留、权重、工作区和碎片允许后的可用内存开始。 |
| 2 | 每次请求的KV缓存取决于上下文和缓存dtype。 |
| 3 | 每条数据的成本还取决于达到的吞吐量和利用率，而不仅仅是GPU的价格。 |

## 2. 推导机制

第一个边界是`requests = floor((usable - weights - workspace) / KV_per_request)`。每单位成本取决于每小时价格除以已实现且质量批准的每小时token数。

重量字节从 `P·bits/8` 开始。每个请求的 KV 字节是 `2·L·S·Hkv·D·cache_bytes`，然后并发量乘以这个数值。安全储备应覆盖kernel、图捕获、分配器行为以及突发峰值，然后将剩余的字节除以每个请求的缓存。

即使内存配置也未能产生成本结果。每百万个token的成本取决于已实现的token/秒、利用率、批处理、电力/云价格、失败率和副本数量。该笔记本故意在不存在引擎吞吐量时停止在算术容量。

## 3. 把理论转化为实验

**实验：**从 RTX GPU 中读取实时的空闲内存并构建 BF16 与...对比 INT4 不分配模型的情况下，70B级模型的容量预测。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | 70B BF16 权重与 BF16 KV 缓存 |
| 候选方案 | 理想 INT4 权重与 BF16 或 INT8 KV 缓存 |
| 保持不变 | 70B参数，80层，8个KV头，头维度128，上下文8192，10%保留 |
| 测量 | 实时总/可用 GiB，权重 GiB，KV GiB/请求，是否适合，预计请求计数 |
| 证据标签 | `capacity-model` |

实验室使用实时 RTX 5090 内存对一个 70B 算术模型进行播种，但明确不分配或基准测试一个 70B 模型。

### 代码导读

该笔记本读取实时 RTX 5090 内存，计算三个计划，保留 10%，然后才计算请求容量。当权重已超出可用内存时，记录零而不是负数或乐观的并发。

INT4 术语明确表示理想状态：它排除了保留的高精度缩放、填充、嵌入/归一化、引擎和工作区。该标签防止算术运算被误认为是成功的模型加载。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| 实时总内存 | 31.358 GiB |
| BF16 权重投影 | 130.385 GiB |
| 理想 INT4 重量投影 | 32.596 GiB |
| BF16 每请求的KB | 2.500 GiB |
| INT8 每请求的KB | 1.250 GiB |
| 理想 INT4 单 GPU拟合 | 否 |

### 这些数字说明了什么

实时总内存为 31.358 GiB。BF16 权重预计为 130.385 GiB；理想的 INT4 仍然需要 32.596 GiB，已经大于总内存，并且相对于 10% 的储备量更大。BF16 KV 缓存为 2.5 GiB/请求，而 INT8 KV 为 1.25 GiB/请求，但每个单 GPU 计划都正确地返回了零请求，因为权重无法容纳。

KV压缩无法拯救未能通过权重匹配门限的基模型。因此，实际部署70B所需的进一步压缩/减少开销、多GPU分割、CPU卸载或不同GPU类别的选择，才能在讨论并发性之前进行。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/28-gpu-capacity-cost/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 使用范围和安全边际，然后用实际的发动机和交通分布验证选择的点。

### 验收与回滚门槛

使用范围和安全边际，然后用实际引擎的峰值、持续并发量、SLO、利用率和云计费单位进行验证。

### 这个结论可能如何失效

使用十进制GB而不是二进制GiB可能会在接近容量时产生误导性的余量。理想的四比特算术不包含元数据和高精度张量，而一个空闲进程中的空闲内存也不是引擎容量。在没有同等SLO的吞吐量和质量的情况下进行成本比较也是毫无意义的。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/28-gpu-capacity-cost/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

添加来自真实引擎的测量开销，包括张量并行分割/通信、碎片化以及批次依赖的工作空间。一旦模型加载完成，基准测试每秒持续的token数和每百万token的计算成本，在候选GPU计划之间保持相同的质量与p95延迟。

## 证据边界

**证据标签:** [`capacity-model`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM 缓存配置](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
