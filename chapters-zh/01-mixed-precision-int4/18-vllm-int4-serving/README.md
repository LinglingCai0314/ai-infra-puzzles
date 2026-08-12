# 第 18 课 — 使用 vLLM 提供 INT4

> **谜题：**如果一个检查点说AWQ或者GPTQ, 将会 vLLM 是否必须在当前GPU上高效地运行它？

[← 第 01 章](../README.md) · [项目主页](../../../README_ZH.md) · [执行的笔记本](../../../chapters/01-mixed-precision-int4/18-vllm-int4-serving/lab.ipynb) · [RTX 5090 结果](../../../chapters/01-mixed-precision-int4/18-vllm-int4-serving/artifacts/rtx5090-result.json)

## 为什么这个谜题重要

服务性能属于运行时，而不是检查点标签。vLLM 结合了量化线性kernel、调度、连续批处理、分页 KV 缓存、前缀缓存和请求分发。PyTorch 微基准测试可以警告形状敏感性，但不能代替 vLLM 服务器每秒请求数或延迟百分位数。

## 阅读结果前，先做出预测

1. 分离检查点格式支持、硬件支持、kernel调度和负载性能。
2. 预测参考 W4 去量化的矩阵路径在每个测试批次中是否获胜。
3. 设计一个服务工作负载，分别报告TTFT和跨token延迟。

## 1. 从具体的张量和状态开始

vLLM 服务结合了检查点格式、量化后端、模型运行器、调度器、分页的KV缓存、CUDA 图表、请求批处理和采样。线性kernel延迟只是其中一个组成部分。

### 三个推理锚点

| # | 本课需要牢记的判断 |
|---:|---|
| 1 | vLLM 通过变化的模型格式和硬件兼容性矩阵选择量化kernel。 |
| 2 | 服务性能包括调度、KV缓存、批量处理和请求分发——不仅包括线性层。 |
| 3 | 一个导入探针不能替代服务器基准测试。 |

## 2. 推导机制

Prefill成本随着提示工作增加，因为解码重复处理小的标记步骤并读取KV缓存。持续的批处理通过组合请求来提高利用率，但排队会改变首次标记时间和尾部延迟。

Prefill 和 Decode 生成的矩阵形状不同，并且与批处理的交互方式也不同。服务吞吐量还取决于到达率、提示/输出长度、调度策略、缓存容量以及队列。一个仅加载权重的检查点即使成功加载，也可能在某些层中退回到慢kernel，或者在长上下文时失去其对 KV 缓存的记忆优势。

接受链的顺序是格式元数据 → 模型加载 → 量化模块/操作符跟踪 → 输出质量 → 受控请求负载 → 延迟/吞吐量/容量。一个导入探针仅到达第一个兼容边缘。

## 3. 把理论转化为实验

**实验：**探测 vLLM 可用性和基准测试一个小 PyTorchW4-解量化矩阵乘法跨批次大小作为后端无关的形状警告。

| 实验角色 | 固定定义 |
|---|---|
| 基准 | BF16 PyTorch 批次矩阵路径1, 8, 和32 |
| 候选方案 | 参考去量化的 W4 矩阵路径具有相同的形状 |
| 保持不变 | 权重/输入形状，GPU，热身，重复次数；无需服务器或调度器 |
| 测量 | 操作中位数/第90百分位数（p90）按批次加上 vLLM 安装和服务基准状态 |
| 证据标签 | `compatibility-probe` |

实验室记录 vLLM 的可用性，并仅使用 PyTorch 的批量形状时间作为警告；它将 vLLM 的服务吞吐量标记为`not_measured`。

### 代码导读

该笔记本检查 vLLM 的可用性，然后运行一个不依赖后端的 PyTorch 形状实验。W4 候选者是一个去量化参考张量，因此它测试的是结果矩阵形状的行为，而不是 vLLM 的 AWQ/GPTQ 核心。结果存储在 `pytorch_shape_warning` 下，以便使该边界可见。

一个真正的服务单元会启动一个服务器，等待就绪状态，发出一个冻结请求跟踪，收集TTFT/ITL/延迟百分位数和吞吐量，然后干净地终止。这里没有合成这些内容。

## 4. 解读仓库内的 RTX 5090 实测结果

**录制环境：**NVIDIA GeForce RTX 5090 计算能力12.0; PyTorch 2.12.0; CUDA 运行时13.0.

| 实测字段 | 已提交值 |
|---|---:|
| vLLM 已安装 | 否 |
| 服务基准测试 | not_measured |
| 批量计算 1 的中位数 BF16。 | 0.019520 ms |
| 批量 1 参考 W4 中位数 | 0.019424 ms |
| 批量计算 32 的中位数 BF16。 | 0.018976 ms |
| 批量 32 参考 W4 中位数 | 0.019072 ms |

### 这些数字说明了什么

微小矩阵探针产生的中位数几乎相等：在批次1中，BF16 为0.019520毫秒，参考W4-dequant张量0.019424为毫秒；在批次8中，它们分别为0.019168和0.018912毫秒；在批次32中，候选者稍微反转，0.019072为0.018976毫秒。vLLM 未安装，服务性能明确为`not_measured`。

这种微秒级的差异不是服务结果。它们表明形状可以反转小操作符比较，并且进一步证明了需要一个完整的请求工作负载。

打开[`artifacts/rtx5090-result.json`](../../../chapters/01-mixed-precision-int4/18-vllm-int4-serving/artifacts/rtx5090-result.json)，当需要每个重复样本或教程表中未选中的字段时。

## 5. 解答谜题并做出决策

> 通过 vLLM 和 INT4 路径之前，必须先通过检查点格式、硬件、负载、operator、质量和服务负载门。

### 验收与回滚门槛

通过格式/硬件负载、operator、质量、TTFT、TPOT/跨token延迟、吞吐量、p90/p99、峰值内存和持续并发门限，使用冻结请求分布。

### 这个结论可能如何失效

将此表报告为 vLLM 速度会错误地标记后端并忽略调度。其他陷阱包括基准测试一个暖缓存提示，混合不同的模型修订版本，省略输出长度，并在不等的延迟或质量下比较吞吐量。量化兼容性矩阵在不同版本中也会发生变化，因此必须锁定确切的发布版本。

## 复现

从仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/18-vllm-int4-serving/lab.ipynb
```

使用**运行所有**并与已提交的文件进行比较。

## 扩展实验

在单独的环境中安装受支持的 vLLM 版本，加载一个已文档化的AWQ或GPTQ模型，确认模块/operator的选择，并使用固定提示/输出分布和并发运行`vllm bench serve`。报告TTFT的p50/p95，ITL，端到端延迟，tokens/s，GPU 内存，以及被拒绝的请求。

## 证据边界

**证据标签:** [`compatibility-probe`](../../../chapters/01-mixed-precision-int4/README.md#evidence-labels).

## 参考资料

- [vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM 基准 CLI](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
