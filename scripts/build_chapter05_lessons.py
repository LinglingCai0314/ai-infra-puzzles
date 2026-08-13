#!/usr/bin/env python3
"""Build Chapter 05 bilingual lesson notes and executable notebooks."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from chapter05_content import COMMON_REFS, LESSONS
from markdown_header import render_markdown_header


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "05-triton-gpu-programming"
CHAPTER_ZH = ROOT / "chapters-zh" / "05-triton-gpu-programming"

EVIDENCE_EN = {
    "native-backend": "A named Triton or PyTorch CUDA path executed on the recorded GPU. The result applies to the printed shape, dtype, implementation, and software stack; internal hardware causes require profiler evidence.",
    "compatibility-probe": "The installed toolchain or API surface was inspected. An available symbol or source file is not reported as native performance on an unexecuted backend.",
    "capacity-model": "Measured values feed an explicit decision or traffic model. The model organizes evidence but does not replace target-backend execution.",
}
EVIDENCE_ZH = {
    "native-backend": "指定 Triton 或 PyTorch CUDA 路径已经在记录的 GPU 上执行。结果只适用于打印出的 shape、dtype、实现和软件栈；内部硬件因果仍需 profiler 证据。",
    "compatibility-probe": "实验检查了已安装工具链或 API 能力。符号可用或源码存在，不会被写成未执行 backend 的原生性能。",
    "capacity-model": "实测值进入显式决策或流量模型。模型用于组织证据，不能替代目标 backend 执行。",
}


def wrap(text: str, width: int = 96) -> str:
    output: list[str] = []
    paragraph: list[str] = []
    fenced = False

    def flush() -> None:
        if paragraph:
            output.extend(textwrap.wrap(" ".join(paragraph), width=width,
                                       break_long_words=False, break_on_hyphens=False))
            paragraph.clear()

    for raw in text.strip().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush(); output.append(line); fenced = not fenced
        elif fenced:
            output.append(line)
        elif not stripped:
            flush()
            if output and output[-1] != "": output.append("")
        elif stripped.startswith(("#", "|", ">", "- ", "[", "**", "<")) or re.match(r"^\d+\.\s", stripped):
            flush(); output.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(output).rstrip() + "\n"


def name(spec: dict[str, Any]) -> str:
    return f"{spec['no']:02d}-{spec['slug']}"


def directory(spec: dict[str, Any]) -> Path:
    return CHAPTER / name(spec)


def directory_zh(spec: dict[str, Any]) -> Path:
    return CHAPTER_ZH / name(spec)


def read_artifact(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = directory(spec) / "artifacts" / "rtx5090-result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def format_value(value: Any, kind: str) -> str:
    if value is None: return "not measured"
    if kind == "ms": return f"{float(value):.4f} ms"
    if kind == "scientific": return f"{float(value):.3e}"
    if kind == "float": return f"{float(value):,.4f}"
    if kind == "ratio": return f"{float(value):.3f}x"
    if kind == "int": return f"{int(value):,}"
    if kind == "bool": return "true" if bool(value) else "false"
    return str(value)


def result_table(spec: dict[str, Any], artifact: dict[str, Any] | None, lang: str) -> str:
    if artifact is None:
        return "_Run the notebook on CUDA to populate this table._" if lang == "en" else "_在 CUDA 环境运行 notebook 后，这里会回填结构化结果。_"
    rows = (["| Measured field | Checked-in value |", "|---|---:|"] if lang == "en"
            else ["| 测量字段 | 仓库记录值 |", "|---|---:|"])
    for en, zh, field, kind in spec["result_fields"]:
        rows.append(f"| {en if lang == 'en' else zh} | {format_value(artifact['metrics'].get(field), kind)} |")
    return "\n".join(rows)


def environment_line(artifact: dict[str, Any] | None, lang: str) -> str:
    if artifact is None: return "pending execution" if lang == "en" else "等待执行"
    env = artifact["environment"]
    return (f"{env['gpu']}; compute capability {env['compute_capability']}; PyTorch {env['torch']}; "
            f"CUDA runtime {env['cuda_runtime']}; Triton {env['triton']}; Python {env['python']}")


def references(spec: dict[str, Any]) -> str:
    return "\n".join(f"- [{COMMON_REFS[key][0]}]({COMMON_REFS[key][1]})" for key in spec["refs"])


def puzzle(spec: dict[str, Any], lang: str) -> str:
    if lang == "en":
        return f"When {spec['concept_en']} change together, which observation tells you whether the kernel, layout, toolchain, or hardware boundary is responsible?"
    return f"当{spec['concept_zh']}同时变化时，怎样判断原因来自 kernel、布局、工具链还是硬件边界？"


def baseline(spec: dict[str, Any], lang: str) -> str:
    if spec["no"] == 5:
        return "reviewed CUDA C++ source plus toolchain availability" if lang == "en" else "可审阅 CUDA C++ 源码与工具链可用性"
    if spec["no"] in {20, 21, 24}:
        return "documented installed capability" if lang == "en" else "文档化的已安装能力"
    return "named PyTorch CUDA/library or standard-grid path" if lang == "en" else "明确命名的 PyTorch CUDA/库函数或标准 grid 路径"


def candidate(spec: dict[str, Any], lang: str) -> str:
    if spec["no"] in {18, 22}:
        return "optimized framework path described in the experiment" if lang == "en" else "实验中明确描述的优化框架路径"
    return "reviewed Triton kernel or explicit model described below" if lang == "en" else "下文可审阅的 Triton kernel 或显式模型"


def mermaid(spec: dict[str, Any], lang: str) -> str:
    if lang == "en":
        return f'''```mermaid
flowchart LR
  A["Frozen input + contract"] --> B["{spec['concept_en']}"]
  B --> C["Triton candidate"]
  B --> D["CUDA / library control"]
  C --> E["correctness + samples"]
  D --> E
  E --> F["bounded decision"]
```'''
    return f'''```mermaid
flowchart LR
  A["固定输入与 contract"] --> B["{spec['concept_zh']}"]
  B --> C["Triton candidate"]
  B --> D["CUDA / 库 control"]
  C --> E["正确性 + 样本"]
  D --> E
  E --> F["有边界的决策"]
```'''


def readme_en(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    no = spec["no"]
    analysis = artifact.get("analysis_en", artifact.get("analysis", "")) if artifact else "Run the notebook to create the first structured RTX 5090 result."
    return render_markdown_header(directory(spec) / "README.md") + wrap(f'''# Lesson {no:02d} — {spec['title_en']}

> **Puzzle:** {puzzle(spec, 'en')}

[← Chapter 05](../README.md) · [中文本课](../../../chapters-zh/05-triton-gpu-programming/{name(spec)}/README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

This lesson isolates **{spec['concept_en']}**. The goal is not to turn every PyTorch operation
into custom code. It is to make one performance claim small enough that correctness, timing,
layout, compilation, and the comparison path can all be inspected. The source material supplies
the theory boundary; the retained lab converts that boundary into a falsifiable experiment.

## Predict before running

1. Predict which path will have the lower warm median and state the mechanism you expect.
2. Predict the awkward input, dtype, stride, or toolchain condition most likely to break the claim.
3. Write the observation that would make you keep the baseline instead of the candidate.

## 1. Build the mechanism

{spec['mechanism_en']}

Three reasoning anchors keep the explanation testable:

1. **Address and work mapping:** identify which program owns each output and which bytes it requests.
2. **Compiler boundary:** separate runtime values from compile-time meta-parameters and cache keys.
3. **Evidence boundary:** distinguish source inspection, native execution, numerical models, and profiler counters.

{mermaid(spec, 'en')}

## 2. Compare Triton with CUDA or the library path

| Question | Triton blocked program | CUDA / library control |
|---|---|---|
| Work mapping | A program evaluates compiler-visible tensor blocks | CUDA maps scalar threads explicitly; a library owns its internal mapping |
| Memory | Pointer tensors and masks express addresses | Thread indices or a documented library contract establish addresses |
| Tuning | `BLOCK`, `num_warps`, stages, specialization, and autotune | block geometry, templates, library algorithms, or architecture-specific code |
| Integration | Python JIT and direct tensor launch | compiled extension or framework/library call |
| Proof needed | correctness, warm samples, target identity, and profiler evidence | the same, plus a built CUDA toolchain for custom source |

{spec['risk_en']}

## 3. Turn theory into an experiment

**Experiment:** {spec['experiment_en']}

| Experimental role | Frozen definition |
|---|---|
| Baseline | {baseline(spec, 'en')} |
| Candidate | {candidate(spec, 'en')} |
| Held constant | input values, shape, dtype, output contract, timing helper, warmup policy, and target GPU |
| Correctness | compare against the named reference before interpreting latency |
| Measurements | two lesson-specific fields, maximum absolute error, full samples in JSON, and a Boolean gate |
| Evidence label | `{spec['evidence_label']}` |

The notebook imports the reviewed kernels from `scripts/chapter05_runtime.py`. That shared file
contains the actual `@triton.jit` functions; the notebook freezes the lesson number, records the
environment, runs one measured experiment, and writes the canonical JSON artifact.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** {environment_line(artifact, 'en')}.

{result_table(spec, artifact, 'en')}

### Interpretation

{analysis}

The table is deliberately small. Full timing samples, target identity, auxiliary byte or shape
fields, and the acceptance result remain in [`rtx5090-result.json`](artifacts/rtx5090-result.json)
so a reader can recompute summaries instead of trusting a rounded screenshot.

## 5. Make the bounded decision

> {spec['conclusion_en']}

This conclusion can fail when the deployment shape, dtype, stride, compiler version, target
architecture, concurrency, or surrounding graph differs. {spec['risk_en']} Reopen the decision
when any of those conditions changes or when a profiler contradicts the proposed mechanism.

## Worked review checklist

1. Verify output semantics before reading speed.
2. Confirm that the baseline is named rather than called only “CUDA.”
3. Keep cold compilation and warm device execution in separate fields.
4. Inspect samples and effect size; do not decide from one minimum.
5. State what was not executed, especially custom CUDA or another hardware backend.
6. Preserve a rollback path whenever the candidate becomes production code.

## Reproduce

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start {no} --end {no}
python3 scripts/build_chapter05_lessons.py
```

## Extend the puzzle

Repeat the experiment over at least one aligned shape, one awkward tail, and one non-contiguous
layout. If the result is performance-sensitive, capture a profiler trace locally and add only the
derived counter fields needed to test the mechanism. Stop when correctness fails; do not tune around
an unexplained numerical or address error.

## Evidence boundary

**Evidence label:** [`{spec['evidence_label']}`](../README.md#evidence-labels). {EVIDENCE_EN[spec['evidence_label']]}

## References

{references(spec)}
''')


def readme_zh(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    no = spec["no"]
    analysis = artifact.get("analysis_zh", "运行 notebook 后会生成第一份结构化 RTX 5090 结果。") if artifact else "运行 notebook 后会生成第一份结构化 RTX 5090 结果。"
    return render_markdown_header(directory_zh(spec) / "README.md") + wrap(f'''# 第 {no:02d} 课 — {spec['title_zh']}

> **问题：**{puzzle(spec, 'zh')}

[← 第 05 章](../README.md) · [English lesson](../../../chapters/05-triton-gpu-programming/{name(spec)}/README.md) · [实验 Notebook](../../../chapters/05-triton-gpu-programming/{name(spec)}/lab.ipynb) · [RTX 5090 结果](../../../chapters/05-triton-gpu-programming/{name(spec)}/artifacts/rtx5090-result.json)

## 为什么值得研究

本课只隔离研究**{spec['concept_zh']}**。目的不是把每个 PyTorch 操作都改成自定义代码，
而是把一个性能判断缩小到可以逐项检查：正确性、计时、布局、编译状态和对照路径都要
说清楚。理论材料提供问题边界，仓库中的实验则把它变成可以被推翻的判断。

## 运行前先预测

1. 预测哪条路径的 warm 中位延迟更低，并写出预期机制。
2. 预测最容易让结论失效的特殊 shape、dtype、stride 或工具链条件。
3. 写出什么观察会让你保留 baseline，而不是采用 candidate。

## 1. 建立机制

{spec['mechanism_zh']}

推理时抓住三个锚点：

1. **地址与工作映射：**明确哪个 program 负责哪个输出、实际请求哪些字节。
2. **编译边界：**分开运行期值、编译期 meta-parameter 与 cache key。
3. **证据边界：**区分源码检查、原生执行、数值模型和 profiler counter。

{mermaid(spec, 'zh')}

## 2. 对比 Triton 与 CUDA 或库函数路径

| 问题 | Triton blocked program | CUDA / 库 control |
|---|---|---|
| 工作映射 | 一个 program 计算编译器可见的 tensor block | CUDA 显式映射标量 thread；库函数内部映射由实现维护 |
| 访存表达 | pointer tensor 与 mask 共同描述地址 | thread index 或文档化 library contract 确定地址 |
| 调优入口 | `BLOCK`、`num_warps`、stage、特化与 autotune | block 几何、template、库算法或架构专用代码 |
| 集成方式 | Python JIT，直接接收 tensor | 编译扩展，或通过 framework/library 调用 |
| 所需证据 | 正确性、warm 样本、target 身份与 profiler | 同样的证据；自定义 CUDA 还必须真正完成工具链构建 |

{spec['risk_zh']}

## 3. 把理论变成实验

**实验：**{spec['experiment_zh']}

| 实验角色 | 固定定义 |
|---|---|
| Baseline | {baseline(spec, 'zh')} |
| Candidate | {candidate(spec, 'zh')} |
| 保持不变 | 输入值、shape、dtype、输出 contract、计时 helper、warmup 策略与目标 GPU |
| 正确性 | 先与明确命名的 reference 对比，再解释 latency |
| 测量内容 | 两个本课字段、最大绝对误差、JSON 内完整样本和 Boolean gate |
| 证据标签 | `{spec['evidence_label']}` |

Notebook 从 `scripts/chapter05_runtime.py` 导入已审阅 kernel。真正的 `@triton.jit` 函数
保存在这份共享文件中；Notebook 固定课号、记录环境、运行本课实验，并写出 canonical
JSON artifact。这样既避免三十份 kernel 漂移，也保留逐课复现入口。

## 4. 阅读仓库中的 RTX 5090 结果

**记录环境：**{environment_line(artifact, 'zh')}。

{result_table(spec, artifact, 'zh')}

### 如何解释结果

{analysis}

表格刻意只保留最关键字段。完整计时样本、target 身份、辅助字节或 shape 字段，以及
验收结果都在 [`rtx5090-result.json`](../../../chapters/05-triton-gpu-programming/{name(spec)}/artifacts/rtx5090-result.json)
中，读者可以重新计算摘要，而不是依赖一张四舍五入的截图。

## 5. 得出有边界的结论

> {spec['conclusion_zh']}

如果部署 shape、dtype、stride、编译器版本、目标架构、并发或周边 graph 改变，本结论
就可能失效。{spec['risk_zh']}出现这些变化，或 profiler 与预期机制冲突时，应重新打开
决策，而不是沿用旧数字。

## 审阅清单

1. 先验证输出语义，再阅读速度。
2. baseline 必须明确命名，不能只写成含糊的“CUDA”。
3. cold 编译与 warm 设备执行必须分列。
4. 查看样本与 effect size，不能用单次最小值决策。
5. 明确哪些路径没有执行，尤其是自定义 CUDA 或另一种硬件 backend。
6. candidate 进入生产时，必须保留 rollback 路径。

## 复现

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start {no} --end {no}
python3 scripts/build_chapter05_lessons.py
```

## 继续实验

至少补测一个对齐 shape、一个特殊 tail 和一个 non-contiguous 布局。如果结果对性能
敏感，可在本地采集 profiler trace，但只把验证机制所需的派生 counter 写入结果。
正确性一旦失败就应停止，不要围绕未解释的数值或地址错误继续调参。

## 证据边界

**证据标签：**[`{spec['evidence_label']}`](../README.md#证据标签)。{EVIDENCE_ZH[spec['evidence_label']]}

## 参考资料

{references(spec)}
''')


def markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(cell_id: str, source: str, old: dict[str, Any] | None) -> dict[str, Any]:
    cell = {"id": cell_id, "cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip() + "\n"}
    if old:
        old_source = old.get("source", "")
        if isinstance(old_source, list): old_source = "".join(old_source)
        if old_source == cell["source"]:
            cell["execution_count"] = old.get("execution_count")
            cell["outputs"] = old.get("outputs", [])
    return cell


def notebook(spec: dict[str, Any], artifact: dict[str, Any] | None,
             old_nb: dict[str, Any] | None) -> dict[str, Any]:
    no = spec["no"]
    old_code = {cell.get("id"): cell for cell in (old_nb or {}).get("cells", []) if cell.get("cell_type") == "code"}
    env_code = f'''from pathlib import Path
import json, sys

ROOT = Path.cwd().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from chapter05_runtime import environment, run_lesson

LESSON_NO = {no}
LESSON_TITLE = {spec['title_en']!r}
ENV = environment(LESSON_NO)
print(json.dumps(ENV, indent=2, ensure_ascii=False))'''
    experiment_code = '''metrics, analysis_en, analysis_zh = run_lesson(LESSON_NO)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print(analysis_en)'''
    artifact_code = f'''artifact = Path("artifacts/rtx5090-result.json")
artifact.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "lesson": LESSON_NO,
    "title": LESSON_TITLE,
    "environment": ENV,
    "evidence_label": {spec['evidence_label']!r},
    "metrics": metrics,
    "analysis_en": analysis_en,
    "analysis_zh": analysis_zh,
    "conclusion": {spec['conclusion_en']!r},
}}
artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))'''
    analysis = artifact.get("analysis_en", "Run all cells to create the structured result.") if artifact else "Run all cells to create the structured result."
    cells = [
        markdown(f"c05-l{no:02d}-title", f"# Lesson {no:02d} Lab — {spec['title_en']}\n\n**Puzzle:** {puzzle(spec, 'en')}\n\nThis notebook retains one complete RTX 5090 execution."),
        markdown(f"c05-l{no:02d}-why", f"## Why this matters\n\nThis lab isolates {spec['concept_en']} and keeps its comparison path explicit."),
        markdown(f"c05-l{no:02d}-predict", "## 0. Predict before running\n\nPredict correctness, warm latency ordering, and the first boundary case. Write what would disprove each prediction."),
        markdown(f"c05-l{no:02d}-theory", f"## 1. Theory and mechanism\n\n{spec['mechanism_en']}"),
        markdown(f"c05-l{no:02d}-map", f"## 2. Trace the mechanism\n\n{mermaid(spec, 'en')}"),
        markdown(f"c05-l{no:02d}-compare", f"## 3. Inspect the comparison boundary\n\nBaseline: {baseline(spec, 'en')}. Candidate: {candidate(spec, 'en')}.\n\n{spec['risk_en']}"),
        markdown(f"c05-l{no:02d}-env", "## 4. Inspect the execution environment\n\nThe next cell asserts CUDA and records GPU, target, PyTorch, CUDA runtime, Triton, Python, and seed."),
        code(f"c05-l{no:02d}-env-code", env_code, old_code.get(f"c05-l{no:02d}-env-code")),
        markdown(f"c05-l{no:02d}-protocol", f"## 5. Freeze the experiment\n\n**Experiment:** {spec['experiment_en']}\n\nInputs, output contract, timer, and target stay fixed across compared paths."),
        markdown(f"c05-l{no:02d}-inspect", "## 6. Inspect and execute the reviewed code\n\nThe next cell calls the shared reviewed kernel source, retains full samples in `metrics`, checks maximum error, and prints the bounded analysis."),
        code(f"c05-l{no:02d}-experiment-code", experiment_code, old_code.get(f"c05-l{no:02d}-experiment-code")),
        markdown(f"c05-l{no:02d}-results", f"## 7. Read the retained RTX 5090 result\n\n**Environment:** {environment_line(artifact, 'en')}.\n\n{result_table(spec, artifact, 'en')}"),
        markdown(f"c05-l{no:02d}-explain", f"## 8. Explain without overclaiming\n\n{analysis}\n\n{EVIDENCE_EN[spec['evidence_label']]}"),
        markdown(f"c05-l{no:02d}-artifact", "## 9. Write the canonical artifact\n\nThe next cell stores the environment, full metrics, bilingual analysis, evidence label, and bounded conclusion."),
        code(f"c05-l{no:02d}-artifact-code", artifact_code, old_code.get(f"c05-l{no:02d}-artifact-code")),
        markdown(f"c05-l{no:02d}-decision", f"## 10. Make the bounded decision\n\n> {spec['conclusion_en']}\n\n**Failure analysis:** {spec['risk_en']}"),
        markdown(f"c05-l{no:02d}-extend", "## 11. Extend and review\n\nAdd an awkward shape and non-contiguous layout. Stop on correctness failure. See `README.md` for references and the full review checklist."),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def phase_tables(lang: str) -> str:
    phases = [
        ("I", "Programming and measurement foundations", "编程与测量基础", LESSONS[:10]),
        ("II", "Core operators and resource trade-offs", "核心算子与资源权衡", LESSONS[10:17]),
        ("III", "Attention, stability, and integration", "Attention、稳定性与集成", LESSONS[17:23]),
        ("IV", "Portability and advanced scheduling", "可移植性与高级调度", LESSONS[23:27]),
        ("V", "CI, selection, and delivery", "CI、选型与交付", LESSONS[27:]),
    ]
    blocks = []
    for numeral, en, zh, items in phases:
        rows = (["| Lesson | Puzzle | Lab |", "|---:|---|---|"] if lang == "en"
                else ["| 课 | 问题 | 实验 |", "|---:|---|---|"])
        for spec in items:
            lesson_name = name(spec)
            if lang == "en": rows.append(f"| {spec['no']:02d} | [{spec['title_en']}]({lesson_name}/README.md) | [notebook]({lesson_name}/lab.ipynb) |")
            else: rows.append(f"| {spec['no']:02d} | [{spec['title_zh']}]({lesson_name}/README.md) | [notebook](../../chapters/05-triton-gpu-programming/{lesson_name}/lab.ipynb) |")
        blocks.append(f"## Phase {numeral} — {en}\n\n" + "\n".join(rows) if lang == "en" else f"## 阶段 {numeral} — {zh}\n\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def chapter_readme_en() -> str:
    return render_markdown_header(CHAPTER / "README.md") + wrap(f'''# Chapter 05 — Triton GPU Programming and CUDA Performance

[Project home](../../README.md) · [中文首页](../../README_ZH.md) · [中文本章](../../chapters-zh/05-triton-gpu-programming/README.md)

Chapter 05 is a 30-lesson path from Triton's blocked programming model to a deliverable custom
kernel. It independently reorganizes Linnea Cai's Triton GPU programming study material around
executable puzzles. Every theory topic gets a prediction, a named CUDA/library control, a reviewed
implementation, a correctness gate, retained samples, and a conclusion that states where it stops.

The checked-in runs use an NVIDIA GeForce RTX 5090, CUDA runtime 13.0, PyTorch 2.13.0, and Triton
3.7.1 targeting CUDA architecture 120. The execution host did not provide `nvcc`: Lesson 05 therefore
retains equivalent CUDA C++ source and records the toolchain as unavailable instead of inventing a
CUDA timing. PyTorch CUDA, cuBLAS-backed `torch.mm`, SDPA, and custom Triton paths are named
separately throughout the chapter.

```mermaid
flowchart LR
  A["blocked programs + masks"] --> B["memory + benchmark"]
  B --> C["Softmax + reduction + GEMM"]
  C --> D["Norm + Attention + stability"]
  D --> E["compile + paged KV + persistence"]
  E --> F["CI + selection + delivery"]
```

## How to study this chapter

1. Predict correctness and latency before opening retained output.
2. Read the baseline name: custom CUDA source, PyTorch CUDA, cuBLAS, SDPA, and a numerical model are not interchangeable.
3. Inspect full timing samples and the environment in the JSON artifact.
4. Re-run awkward tails and layouts before using a conclusion in another operator.
5. Keep a library or PyTorch rollback until the custom kernel passes its declared gate.

## Evidence labels

| Label | What it establishes |
|---|---|
| `native-backend` | A named Triton or PyTorch CUDA path executed on the recorded RTX 5090 stack |
| `compatibility-probe` | An installed API, backend target, source, or compiler capability was inspected without claiming unexecuted performance |
| `capacity-model` | Measured values feed a transparent traffic or decision model |

{phase_tables('en')}

## Shared implementation

The executable kernels live in [`scripts/chapter05_runtime.py`](../../scripts/chapter05_runtime.py).
Keeping one reviewed source prevents thirty notebooks from drifting while every lesson still has an
independent entry point and canonical result. Lesson 05 also contains
[`vector_affine.cu`](05-explicit-cuda-control/vector_affine.cu), the explicit CUDA control that can be
built when a local CUDA Toolkit is available.

## Reproduce and validate

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start 1 --end 30
python3 scripts/build_chapter05_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 05
python3 scripts/audit_chapter05_delivery.py
```
''')


def chapter_readme_zh() -> str:
    return render_markdown_header(CHAPTER_ZH / "README.md") + wrap(f'''# 第 05 章 — Triton GPU 编程与 CUDA 性能对比

[← 中文首页](../../README_ZH.md) · [English chapter](../../chapters/05-triton-gpu-programming/README.md)

第 05 章共 30 课，从 Triton blocked programming model 一直讲到可交付自定义 kernel。
课程以 Linnea Cai 的 Triton GPU 编程学习材料为理论底稿，重新组织成可执行 puzzle。
每个主题都包含预测、明确命名的 CUDA/库函数 control、可审阅实现、正确性 gate、完整
计时样本，以及说明适用边界的结论。

仓库结果来自 NVIDIA GeForce RTX 5090、CUDA runtime 13.0、PyTorch 2.13.0 和 Triton
3.7.1，目标是 CUDA architecture 120。执行环境没有 `nvcc`，因此第 05 课保留等价 CUDA
C++ 源码，并把工具链记录为不可用，不虚构 CUDA latency。全章会分别标注 PyTorch CUDA、
cuBLAS-backed `torch.mm`、SDPA 和自定义 Triton 路径，避免把不同 baseline 混成“CUDA”。

```mermaid
flowchart LR
  A["blocked program + mask"] --> B["访存 + benchmark"]
  B --> C["Softmax + reduction + GEMM"]
  C --> D["Norm + Attention + 稳定性"]
  D --> E["compile + paged KV + persistence"]
  E --> F["CI + 选型 + 交付"]
```

## 学习方法

1. 先预测正确性与 latency，再打开保留输出。
2. 先看 baseline 名称：自定义 CUDA 源码、PyTorch CUDA、cuBLAS、SDPA 与数值模型不能互换。
3. 在 JSON artifact 中检查完整计时样本与环境身份。
4. 把结论迁移到其他算子前，先重跑特殊 tail 与布局。
5. 自定义 kernel 通过声明 gate 前，始终保留库函数或 PyTorch rollback。

## 证据标签

| 标签 | 能说明什么 |
|---|---|
| `native-backend` | 指定 Triton 或 PyTorch CUDA 路径在记录的 RTX 5090 软件栈上执行 |
| `compatibility-probe` | 检查 API、backend target、源码或编译器能力，不声称未执行路径的性能 |
| `capacity-model` | 实测值进入透明的流量或决策模型 |

{phase_tables('zh')}

## 共享实现

可执行 kernel 统一保存在 [`scripts/chapter05_runtime.py`](../../scripts/chapter05_runtime.py)。
一份审阅源码可以避免 30 个 Notebook 相互漂移，同时每课仍有独立入口与 canonical result。
第 05 课还提供 [`vector_affine.cu`](../../chapters/05-triton-gpu-programming/05-explicit-cuda-control/vector_affine.cu)，
在本地 CUDA Toolkit 可用时可以构建这条显式 CUDA control。

## 复现与验证

```bash
python3 -m pip install -r requirements-triton.txt
python3 scripts/execute_chapter_notebooks.py --chapter 05 --start 1 --end 30
python3 scripts/build_chapter05_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 05
python3 scripts/audit_chapter05_delivery.py
```
''')


CUDA_SOURCE = r'''#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do {                                              \
  cudaError_t status = (call);                                             \
  if (status != cudaSuccess) {                                             \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                 cudaGetErrorString(status));                              \
    return 1;                                                              \
  }                                                                        \
} while (0)

__global__ void vector_affine(const float* x, float* y, int n,
                              float scale, float bias) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) y[index] = x[index] * scale + bias;
}

int launch_vector_affine(const float* x, float* y, int n,
                         float scale, float bias, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_affine<<<blocks, threads, 0, stream>>>(x, y, n, scale, bias);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}
'''


def build_chapter(*, refresh_chapter_readme: bool = False) -> None:
    CHAPTER.mkdir(parents=True, exist_ok=True)
    CHAPTER_ZH.mkdir(parents=True, exist_ok=True)
    for spec in LESSONS:
        path = directory(spec); path_zh = directory_zh(spec)
        (path / "artifacts").mkdir(parents=True, exist_ok=True); path_zh.mkdir(parents=True, exist_ok=True)
        artifact = read_artifact(spec)
        notebook_path = path / "lab.ipynb"
        old_nb = json.loads(notebook_path.read_text(encoding="utf-8")) if notebook_path.exists() else None
        (path / "README.md").write_text(readme_en(spec, artifact), encoding="utf-8")
        (path_zh / "README.md").write_text(readme_zh(spec, artifact), encoding="utf-8")
        notebook_path.write_text(json.dumps(notebook(spec, artifact, old_nb), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory(LESSONS[4]) / "vector_affine.cu").write_text(CUDA_SOURCE, encoding="utf-8")
    if refresh_chapter_readme or not (CHAPTER / "README.md").exists():
        (CHAPTER / "README.md").write_text(chapter_readme_en(), encoding="utf-8")
        (CHAPTER_ZH / "README.md").write_text(chapter_readme_zh(), encoding="utf-8")
    print(f"Built {len(LESSONS)} Chapter 05 bilingual lessons and notebooks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-readme", action="store_true")
    args = parser.parse_args()
    build_chapter(refresh_chapter_readme=args.chapter_readme)


if __name__ == "__main__":
    main()
