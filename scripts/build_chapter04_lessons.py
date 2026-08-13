#!/usr/bin/env python3
"""Build Chapter 04 bilingual notes and executable notebooks."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from chapter04_content import COMMON_REFS, LESSONS
from chapter04_experiments import ENV_CODE, EXPERIMENTS
from markdown_header import render_markdown_header


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "04-gpu-hardware-foundations"
CHAPTER_ZH = ROOT / "chapters-zh" / "04-gpu-hardware-foundations"

EVIDENCE_EN = {
    "pytorch-gpu": "CUDA work executed through PyTorch. It does not identify an internal instruction, cache event, or proprietary hardware block without additional profiler evidence.",
    "numerical-model": "A transparent mechanism model executed. It establishes the stated relationship under printed assumptions, not native hardware latency, energy, or topology.",
    "capacity-model": "Measured environment facts feed explicit capacity or Roofline arithmetic. Declared hierarchy and resource fields remain assumptions until native counters confirm them.",
    "compatibility-probe": "Repository artifacts and installed surfaces were inspected. Schema or API availability is not equivalent to validating an experiment's causal conclusion.",
}

EVIDENCE_ZH = {
    "pytorch-gpu": "CUDA 工作通过 PyTorch 执行。没有额外 profiler 证据时，它不能定位内部指令、cache event 或私有硬件模块。",
    "numerical-model": "运行的是透明机制模型。它只在打印出的假设下证明所述关系，不代表原生硬件延迟、能耗或拓扑。",
    "capacity-model": "实测环境事实进入显式容量或 Roofline 计算。层级和资源字段仍是声明的假设，需原生 counter 才能确认。",
    "compatibility-probe": "实验检查仓库 artifact 与已安装接口。schema 或 API 可用不等于实验因果结论已经成立。",
}


def wrap(text: str, width: int = 92) -> str:
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
        elif stripped.startswith(("#", "|", ">", "- ", "[", "**", "![", "<")) or re.match(r"^\d+\.\s", stripped):
            flush(); output.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(output).rstrip() + "\n"


def lesson_name(spec: dict[str, Any]) -> str:
    return f"{spec['no']:02d}-{spec['slug']}"


def lesson_dir(spec: dict[str, Any]) -> Path:
    return CHAPTER / lesson_name(spec)


def lesson_dir_zh(spec: dict[str, Any]) -> Path:
    return CHAPTER_ZH / lesson_name(spec)


def get_metric(metrics: dict[str, Any], dotted: str) -> Any:
    value: Any = metrics
    for part in dotted.split("."):
        value = value[part]
    return value


def format_value(value: Any, kind: str) -> str:
    if value is None: return "not measured"
    if kind == "int": return f"{int(value):,}"
    if kind == "percent": return f"{float(value):.2%}"
    if kind == "ms": return f"{float(value):.3f} ms"
    if kind == "float": return f"{float(value):,.4f}"
    if kind == "ratio": return f"{float(value):.3f}x"
    if kind == "mib": return f"{float(value):,.3f} MiB"
    if kind == "bytes": return f"{int(value):,} bytes"
    return str(value)


def read_artifact(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = lesson_dir(spec) / "artifacts" / "rtx5090-result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def result_table(spec: dict[str, Any], artifact: dict[str, Any] | None, lang: str) -> str:
    if artifact is None:
        return ("_Run the notebook on CUDA to populate this checked-in result table._" if lang == "en"
                else "_在 CUDA 环境运行 notebook 后，这里会自动回填本章的结构化结果。_")
    if lang == "en":
        rows = ["| Measured field | Checked-in value |", "|---|---:|"]
    else:
        rows = ["| 测量字段 | 仓库记录值 |", "|---|---:|"]
    for label_en, label_zh, path, kind in spec["result_fields"]:
        try: value = get_metric(artifact["metrics"], path)
        except (KeyError, TypeError): value = None
        rows.append(f"| {label_en if lang == 'en' else label_zh} | {format_value(value, kind)} |")
    return "\n".join(rows)


def environment_line(artifact: dict[str, Any] | None, lang: str) -> str:
    if artifact is None: return "pending execution" if lang == "en" else "等待执行"
    env = artifact["environment"]
    return (f"{env['gpu']}; compute capability {env['compute_capability']}; "
            f"PyTorch {env['torch']}; CUDA runtime {env['cuda_runtime']}; Python {env['python']}")


def analysis_zh(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "运行 notebook 后会生成第一份结构化 RTX 5090 结果。"
    values = []
    for _, label_zh, path, kind in spec["result_fields"][:3]:
        try: value = get_metric(artifact["metrics"], path)
        except (KeyError, TypeError): continue
        values.append(f"{label_zh}：{format_value(value, kind)}")
    summary = "，".join(values)
    return (f"本次记录的关键结果是：{summary}。这些数值只适用于上方记录的 GPU、"
            f"软件栈、shape 与测量协议。结合本课的证据边界，结论是：{spec['conclusion_zh']}")


def references(spec: dict[str, Any]) -> str:
    return "\n".join(f"- [{COMMON_REFS[key][0]}]({COMMON_REFS[key][1]})" for key in spec["refs"])


def visual_block(spec: dict[str, Any], lang: str, *, notebook: bool = False) -> str:
    if not spec["visuals"]:
        return ("This lesson is driven by a Mermaid mechanism map and executable measurements."
                if lang == "en" else "本课以 Mermaid 机制图和可执行测量为主。")
    prefix = ("../../../chapters/04-gpu-hardware-foundations/"
              if lang == "zh" else "../")
    lines = []
    for asset, caption_en, caption_zh in spec["visuals"]:
        caption = caption_en if lang == "en" else caption_zh
        href = prefix + asset
        if Path(asset).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            lines.extend([f"![{caption}]({href})", ""])
        else:
            lines.append(f"- [{caption}]({href})")
    boundary = ("These are conceptual teaching diagrams. They explain the named data path and are not die-accurate schematics of a particular commercial GPU."
                if lang == "en" else "这些图用于解释数据路径，是概念性教学图，不是某款商业 GPU 的 die-accurate 电路图。")
    lines.extend(["", boundary])
    return "\n".join(lines).strip()


def mermaid_guide(spec: dict[str, Any], lang: str) -> str:
    title = "Mechanism map" if lang == "en" else "机制图"
    return f"### {title}\n\n```mermaid\n{spec['mermaid']}\n```"


def extra_section(spec: dict[str, Any], lang: str) -> str:
    return spec.get(f"extra_{lang}", "")


def readme_en(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    predictions = "\n".join(f"{i}. {item}" for i, item in enumerate(spec["predictions_en"], 1))
    anchors = "\n".join(f"| {i} | {item} |" for i, item in enumerate(spec["anchors_en"], 1))
    analysis = artifact["analysis"] if artifact else "Run the notebook to create the first structured RTX 5090 result."
    no = spec["no"]
    return render_markdown_header(lesson_dir(spec) / "README.md") + wrap(f'''# Lesson {no:02d} — {spec['title_en']}

> **Puzzle:** {spec['puzzle_en']}

[← Chapter 04](../README.md) · [中文本课](../../../chapters-zh/04-gpu-hardware-foundations/{lesson_name(spec)}/README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

{spec['hook_en']}

## Predict before running

{predictions}

## 1. Put the mechanism in physical space

{spec['mechanism_en']}

| # | Reasoning anchor |
|---:|---|
{anchors}

{mermaid_guide(spec, 'en')}

## 2. Read the visual

{visual_block(spec, 'en')}

{extra_section(spec, 'en')}

## 3. Turn theory into an experiment

**Experiment:** {spec['experiment_en']}

| Experimental role | Frozen definition |
|---|---|
| Baseline | {spec['baseline_en']} |
| Candidate | {spec['candidate_en']} |
| Held constant | {spec['controlled_en']} |
| Measurements | {spec['metrics_en']} |
| Evidence label | `{spec['evidence_label']}` |

### Code walk-through

{spec['code_en']}

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** {environment_line(artifact, 'en')}.

{result_table(spec, artifact, 'en')}

### What the result means

{analysis}

## 5. Make the bounded decision

> {spec['conclusion_en']}

### How this conclusion can fail

{spec['failure_en']}

## Reproduce

```bash
python3 -m pip install -r requirements-gpu-foundations.txt
python3 scripts/execute_chapter_notebooks.py --chapter 04 --start {no} --end {no}
python3 scripts/build_chapter04_lessons.py
```

## Extend the experiment

{spec['extend_en']}

## Evidence boundary

**Evidence label:** [`{spec['evidence_label']}`](../README.md#evidence-labels). {EVIDENCE_EN[spec['evidence_label']]}

## References

{references(spec)}
''')


def readme_zh(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    predictions = "\n".join(f"{i}. {item}" for i, item in enumerate(spec["predictions_zh"], 1))
    anchors = "\n".join(f"| {i} | {item} |" for i, item in enumerate(spec["anchors_zh"], 1))
    analysis = analysis_zh(spec, artifact)
    no = spec["no"]
    return render_markdown_header(lesson_dir_zh(spec) / "README.md") + wrap(f'''# 第 {no:02d} 课 — {spec['title_zh']}

> **问题：**{spec['puzzle_zh']}

[← 第 04 章](../README.md) · [English lesson](../../../chapters/04-gpu-hardware-foundations/{lesson_name(spec)}/README.md) · [实验 Notebook](../../../chapters/04-gpu-hardware-foundations/{lesson_name(spec)}/lab.ipynb) · [RTX 5090 结果](../../../chapters/04-gpu-hardware-foundations/{lesson_name(spec)}/artifacts/rtx5090-result.json)

## 为什么值得研究

{spec['hook_zh']}

## 运行前先预测

{predictions}

## 1. 把机制放回物理空间

{spec['mechanism_zh']}

| # | 推理锚点 |
|---:|---|
{anchors}

{mermaid_guide(spec, 'zh')}

## 2. 读图

{visual_block(spec, 'zh')}

{extra_section(spec, 'zh')}

## 3. 把理论变成实验

**实验：**{spec['experiment_zh']}

| 实验角色 | 固定定义 |
|---|---|
| Baseline | {spec['baseline_zh']} |
| Candidate | {spec['candidate_zh']} |
| 保持不变 | {spec['controlled_zh']} |
| 测量内容 | {spec['metrics_zh']} |
| 证据标签 | `{spec['evidence_label']}` |

### 代码说明

{spec['code_zh']}

## 4. 阅读仓库中的 RTX 5090 结果

**记录环境：**{environment_line(artifact, 'zh')}。

{result_table(spec, artifact, 'zh')}

### 如何解释结果

{analysis}

## 5. 得出有边界的结论

> {spec['conclusion_zh']}

### 结论可能失效的条件

{spec['failure_zh']}

## 复现

```bash
python3 -m pip install -r requirements-gpu-foundations.txt
python3 scripts/execute_chapter_notebooks.py --chapter 04 --start {no} --end {no}
python3 scripts/build_chapter04_lessons.py
```

## 继续实验

{spec['extend_zh']}

## 证据边界

**证据标签：**[`{spec['evidence_label']}`](../README.md#证据标签)。{EVIDENCE_ZH[spec['evidence_label']]}

## 参考资料

{references(spec)}
''')


def markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(cell_id: str, source: str, old: dict[str, Any] | None) -> dict[str, Any]:
    cell = {"id": cell_id, "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [], "source": source.strip() + "\n"}
    old_source = old.get("source", "") if old else ""
    if isinstance(old_source, list): old_source = "".join(old_source)
    if old and old_source == cell["source"]:
        cell["execution_count"] = old.get("execution_count")
        cell["outputs"] = old.get("outputs", [])
    return cell


def notebook(spec: dict[str, Any], artifact: dict[str, Any] | None,
             old_nb: dict[str, Any] | None) -> dict[str, Any]:
    old_code = {cell.get("id"): cell for cell in (old_nb or {}).get("cells", [])
                if cell.get("cell_type") == "code"}
    no = spec["no"]
    predictions = "\n".join(f"{i}. {item}" for i, item in enumerate(spec["predictions_en"], 1))
    anchors = "\n".join(f"- {item}" for item in spec["anchors_en"])
    protocol = (f"| Role | Frozen value |\n|---|---|\n| Baseline | {spec['baseline_en']} |\n"
                f"| Candidate | {spec['candidate_en']} |\n| Held constant | {spec['controlled_en']} |\n"
                f"| Measurements | {spec['metrics_en']} |\n| Evidence | `{spec['evidence_label']}` |")
    analysis = artifact["analysis"] if artifact else "Run all cells to create the structured result."
    artifact_code = f'''artifact = Path("artifacts/rtx5090-result.json")
artifact.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "lesson": {no}, "title": {spec['title_en']!r}, "environment": ENV,
    "evidence_label": {spec['evidence_label']!r}, "metrics": metrics,
    "analysis": analysis, "conclusion": {spec['conclusion_en']!r},
}}
artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))'''
    visual = visual_block(spec, "en", notebook=True)
    cells = [
        markdown(f"c04-l{no:02d}-title", f"# Lesson {no:02d} Lab — {spec['title_en']}\n\n**Puzzle:** {spec['puzzle_en']}\n\nThis notebook retains one complete RTX 5090 execution."),
        markdown(f"c04-l{no:02d}-why", f"## Why this matters\n\n{spec['hook_en']}"),
        markdown(f"c04-l{no:02d}-predict", f"## 0. Predict before running\n\n{predictions}\n\nFor each prediction, write the observation that would disprove it."),
        markdown(f"c04-l{no:02d}-mechanism", f"## 1. Theory and mechanism\n\n{spec['mechanism_en']}\n\n{anchors}"),
        markdown(f"c04-l{no:02d}-map", f"## 2. Trace the mechanism\n\n{mermaid_guide(spec, 'en')}"),
        markdown(f"c04-l{no:02d}-visual", f"## 3. Inspect the visual boundary\n\n{visual}"),
        markdown(f"c04-l{no:02d}-env", "## 4. Inspect the execution environment\n\nThe next cell asserts CUDA, records GPU/PyTorch/CUDA identity, fixes the seed, and defines the common event-timing helpers."),
        code(f"c04-l{no:02d}-env-code", f"LESSON_NO = {no}\nLESSON_TITLE = {spec['title_en']!r}\n\n{ENV_CODE}", old_code.get(f"c04-l{no:02d}-env-code")),
        markdown(f"c04-l{no:02d}-protocol", f"## 5. Freeze the experiment\n\n{protocol}\n\n**Experiment:** {spec['experiment_en']}"),
        markdown(f"c04-l{no:02d}-codewalk", f"## 6. Inspect the code\n\n{spec['code_en']}\n\nDo not run until the code matches the frozen table."),
        code(f"c04-l{no:02d}-experiment-code", EXPERIMENTS[no], old_code.get(f"c04-l{no:02d}-experiment-code")),
        markdown(f"c04-l{no:02d}-result", f"## 7. Read the retained RTX 5090 result\n\n**Recorded environment:** {environment_line(artifact, 'en')}.\n\n{result_table(spec, artifact, 'en')}"),
        markdown(f"c04-l{no:02d}-interpret", f"## 8. Explain rather than overclaim\n\n{analysis}\n\n**Evidence boundary:** {EVIDENCE_EN[spec['evidence_label']]}"),
        markdown(f"c04-l{no:02d}-artifact", "## 9. Write the canonical artifact\n\nThe next cell stores the environment, metrics, analysis, evidence label, and bounded conclusion, then prints the exact JSON."),
        code(f"c04-l{no:02d}-artifact-code", artifact_code, old_code.get(f"c04-l{no:02d}-artifact-code")),
        markdown(f"c04-l{no:02d}-decision", f"## 10. Make the decision\n\n> {spec['conclusion_en']}\n\n**Failure analysis:** {spec['failure_en']}"),
        markdown(f"c04-l{no:02d}-extend", f"## 11. Extend the evidence\n\n{spec['extend_en']}\n\nSee [`README.md`](README.md) for the full explanation and references."),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


def phase_tables(lang: str) -> str:
    phases = [
        ("I", "Circuit and memory physics", "电路与存储物理", LESSONS[:3]),
        ("II", "Movement and compute data paths", "数据搬运与计算通路", LESSONS[3:7]),
        ("III", "On-chip organization and contention", "片上组织与竞争", LESSONS[7:11]),
        ("IV", "CUDA execution and optimization", "CUDA 执行与优化", LESSONS[11:15]),
        ("V", "Engineering evidence and hardware decisions", "工程证据与硬件决策", LESSONS[15:]),
    ]
    blocks = []
    for numeral, title_en, title_zh, lessons in phases:
        if lang == "en":
            rows = ["| Lesson | Puzzle | Lab |", "|---:|---|---|"]
            for spec in lessons:
                name = lesson_name(spec)
                rows.append(f"| {spec['no']:02d} | [{spec['title_en']}]({name}/README.md) | [notebook]({name}/lab.ipynb) |")
            blocks.append(f"## Phase {numeral} — {title_en}\n\n" + "\n".join(rows))
        else:
            rows = ["| 课 | 问题 | 实验 |", "|---:|---|---|"]
            for spec in lessons:
                name = lesson_name(spec)
                rows.append(f"| {spec['no']:02d} | [{spec['title_zh']}]({name}/README.md) | [notebook](../../chapters/04-gpu-hardware-foundations/{name}/lab.ipynb) |")
            blocks.append(f"## 阶段 {numeral} — {title_zh}\n\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def chapter_readme_en() -> str:
    return render_markdown_header(CHAPTER / "README.md") + wrap(f'''# Chapter 04 — GPU Hardware Foundations: From CMOS to Attention

[Project home](../../README.md) · [中文首页](../../README_ZH.md) · [中文本章](../../chapters-zh/04-gpu-hardware-foundations/README.md)

This 17-lesson chapter connects circuit intuition to CUDA and LLM inference. It begins
with CMOS switching and 1T1C DRAM, crosses the spatial memory hierarchy, HBM/GDDR
packaging, L2 slices, the on-chip network, and SM data paths, then turns those mechanisms
into experiments on data movement, attention IO, coalescing, atomics, reductions, events,
streams, and GPU specification audits.

The visual material was developed with Linnea Cai's GPU hardware study notes. Every
diagram is used as a conceptual teaching aid; commercial GPUs may differ in topology,
counts, circuit details, and product generation. Each lab separates a numerical model
from a native PyTorch/CUDA measurement and retains the exact evidence label.

```mermaid
flowchart LR
  A["CMOS + 1T1C"] --> B["register / SRAM / external memory"]
  B --> C["L2 + NoC + SM"]
  C --> D["CUDA execution"]
  D --> E["coalescing + reduction + streams"]
  E --> F["Attention + inference decisions"]
  F -->|"measured evidence"| B
```

## How to study this chapter

1. Make the prediction before reading retained output.
2. Use the diagram to trace where bits, requests, and partial results move.
3. Check whether the evidence is a model, capacity calculation, or native GPU execution.
4. Reuse the conclusion only when your shape, dtype, software, and hardware match.

## Evidence labels

| Label | What it establishes |
|---|---|
| `pytorch-gpu` | A named PyTorch CUDA operation ran on the recorded GPU and software stack |
| `numerical-model` | A transparent equation, queue, or SIMT model established one mechanism invariant |
| `capacity-model` | Measured environment facts fed explicit hierarchy, resource, or Roofline arithmetic |
| `compatibility-probe` | Repository/API structure was inspected without claiming performance causality |

{phase_tables('en')}

## Visual atlas

All source visuals are preserved under [`assets/`](assets/). The lessons embed every PNG
and link the interactive HTML and printable PDF variants at the point where they are used.

- [Interactive CMOS inverter](assets/visualizations/cmos-inverter.html)
- [Interactive 1T1C DRAM read](assets/visualizations/dram-1t1c-read-mechanism.html)
- [Interactive GPU memory layout](assets/visualizations/gpu-memory-spatial-layout.html)
- [Printable HBM path](assets/HBM_circuit_to_gpu_connection_A4_portrait.pdf)
- [Printable L2 slice](assets/L2_cache_slice_circuit_structure_A4_portrait.pdf)
- [Printable NoC](assets/NoC_on_chip_network_circuit_structure_A4_portrait.pdf)
- [Printable NoC and SM](assets/NoC_and_SM_circuit_structures_A4_portrait.pdf)
- [Four-page GPU circuit atlas](assets/GPU_circuit_structures_from_L2_A4_landscape.pdf)

## Reproduce and validate

```bash
python3 -m pip install -r requirements-gpu-foundations.txt
python3 scripts/execute_chapter_notebooks.py --chapter 04 --start 1 --end 17
python3 scripts/build_chapter04_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 04
python3 scripts/audit_chapter04_delivery.py
```
''')


def chapter_readme_zh() -> str:
    return render_markdown_header(CHAPTER_ZH / "README.md") + wrap(f'''# 第 04 章 — GPU 底层原理：从 CMOS 到 Attention

[← 中文首页](../../README_ZH.md) · [English chapter](../../chapters/04-gpu-hardware-foundations/README.md)

本章共 17 课，把电路直觉一直连接到 CUDA 与大模型推理。课程从 CMOS 开关和
1T1C DRAM 出发，经过 GPU 存储空间层级、HBM/GDDR 封装、L2 slice、NoC 和 SM
数据通路，最后把这些机制变成数据搬运、Attention IO、合并访存、原子操作、归约、
Event、Stream 与 GPU 参数审计实验。

视觉素材来自 Linnea Cai 的 GPU 底层原理学习笔记，并在本章逐一使用。图中结构是
概念性教学表达；不同商业 GPU 的拓扑、数量、电路细节与产品代际可能不同。每个实验
都会明确区分数值模型、容量计算与原生 PyTorch/CUDA 测量，并保留对应证据标签。

```mermaid
flowchart LR
  A["CMOS + 1T1C"] --> B["寄存器 / SRAM / 外部显存"]
  B --> C["L2 + NoC + SM"]
  C --> D["CUDA 执行"]
  D --> E["合并访存 + 归约 + Stream"]
  E --> F["Attention + 推理决策"]
  F -->|"实验证据"| B
```

## 学习方法

1. 先写预测，再看 notebook 中保留的结果。
2. 沿图追踪 bit、request 与 partial result 的移动路径。
3. 判断当前证据是模型、容量计算，还是原生 GPU 执行。
4. 只有 shape、dtype、软件与硬件条件一致时，才复用结论。

## 证据标签

| 标签 | 能说明什么 |
|---|---|
| `pytorch-gpu` | 指定 PyTorch CUDA 操作在记录的 GPU 与软件栈上运行 |
| `numerical-model` | 透明方程、排队或 SIMT 模型证明一个机制不变量 |
| `capacity-model` | 实测环境事实进入显式层级、资源或 Roofline 计算 |
| `compatibility-probe` | 检查仓库/API 结构，不声称性能因果关系 |

{phase_tables('zh')}

## 视觉资料

全部素材保存在英文主章节的 [`assets/`](../../chapters/04-gpu-hardware-foundations/assets/)
目录。PNG 会直接嵌入相关课程，交互 HTML 和打印版 PDF 也会在对应位置提供入口。

- [交互式 CMOS 反相器](../../chapters/04-gpu-hardware-foundations/assets/visualizations/cmos-inverter.html)
- [交互式 1T1C DRAM 读取](../../chapters/04-gpu-hardware-foundations/assets/visualizations/dram-1t1c-read-mechanism.html)
- [交互式 GPU 存储布局](../../chapters/04-gpu-hardware-foundations/assets/visualizations/gpu-memory-spatial-layout.html)
- [HBM 打印版](../../chapters/04-gpu-hardware-foundations/assets/HBM_circuit_to_gpu_connection_A4_portrait.pdf)
- [L2 slice 打印版](../../chapters/04-gpu-hardware-foundations/assets/L2_cache_slice_circuit_structure_A4_portrait.pdf)
- [NoC 打印版](../../chapters/04-gpu-hardware-foundations/assets/NoC_on_chip_network_circuit_structure_A4_portrait.pdf)
- [NoC 与 SM 打印版](../../chapters/04-gpu-hardware-foundations/assets/NoC_and_SM_circuit_structures_A4_portrait.pdf)
- [四页 GPU 电路图册](../../chapters/04-gpu-hardware-foundations/assets/GPU_circuit_structures_from_L2_A4_landscape.pdf)

## 复现与验证

```bash
python3 -m pip install -r requirements-gpu-foundations.txt
python3 scripts/execute_chapter_notebooks.py --chapter 04 --start 1 --end 17
python3 scripts/build_chapter04_lessons.py --chapter-readme
python3 scripts/validate_chapter.py 04
python3 scripts/audit_chapter04_delivery.py
```
''')


def build_chapter(*, refresh_chapter_readme: bool = False) -> None:
    CHAPTER.mkdir(parents=True, exist_ok=True)
    CHAPTER_ZH.mkdir(parents=True, exist_ok=True)
    for spec in LESSONS:
        directory = lesson_dir(spec)
        directory_zh = lesson_dir_zh(spec)
        (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        directory_zh.mkdir(parents=True, exist_ok=True)
        artifact = read_artifact(spec)
        notebook_path = directory / "lab.ipynb"
        old_nb = json.loads(notebook_path.read_text(encoding="utf-8")) if notebook_path.exists() else None
        (directory / "README.md").write_text(readme_en(spec, artifact), encoding="utf-8")
        (directory_zh / "README.md").write_text(readme_zh(spec, artifact), encoding="utf-8")
        notebook_path.write_text(json.dumps(notebook(spec, artifact, old_nb), indent=1,
                                            ensure_ascii=False) + "\n", encoding="utf-8")
    if refresh_chapter_readme or not (CHAPTER / "README.md").exists():
        (CHAPTER / "README.md").write_text(chapter_readme_en(), encoding="utf-8")
        (CHAPTER_ZH / "README.md").write_text(chapter_readme_zh(), encoding="utf-8")
    print(f"Built {len(LESSONS)} Chapter 04 bilingual lessons and notebooks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-readme", action="store_true")
    args = parser.parse_args()
    build_chapter(refresh_chapter_readme=args.chapter_readme)


if __name__ == "__main__":
    main()
