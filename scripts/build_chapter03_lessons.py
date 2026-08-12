#!/usr/bin/env python3
"""Build Chapter 03 notes and executable notebooks from original lesson specs."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from chapter03_content import COMMON_REFS, LESSONS
from chapter03_experiments import ENV_CODE, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "03-vllm-inference-serving"

EVIDENCE = {
    "native-backend": "The named vLLM runtime executed on the recorded GPU/model/workload. The result does not transfer to another version, model, endpoint, or traffic distribution.",
    "pytorch-gpu": "CUDA work executed through PyTorch. An unnamed vLLM kernel or service property is not inferred.",
    "numerical-model": "A transparent allocator, scheduler, gateway, or policy model executed. It establishes the stated invariant, not native vLLM performance.",
    "capacity-model": "Measured environment facts feed explicit planning arithmetic. Assumed topology, demand, bandwidth, and reserve fields remain assumptions until a native deployment test.",
    "compatibility-probe": "The installed package/API/configuration surface was inspected. Availability or lint success is not equivalent to native feature execution.",
}


def wrap(text: str, width: int = 88) -> str:
    output: list[str] = []
    paragraph: list[str] = []
    fenced = False

    def flush() -> None:
        if paragraph:
            output.extend(textwrap.wrap(" ".join(paragraph), width=width,
                                       break_long_words=False, break_on_hyphens=False))
            paragraph.clear()

    for raw in text.strip().splitlines():
        line = raw.rstrip(); stripped = line.strip()
        if stripped.startswith("```"):
            flush(); output.append(line); fenced = not fenced
        elif fenced:
            output.append(line)
        elif not stripped:
            flush()
            if output and output[-1] != "": output.append("")
        elif stripped.startswith(("#", "|", ">", "- ", "[", "**")) or re.match(r"^\d+\.\s", stripped):
            flush(); output.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(output).rstrip() + "\n"


def lesson_dir(spec: dict[str, Any]) -> Path:
    return CHAPTER / f"{spec['no']:02d}-{spec['slug']}"


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
    if kind == "float": return f"{float(value):.6f}"
    if kind == "ratio": return f"{float(value):.3f}x"
    if kind == "rate": return f"{float(value):,.1f}/s"
    if kind == "mib": return f"{float(value):,.3f} MiB"
    if kind == "bytes": return f"{int(value):,} bytes"
    if kind == "bool": return "yes" if bool(value) else "no"
    if kind == "short_hash": return f"`{str(value)[:12]}`"
    return str(value)


def result_table(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "_Run the notebook on CUDA to populate this checked-in result table._"
    rows = ["| Measured field | Checked-in value |", "|---|---:|"]
    for label, path, kind in spec["result_fields"]:
        try: value = get_metric(artifact["metrics"], path)
        except (KeyError, TypeError): value = None
        rows.append(f"| {label} | {format_value(value, kind)} |")
    return "\n".join(rows)


def environment_line(artifact: dict[str, Any] | None) -> str:
    if artifact is None: return "pending execution"
    env = artifact["environment"]
    return (f"{env['gpu']}; compute capability {env['compute_capability']}; PyTorch "
            f"{env['torch']}; CUDA runtime {env['cuda_runtime']}; vLLM {env.get('vllm', 'unknown')}")


def reference_lines(spec: dict[str, Any]) -> str:
    return "\n".join(f"- [{COMMON_REFS[key][0]}]({COMMON_REFS[key][1]})" for key in spec["refs"])


def render_guide(spec: dict[str, Any]) -> str:
    steps = "\n".join(f"{index}. **{title}.** {explanation}"
                      for index, (title, explanation) in enumerate(spec["steps"], 1))
    return f"""### Mechanism at a glance

```mermaid
{spec['mermaid']}
```

### Walk it step by step

{steps}
"""


def read_artifact(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = lesson_dir(spec) / "artifacts" / "rtx5090-result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def readme(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    checks = "\n".join(f"{index}. {item}" for index, item in enumerate(spec["checks"], 1))
    anchors = "\n".join(f"| {index} | {item} |" for index, item in enumerate(spec["anchors"], 1))
    analysis = artifact["analysis"] if artifact else "Run the notebook to create the first structured RTX 5090 result."
    guide = render_guide(spec)
    return wrap(f'''# Lesson {spec['no']:02d} — {spec['title']}

> **Puzzle:** {spec['puzzle']}

[← Chapter 03](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

{spec['hook']}

## Predict before reading the result

{checks}

## 1. Start from concrete requests and state

{spec['objects']}

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
{anchors}

## 2. Derive the mechanism

{spec['mechanism']}

{guide}

## 3. Translate the theory into an experiment

**Experiment:** {spec['experiment']}

| Experimental role | Frozen definition |
|---|---|
| Baseline | {spec['baseline']} |
| Candidate | {spec['candidate']} |
| Held constant | {spec['controlled']} |
| Measurements | {spec['metrics']} |
| Evidence label | `{spec['evidence_label']}` |

### Code walk-through

{spec['code_walk']}

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** {environment_line(artifact)}.

{result_table(spec, artifact)}

### What the numbers mean

{analysis}

## 5. Solve the puzzle and make a decision

> {spec['conclusion']}

### Acceptance and rollback gate

{spec['gate']}

### How this conclusion can fail

{spec['failure']}

## Reproduce

The checked-in run pins vLLM 0.27.1 and a local Qwen2.5-1.5B-Instruct checkpoint.
On a Linux CUDA host, create a clean environment and point `CH3_MODEL` at your local
checkpoint:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.27.1 nbclient nbformat ipykernel requests pyyaml --torch-backend=auto
export CH3_MODEL=/path/to/Qwen2.5-1.5B-Instruct
jupyter lab chapters/03-vllm-inference-serving/{spec['no']:02d}-{spec['slug']}/lab.ipynb
```

## Extend the experiment

{spec['next_step']}

## Evidence boundary

**Evidence label:** [`{spec['evidence_label']}`](../README.md#evidence-labels). {EVIDENCE[spec['evidence_label']]}

## References

{reference_lines(spec)}
''')


def markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(cell_id: str, source: str, old: dict[str, Any] | None) -> dict[str, Any]:
    cell = {"id": cell_id, "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [], "source": source.strip() + "\n"}
    old_source = old.get("source", "") if old else ""
    if isinstance(old_source, list): old_source = "".join(old_source)
    if old and old_source == cell["source"]:
        cell["execution_count"] = old.get("execution_count"); cell["outputs"] = old.get("outputs", [])
    return cell


def notebook(spec: dict[str, Any], artifact: dict[str, Any] | None,
             old_nb: dict[str, Any] | None) -> dict[str, Any]:
    old_code = {cell.get("id"): cell for cell in (old_nb or {}).get("cells", [])
                if cell.get("cell_type") == "code"}
    no = spec["no"]
    checks = "\n".join(f"{index}. {item}" for index, item in enumerate(spec["checks"], 1))
    anchors = "\n".join(f"- {item}" for item in spec["anchors"])
    protocol = (f"| Role | Frozen value |\n|---|---|\n| Baseline | {spec['baseline']} |\n"
                f"| Candidate | {spec['candidate']} |\n| Held constant | {spec['controlled']} |\n"
                f"| Measurements | {spec['metrics']} |\n| Evidence | `{spec['evidence_label']}` |")
    analysis = artifact["analysis"] if artifact else "Run all cells to create the structured result."
    artifact_code = f'''artifact = Path("artifacts/rtx5090-result.json")
artifact.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "lesson": {no}, "title": {spec['title']!r}, "environment": ENV,
    "evidence_label": {spec['evidence_label']!r}, "metrics": metrics,
    "analysis": analysis, "conclusion": {spec['conclusion']!r},
}}
artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))'''
    cells = [
        markdown(f"c03-l{no:02d}-title", f"# Lesson {no:02d} Lab — {spec['title']}\n\n**Puzzle:** {spec['puzzle']}\n\nThis notebook retains the output of a complete RTX 5090 run."),
        markdown(f"c03-l{no:02d}-why", f"## Why this matters\n\n{spec['hook']}"),
        markdown(f"c03-l{no:02d}-predict", f"## 0. Predict before running\n\n{checks}\n\nFor every answer, name the observation that would disprove it."),
        markdown(f"c03-l{no:02d}-objects", f"## 1. Name the concrete objects\n\n{spec['objects']}\n\n{anchors}"),
        markdown(f"c03-l{no:02d}-derive", f"## 2. Derive the mechanism\n\n{spec['mechanism']}\n\n{render_guide(spec)}"),
        markdown(f"c03-l{no:02d}-env", "## 3. Inspect the execution environment\n\nThe next cell asserts CUDA, prints the RTX 5090/PyTorch/CUDA/vLLM identity, fixes a seed, and defines only the helpers used by this chapter."),
        code(f"c03-l{no:02d}-env-code", f"LESSON_NO = {no}\nLESSON_TITLE = {spec['title']!r}\n\n{ENV_CODE}", old_code.get(f"c03-l{no:02d}-env-code")),
        markdown(f"c03-l{no:02d}-protocol", f"## 4. Freeze the comparison\n\n{protocol}\n\n**Experiment:** {spec['experiment']}"),
        markdown(f"c03-l{no:02d}-codewalk", f"## 5. Inspect the experiment code\n\n{spec['code_walk']}\n\nDo not execute until the code matches the frozen table."),
        code(f"c03-l{no:02d}-experiment-code", EXPERIMENTS[no], old_code.get(f"c03-l{no:02d}-experiment-code")),
        markdown(f"c03-l{no:02d}-results", f"## 6. Read the retained RTX 5090 result\n\n**Recorded environment:** {environment_line(artifact)}.\n\n{result_table(spec, artifact)}"),
        markdown(f"c03-l{no:02d}-interpret", f"## 7. Explain the result\n\n{analysis}\n\nThis interpretation is bounded to the printed model, GPU, packages, workload, and evidence label."),
        markdown(f"c03-l{no:02d}-evidence", f"## 8. Keep the evidence label honest\n\nThis run is labeled **`{spec['evidence_label']}`**. {EVIDENCE[spec['evidence_label']]}\n\nThe next cell writes and prints the canonical JSON artifact."),
        code(f"c03-l{no:02d}-artifact-code", artifact_code, old_code.get(f"c03-l{no:02d}-artifact-code")),
        markdown(f"c03-l{no:02d}-decision", f"## 9. Make the bounded decision\n\n> {spec['conclusion']}\n\n**Acceptance/rollback:** {spec['gate']}\n\n**Failure analysis:** {spec['failure']}"),
        markdown(f"c03-l{no:02d}-next", f"## 10. Extend the evidence\n\n{spec['next_step']}\n\nThe full boundary and references are in [`README.md`](README.md)."),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}


def chapter_readme() -> str:
    phases = [
        ("I", "Serving foundations: phases, scheduling, memory, and environment", LESSONS[:6]),
        ("II", "Core APIs, request contracts, provenance, and parallel placement", LESSONS[6:11]),
        ("III", "Cache, quantization, adapters, speculation, and model capabilities", LESSONS[11:19]),
        ("IV", "Benchmarking, observability, deployment, tenancy, and diagnosis", LESSONS[19:25]),
        ("V", "Tuning, disaggregation, capacity, security, and release", LESSONS[25:]),
    ]
    blocks = []
    for numeral, title, lessons in phases:
        rows = ["| Lesson | Core decision | Lab |", "|---:|---|---|"]
        for spec in lessons:
            directory = f"{spec['no']:02d}-{spec['slug']}"
            rows.append(f"| {spec['no']:02d} | [{spec['title']}]({directory}/README.md) | [notebook]({directory}/lab.ipynb) |")
        blocks.append(f"## Phase {numeral} — {title}\n\n" + "\n".join(rows))
    return wrap(f'''# Chapter 03 — vLLM Inference and Serving

[Project home](../../README.md) · [中文首页](../../README_ZH.md) ·
[中文本章](../../chapters-zh/03-vllm-inference-serving/README.md)

This 30-lesson chapter follows a request from prompt ingestion to a reversible production
release. It covers Prefill/Decode/KV state, PagedAttention, continuous batching, memory
budgets, offline and OpenAI-compatible APIs, prefix caching, quantized KV, LoRA,
speculation, structured outputs, benchmarking, metrics, containers, Kubernetes,
multi-tenancy, diagnosis, capacity, security, and launch gates.

The chapter is independently written from the engineering topics in the study material;
its HTML prose is not copied. Every lab makes a prediction, freezes a comparison, retains
RTX 5090 output, and marks the exact evidence class. Single-GPU labs never claim that an
eight-GPU topology, Kubernetes cluster, or disaggregated deployment was measured.

```mermaid
flowchart LR
  A["request contract"] --> B["Prefill + KV allocation"]
  B --> C["continuous Decode scheduling"]
  C --> D["API + observability"]
  D --> E["capacity + deployment"]
  E --> F["canary + rollback"]
  F -->|"new evidence"| A
```

## How to read a lesson

1. Commit to the prediction before opening the retained result.
2. Trace the Mermaid diagram into concrete requests, cache state, and scheduler decisions.
3. Verify the frozen model, sampling, engine, and environment before comparing metrics.
4. Apply the evidence label and rollback gate before reusing a conclusion.

## Evidence labels

| Label | What it establishes |
|---|---|
| `native-backend` | The named vLLM runtime executed for the recorded model/workload |
| `pytorch-gpu` | CUDA execution through PyTorch without an unnamed runtime claim |
| `numerical-model` | A transparent mechanism/policy model, not native service performance |
| `capacity-model` | Planning arithmetic anchored by measured facts and explicit assumptions |
| `compatibility-probe` | Installed APIs/configurations and the boundary of missing native execution |

{chr(10).join(blocks)}

## Reproduce and validate

```bash
export CH3_MODEL=/path/to/Qwen2.5-1.5B-Instruct
python3 scripts/execute_chapter_notebooks.py --chapter 03 --start 1 --end 30
python3 scripts/build_chapter03_lessons.py
python3 scripts/validate_chapter.py 03
python3 scripts/audit_chapter03_delivery.py
```

The checked-in environment uses vLLM 0.27.1. Do not silently replace it with a newer
release and compare numbers as though the software stack were unchanged.
''')


def build_chapter(*, refresh_chapter_readme: bool = False) -> None:
    CHAPTER.mkdir(parents=True, exist_ok=True)
    for spec in LESSONS:
        directory = lesson_dir(spec); (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        artifact = read_artifact(spec); notebook_path = directory / "lab.ipynb"
        old_nb = json.loads(notebook_path.read_text(encoding="utf-8")) if notebook_path.exists() else None
        (directory / "README.md").write_text(readme(spec, artifact), encoding="utf-8")
        notebook_path.write_text(json.dumps(notebook(spec, artifact, old_nb), indent=1,
                                            ensure_ascii=False) + "\n", encoding="utf-8")
    if refresh_chapter_readme or not (CHAPTER / "README.md").exists():
        (CHAPTER / "README.md").write_text(chapter_readme(), encoding="utf-8")
    print(f"Built {len(LESSONS)} Chapter 03 lesson notes and notebooks")


def sanitize_execution_records() -> None:
    """Remove host-private absolute prefixes while retaining model provenance hashes."""

    replacements = {"/" + "root" + "/": "<remote-home>/"}
    configured_model = os.environ.get("CH3_MODEL")
    if configured_model:
        replacements[configured_model] = "$CH3_MODEL"
    changed = 0
    for path in list(CHAPTER.glob("[0-9][0-9]-*/lab.ipynb")) + list(
        CHAPTER.glob("[0-9][0-9]-*/artifacts/rtx5090-result.json")
    ):
        text = path.read_text(encoding="utf-8")
        sanitized = text
        for private, public in replacements.items():
            sanitized = sanitized.replace(private, public)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8"); changed += 1
    print(f"Sanitized {changed} Chapter 03 execution records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-readme", action="store_true")
    args = parser.parse_args(); build_chapter(refresh_chapter_readme=args.chapter_readme)


if __name__ == "__main__":
    main()
