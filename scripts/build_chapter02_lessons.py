#!/usr/bin/env python3
"""Build Chapter 02 notes and executable notebooks from hand-written lesson specs."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from chapter02_content import COMMON_REFS, LESSONS
from chapter02_experiments import ENV_CODE, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "02-sparsity-structured-pruning"

EVIDENCE = {
    "pytorch-gpu": "The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel identity is not inferred unless a trace or backend artifact names it.",
    "numerical-model": "The CUDA experiment isolates a numerical mechanism. It is not a full paper reproduction, trained production model, or native sparse-kernel benchmark.",
    "compatibility-probe": "The notebook records real package/API availability and preserves the native success or failure state. Missing backend execution remains unmeasured.",
    "native-backend": "A named non-PyTorch backend executed and its checker/runtime output is retained. This still does not transfer to another backend or workload.",
    "capacity-model": "Measured CUDA facts and transparent storage arithmetic feed a decision model; unmeasured platform rows remain pending.",
}

OPTIONAL_INSTALLS = {
    15: "pip install torch-pruning",
    16: "pip install tensorflow tensorflow-model-optimization",
    17: "pip install openvino nncf neural-compressor",
    18: "pip install tensorrt polygraphy",
    19: "pip install onnx onnxruntime",
}


def wrap(text: str, width: int = 88) -> str:
    out: list[str] = []
    paragraph: list[str] = []
    fence = False

    def flush() -> None:
        if paragraph:
            out.extend(textwrap.wrap(" ".join(paragraph), width=width, break_long_words=False, break_on_hyphens=False))
            paragraph.clear()

    for raw in text.strip().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush(); out.append(line); fence = not fence
        elif fence:
            out.append(line)
        elif not stripped:
            flush()
            if out and out[-1] != "": out.append("")
        elif stripped.startswith(("#", "|", ">", "- ", "[", "**")) or re.match(r"^\d+\.\s", stripped):
            flush(); out.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out).rstrip() + "\n"


def lesson_dir(spec: dict[str, Any]) -> Path:
    return CHAPTER / f"{spec['no']:02d}-{spec['slug']}"


def get_metric(metrics: dict[str, Any], dotted: str) -> Any:
    value: Any = metrics
    for part in dotted.split("."):
        value = value[part]
    return value


def format_value(value: Any, kind: str) -> str:
    if value is None:
        return "not measured"
    if kind == "int": return f"{int(value):,}"
    if kind == "percent": return f"{float(value):.2%}"
    if kind == "ms": return f"{float(value):.6f} ms"
    if kind == "float": return f"{float(value):.6f}"
    if kind == "ratio": return f"{float(value):.3f}x"
    if kind == "rate": return f"{float(value):,.1f}/s"
    if kind == "mib": return f"{float(value):.3f} MiB"
    if kind == "bytes": return f"{int(value):,} bytes"
    if kind == "bool": return "yes" if bool(value) else "no"
    if kind == "short_hash": return f"`{str(value)[:12]}`"
    return str(value)


def result_table(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "_Run the notebook on CUDA to populate the checked-in result table._"
    rows = ["| Measured field | Checked-in value |", "|---|---:|"]
    metrics = artifact["metrics"]
    for label, path, kind in spec["result_fields"]:
        rows.append(f"| {label} | {format_value(get_metric(metrics, path), kind)} |")
    return "\n".join(rows)


def environment_line(artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "pending execution"
    env = artifact["environment"]
    return f"{env['gpu']}; compute capability {env['compute_capability']}; PyTorch {env['torch']}; CUDA runtime {env['cuda_runtime']}"


def reference_lines(spec: dict[str, Any]) -> str:
    return "\n".join(f"- [{COMMON_REFS[key][0]}]({COMMON_REFS[key][1]})" for key in spec["refs"])


def read_artifact(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = lesson_dir(spec) / "artifacts" / "rtx5090-result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def readme(spec: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    checks = "\n".join(f"{i}. {item}" for i, item in enumerate(spec["checks"], 1))
    anchors = "\n".join(f"| {i} | {item} |" for i, item in enumerate(spec["anchors"], 1))
    analysis = artifact["analysis"] if artifact else "The interpretation will be generated from the structured result after a complete CUDA run."
    optional_install = ""
    if spec["no"] in OPTIONAL_INSTALLS:
        optional_install = f'''\nThis lesson's optional/native backend path requires:\n\n```bash\n{OPTIONAL_INSTALLS[spec["no"]]}\n```\n'''
    return wrap(f'''# Lesson {spec['no']:02d} — {spec['title']}

> **Puzzle:** {spec['puzzle']}

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

{spec['hook']}

For **{spec['title']}**, the engineering question is not whether a definition can be
repeated; it is whether the following claim survives a controlled GPU test:
*{spec['puzzle']}* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

{checks}

Before opening Lesson {spec['no']:02d}'s retained output, answer the first prompt—
*{spec['checks'][0]}*—and write one observation that would falsify the answer. If the
result is already visible, hide it and make the commitment first; otherwise this becomes
post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

{spec['objects']}

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
{anchors}

Lesson {spec['no']:02d} tracks three layers through {spec['title']}: *value state* says
which entries are zero, *shape state* says which axes physically changed, and *execution
state* says which operator actually ran. The anchors above identify where this lesson's
claim lives, so a zero count cannot silently turn into a latency claim.

## 2. Derive the mechanism

{spec['mechanism']}

The inspectable invariant for **{spec['title']}** is tested by: {spec['experiment']} Its
purpose is to prevent the specific category error behind this puzzle. An algorithmic
change, a stored representation, and a runtime observation remain separate until the
candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** {spec['experiment']}

| Experimental role | Frozen definition |
|---|---|
| Baseline | {spec['baseline']} |
| Candidate | {spec['candidate']} |
| Held constant | {spec['controlled']} |
| Measurements | {spec['metrics']} |
| Evidence label | `{spec['evidence_label']}` |

This Lesson {spec['no']:02d} comparison is deliberately small enough to rerun on a
reader's GPU. Its control is **{spec['controlled']}**. That frozen condition preserves
the dependency or runtime boundary at issue; the small scale limits transfer to larger
models but does not permit the baseline and candidate to answer different questions.

### Code walk-through

{spec['code_walk']}

For **{spec['title']}**, the environment cell asserts CUDA and fixes a lesson-specific
seed. The experiment cell implements {spec['candidate']} and records {spec['metrics']}.
The artifact cell serializes those same fields. Only optional-backend import or API
failures become compatibility evidence; an error in the core comparison still fails the
notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** {environment_line(artifact)}.

{result_table(spec, artifact)}

### What the numbers mean

{analysis}

Lesson {spec['no']:02d}'s full [`rtx5090-result.json`](artifacts/rtx5090-result.json)
retains the arrays or diagnostic fields behind the compact selection above. For this
lesson, the interpretation is bounded by **{spec['evidence_label']}** evidence; the
printed notebook payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> {spec['conclusion']}

### Acceptance and rollback gate

{spec['gate']}

The gate for **{spec['title']}** is stricter than “the code ran” because it binds this
lesson's tensor or model identity, quality tolerance, workload, runtime path, and
rollback evidence. A missing optional package can settle a compatibility question, but
it cannot satisfy the native-performance decision stated above.

### How this conclusion can fail

{spec['failure']}

## 6. Follow the theory inside the notebook

In Lesson {spec['no']:02d}'s [`lab.ipynb`](lab.ipynb), first identify **{spec['baseline']}**
and **{spec['candidate']}** without running them. Next inspect the dimensions or
lifecycle state that implements the derivation. After **Run All**, verify the RTX 5090
environment and the frozen fields before reconciling the result table with the artifact.

The reader loop for **{spec['title']}** is **predict → execute → inspect → explain →
decide**. Transferring its final number to another architecture, workload shape, or
backend requires a new run because those variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/{spec['no']:02d}-{spec['slug']}/lab.ipynb
```

{optional_install}

To reproduce **{spec['title']}**, use a PyTorch build compiled for the target GPU and
select `Run All`. Compare the measurements in the frozen protocol with the checked-in
artifact. If this lesson touches an optional toolchain, install that named backend
before claiming native execution; otherwise only the compatibility fields are valid.

## Extend the experiment

{spec['next_step']}

For Lesson {spec['no']:02d}, the proposed extension is a new evidence layer rather than
a replacement for the checked-in control. Add one of its requested dimensions at a time
and retain this mechanism run, so a quality, export, operator, or service-level reversal
can be localized.

## Evidence boundary

{EVIDENCE[spec['evidence_label']]}

The checked-in **{spec['title']}** observation belongs to Lesson {spec['no']:02d}'s RTX
5090 environment, shapes, seed, and protocol. It does not establish the unmeasured task
quality or platform properties named in the failure analysis. This independently written
tutorial uses the study topic as a question, without redistributing source HTML, model
weights, private paths, or infrastructure.

## References

{reference_lines(spec)}
''')


def markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(cell_id: str, source: str, old: dict[str, Any] | None) -> dict[str, Any]:
    cell = {"id": cell_id, "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip() + "\n"}
    # nbclient may serialize an executed cell's source as a list of lines while
    # the generator uses one string. Compare normalized text so theory-only
    # rebuilds retain valid execution counts and outputs.
    old_source = old.get("source", "") if old else ""
    if isinstance(old_source, list):
        old_source = "".join(old_source)
    if old and old_source == cell["source"]:
        cell["execution_count"] = old.get("execution_count")
        cell["outputs"] = old.get("outputs", [])
    return cell


def notebook(spec: dict[str, Any], artifact: dict[str, Any] | None, old_nb: dict[str, Any] | None) -> dict[str, Any]:
    old_code = {cell.get("id"): cell for cell in (old_nb or {}).get("cells", []) if cell.get("cell_type") == "code"}
    no = spec["no"]
    checks = "\n".join(f"{i}. {item}" for i, item in enumerate(spec["checks"], 1))
    anchors = "\n".join(f"- {item}" for item in spec["anchors"])
    protocol = f"| Role | Frozen value |\n|---|---|\n| Baseline | {spec['baseline']} |\n| Candidate | {spec['candidate']} |\n| Held constant | {spec['controlled']} |\n| Measurements | {spec['metrics']} |\n| Evidence | `{spec['evidence_label']}` |"
    analysis = artifact["analysis"] if artifact else "Run the code cells to create the first structured result."
    artifact_code = f'''artifact = Path("artifacts/rtx5090-result.json")
artifact.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "lesson": {no},
    "title": {spec['title']!r},
    "environment": ENV,
    "evidence_label": {spec['evidence_label']!r},
    "metrics": metrics,
    "analysis": analysis,
    "conclusion": {spec['conclusion']!r},
}}
artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))'''
    cells = [
        markdown(f"c02-l{no:02d}-title", f"# Lesson {no:02d} Lab — {spec['title']}\n\n**Puzzle:** {spec['puzzle']}\n\nThis notebook is designed for a CUDA GPU and retains the output of a complete RTX 5090 run."),
        markdown(f"c02-l{no:02d}-why", f"## Why this matters\n\n{spec['hook']}"),
        markdown(f"c02-l{no:02d}-predict", f"## 0. Predict before running\n\n{checks}\n\nFor every answer, name the observation that would prove it wrong."),
        markdown(f"c02-l{no:02d}-objects", f"## 1. Name the concrete objects\n\n{spec['objects']}\n\n{anchors}"),
        markdown(f"c02-l{no:02d}-derive", f"## 2. Derive the mechanism\n\n{spec['mechanism']}\n\nKeep value sparsity, physical shape, representation, and runtime evidence separate."),
        markdown(f"c02-l{no:02d}-env", "## 3. Verify the execution environment\n\nInspect the next cell before running it: it asserts CUDA, fixes the seed, defines transparent timing/numerical helpers, and prints the GPU/PyTorch/CUDA record needed to interpret every output."),
        code(f"c02-l{no:02d}-env-code", f"LESSON_NO = {no}\nLESSON_TITLE = {spec['title']!r}\n\n{ENV_CODE}", old_code.get(f"c02-l{no:02d}-env-code")),
        markdown(f"c02-l{no:02d}-protocol", f"## 4. Freeze the comparison\n\n{protocol}\n\n**Experiment:** {spec['experiment']}"),
        markdown(f"c02-l{no:02d}-codewalk", f"## 5. Read the experiment code\n\n{spec['code_walk']}\n\nDo not execute until the code implements the frozen table above."),
        code(f"c02-l{no:02d}-experiment-code", EXPERIMENTS[no], old_code.get(f"c02-l{no:02d}-experiment-code")),
        markdown(f"c02-l{no:02d}-results", f"## 6. Read the retained RTX 5090 result\n\n**Recorded environment:** {environment_line(artifact)}.\n\n{result_table(spec, artifact)}"),
        markdown(f"c02-l{no:02d}-interpret", f"## 7. Interpret rather than merely print\n\n{analysis}\n\nThe result is bounded to the shapes, seed, packages, and evidence label printed here."),
        markdown(f"c02-l{no:02d}-evidence", f"## 8. Keep the evidence label honest\n\nThis run is labeled **`{spec['evidence_label']}`**. {EVIDENCE[spec['evidence_label']]}\n\nThe next cell writes the canonical JSON artifact and prints the same payload."),
        code(f"c02-l{no:02d}-artifact-code", artifact_code, old_code.get(f"c02-l{no:02d}-artifact-code")),
        markdown(f"c02-l{no:02d}-decision", f"## 9. Make the bounded decision\n\n> {spec['conclusion']}\n\n**Acceptance/rollback:** {spec['gate']}\n\n**Failure analysis:** {spec['failure']}"),
        markdown(f"c02-l{no:02d}-next", f"## 10. Extend the evidence\n\n{spec['next_step']}\n\nThe full evidence boundary and references are in [`README.md`](README.md)."),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def chapter_readme() -> str:
    phases = [("I", "Objectives and pruning mechanics", LESSONS[:7]), ("II", "Dependencies, schedules, and framework lifecycles", LESSONS[7:14]), ("III", "Native toolchains and model families", LESSONS[14:21]), ("IV", "Transformers, production evidence, and platform decisions", LESSONS[21:])]
    blocks = []
    for numeral, title, lessons in phases:
        rows = ["| Lesson | Core decision | Lab |", "|---:|---|---|"]
        for spec in lessons:
            directory = f"{spec['no']:02d}-{spec['slug']}"
            rows.append(f"| {spec['no']:02d} | [{spec['title']}]({directory}/README.md) | [notebook]({directory}/lab.ipynb) |")
        blocks.append(f"## Phase {numeral} — {title}\n\n" + "\n".join(rows))
    return wrap(f'''# Chapter 02 — Sparsity and Structured Pruning

This chapter turns model sparsity from a zero-count exercise into a chain of testable
decisions. Its 28 lessons cover objectives, granularities, masks, physical channel
deletion, dependency graphs, recovery schedules, N:M constraints, framework lifecycles,
ONNX/TensorRT boundaries, CNN/Transformer/LLM cases, benchmarking, rollback,
reproducibility, and platform-specific deployment.

Every lesson follows one delivery contract:

```text
Concrete tensors/state → mechanism or equation → frozen comparison
                       → retained RTX 5090 evidence → acceptance/rollback
```

The notes are independently written from the ideas and engineering problems in the
study material. The source HTML is not copied into this repository. Numerical models,
compatibility probes, native backends, and performance runs carry different evidence
labels so a package check or zero-rate calculation cannot be mistaken for acceleration.

{'\n\n'.join(blocks)}

## Reproduce and validate

Execute all labs from the repository root on a CUDA GPU:

```bash
python3 scripts/execute_chapter_notebooks.py --chapter 02 --start 1 --end 28
python3 scripts/build_chapter02_lessons.py
python3 scripts/validate_chapter.py 02
python3 scripts/audit_chapter02_delivery.py
```

Optional framework lessons retain a bounded compatibility result when their native
package is absent. Install the named backend and rerun that notebook before making a
backend-performance claim.
''')


def build_chapter(*, refresh_chapter_readme: bool = False) -> None:
    """Refresh lesson prose/notebook theory while preserving unchanged outputs."""

    CHAPTER.mkdir(parents=True, exist_ok=True)
    for spec in LESSONS:
        directory = lesson_dir(spec); (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        artifact = read_artifact(spec)
        nb_path = directory / "lab.ipynb"
        old_nb = json.loads(nb_path.read_text(encoding="utf-8")) if nb_path.exists() else None
        (directory / "README.md").write_text(readme(spec, artifact), encoding="utf-8")
        nb_path.write_text(json.dumps(notebook(spec, artifact, old_nb), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if refresh_chapter_readme or not (CHAPTER / "README.md").exists():
        (CHAPTER / "README.md").write_text(chapter_readme(), encoding="utf-8")
    print(f"Built {len(LESSONS)} Chapter 02 lesson notes and notebooks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-readme", action="store_true", help="also refresh the hand-written chapter map")
    args = parser.parse_args()
    build_chapter(refresh_chapter_readme=args.chapter_readme)


if __name__ == "__main__":
    main()
