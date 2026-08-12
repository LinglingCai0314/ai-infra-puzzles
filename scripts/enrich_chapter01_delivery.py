#!/usr/bin/env python3
"""Turn Chapter 01 mechanism labs into full reader-facing tutorials."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from build_chapter01_lessons import CHAPTER, LESSONS, THEORY
from chapter01_delivery_content import DELIVERY, environment_line, result_table
from tutorial_guides import CHAPTER_01_GUIDES, render_guide


EVIDENCE_EXPLANATIONS = {
    "pytorch-gpu": (
        "The measured tensors and operations ran on CUDA through PyTorch. The result "
        "does not name a separate production backend unless an operator trace identifies it."
    ),
    "numerical-model": (
        "The CUDA numerical experiment isolates an algorithmic mechanism. It is not the "
        "paper's complete implementation and does not establish a production kernel speedup."
    ),
    "compatibility-probe": (
        "The named optional backend did not complete a native run in this environment. "
        "Package and failure evidence are retained; service or kernel performance is not inferred."
    ),
    "capacity-model": (
        "The calculation uses live GPU information and/or a CUDA probe, but it remains a "
        "planning model until a named full engine, quality suite, and service workload execute."
    ),
}


def markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def unique_references(lesson: dict[str, Any], delivery: dict[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, url in [*lesson["refs"], *delivery["extra_refs"]]:
        if url not in seen:
            seen.add(url)
            result.append((name, url))
    return result


def format_markdown(text: str, width: int = 88) -> str:
    """Wrap prose like a hand-edited README while preserving Markdown blocks."""

    output: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        if paragraph:
            output.extend(
                textwrap.wrap(
                    " ".join(paragraph),
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
            paragraph.clear()

    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            output.append(line)
            in_fence = not in_fence
        elif in_fence:
            output.append(line)
        elif not stripped:
            flush()
            if output and output[-1] != "":
                output.append("")
        elif (
            stripped.startswith(("#", "|", ">", "- ", "[", "**"))
            or re.match(r"^\d+\.\s", stripped)
        ):
            flush()
            output.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(output).rstrip() + "\n"


def readme_for(lesson: dict[str, Any], artifact: dict[str, Any]) -> str:
    no = lesson["no"]
    theory = THEORY[no]
    delivery = DELIVERY[no]
    concepts = "\n".join(f"| {index} | {item} |" for index, item in enumerate(lesson["concepts"], 1))
    checks = "\n".join(f"{index}. {item}" for index, item in enumerate(delivery["checks"], 1))
    references = "\n".join(
        f"- [{name}]({url})" for name, url in unique_references(lesson, delivery)
    )
    results = result_table(no, artifact)
    guide = render_guide(CHAPTER_01_GUIDES.get(no))
    return format_markdown(f"""# Lesson {no:02d} — {lesson['title']}

> **Puzzle:** {lesson['puzzle']}

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

{delivery['hook']}

## Predict before reading the result

{checks}

## 1. Start from concrete tensors and state

{theory['objects']}

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
{concepts}

## 2. Derive the mechanism

{theory['mechanism']}

{delivery['derivation']}

{guide}

## 3. Translate the theory into an experiment

**Experiment:** {lesson['experiment']}

| Experimental role | Frozen definition |
|---|---|
| Baseline | {delivery['baseline']} |
| Candidate | {delivery['candidate']} |
| Held constant | {delivery['controlled']} |
| Measurements | {delivery['metrics']} |
| Evidence label | `{lesson['label']}` |

{theory['code']}

### Code walk-through

{delivery['code_walk']}

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** {environment_line(artifact)}.

{results}

### What the numbers mean

{delivery['result_reading']}

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you
need every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> {lesson['conclusion']}

### Acceptance and rollback gate

{theory['gate']}

### How this conclusion can fail

{delivery['failure']}

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/{no:02d}-{lesson['slug']}/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

{delivery['next']}

## Evidence boundary

**Evidence label:** [`{lesson['label']}`](../README.md#evidence-labels).

## References

{references}
""")


def notebook_for(
    lesson: dict[str, Any], artifact: dict[str, Any], original: dict[str, Any]
) -> dict[str, Any]:
    no = lesson["no"]
    theory = THEORY[no]
    delivery = DELIVERY[no]
    code_cells = [cell for cell in original["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) != 3:
        raise ValueError(f"Lesson {no:02d}: expected three code cells, found {len(code_cells)}")
    concepts = "\n".join(f"- {item}" for item in lesson["concepts"])
    checks = "\n".join(f"{index}. {item}" for index, item in enumerate(delivery["checks"], 1))
    protocol = (
        "| Role | This run |\n|---|---|\n"
        f"| Baseline | {delivery['baseline']} |\n"
        f"| Candidate | {delivery['candidate']} |\n"
        f"| Held constant | {delivery['controlled']} |\n"
        f"| Measurements | {delivery['metrics']} |\n"
        f"| Evidence | `{lesson['label']}` |"
    )
    guide = render_guide(CHAPTER_01_GUIDES.get(no))
    cells = [
        markdown(
            f"l{no:02d}-title-v2",
            f"# Lesson {no:02d} Lab — {lesson['title']}\n\n"
            f"**Puzzle:** {lesson['puzzle']}\n\n"
            "This notebook keeps the RTX 5090 outputs from a complete run. Read the "
            "theory cells, make a prediction, and then use **Run All** on your own GPU.",
        ),
        markdown(f"l{no:02d}-why-v2", f"## Why this matters\n\n{delivery['hook']}"),
        markdown(
            f"l{no:02d}-predict-v2",
            f"## 0. Predict before running\n\n{checks}\n\n"
            "For each answer, name the observation that would prove you wrong.",
        ),
        markdown(
            f"l{no:02d}-objects-v2",
            f"## 1. Name the concrete objects\n\n{theory['objects']}\n\n{concepts}",
        ),
        markdown(
            f"l{no:02d}-derive-v2",
            f"## 2. Derive the mechanism\n\n{theory['mechanism']}\n\n"
            f"{delivery['derivation']}\n\n{guide}",
        ),
        markdown(
            f"l{no:02d}-env-v2",
            "## 3. Verify the execution environment\n\n"
            "The next cell asserts CUDA availability, fixes the seed, locates the lesson, "
            "and prints a sanitized GPU/PyTorch/CUDA record. Check it before interpreting output.",
        ),
        code_cells[0],
        markdown(
            f"l{no:02d}-protocol-v2",
            f"## 4. Freeze the comparison\n\n{protocol}\n\n**Experiment:** {lesson['experiment']}",
        ),
        markdown(
            f"l{no:02d}-codewalk-v2",
            f"## 5. Read the experiment code\n\n{theory['code']}\n\n{delivery['code_walk']}\n\n"
            "Only after these variables match the protocol should the cell be executed.",
        ),
        code_cells[1],
        markdown(
            f"l{no:02d}-results-v2",
            f"## 6. Read the retained RTX 5090 result\n\n"
            f"**Recorded environment:** {environment_line(artifact)}.\n\n"
            f"{result_table(no, artifact)}",
        ),
        markdown(
            f"l{no:02d}-interpret-v2",
            f"## 7. Interpret rather than merely print\n\n{delivery['result_reading']}\n\n"
            f"**Inspection rule:** {lesson['inspect']}",
        ),
        markdown(
            f"l{no:02d}-evidence-v2",
            f"## 8. Keep the evidence label honest\n\n"
            f"This run is labeled **`{lesson['label']}`**. "
            f"{EVIDENCE_EXPLANATIONS[lesson['label']]}\n\n"
            "The next cell writes the complete structured result; its existing saved output "
            "is part of the checked-in evidence.",
        ),
        code_cells[2],
        markdown(
            f"l{no:02d}-decision-v2",
            f"## 9. Make the bounded decision\n\n> {lesson['conclusion']}\n\n"
            f"**Acceptance/rollback:** {theory['gate']}\n\n"
            f"**Failure analysis:** {delivery['failure']}",
        ),
        markdown(
            f"l{no:02d}-next-v2",
            f"## 10. Extend the evidence\n\n{delivery['next']}\n\n"
            "The full derivation, reproduction command, evidence boundary and primary "
            "references are in [`README.md`](README.md).",
        ),
    ]
    updated = dict(original)
    updated["cells"] = cells
    return updated


def main() -> None:
    changed = 0
    for lesson in LESSONS:
        no = lesson["no"]
        directory = CHAPTER / f"{no:02d}-{lesson['slug']}"
        artifact_path = directory / "artifacts" / "rtx5090-result.json"
        notebook_path = directory / "lab.ipynb"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        original = json.loads(notebook_path.read_text(encoding="utf-8"))
        (directory / "README.md").write_text(
            readme_for(lesson, artifact), encoding="utf-8"
        )
        notebook_path.write_text(
            json.dumps(notebook_for(lesson, artifact, original), ensure_ascii=False, indent=1)
            + "\n",
            encoding="utf-8",
        )
        changed += 1
    print(f"Enriched {changed} lesson READMEs and notebooks")


if __name__ == "__main__":
    main()
