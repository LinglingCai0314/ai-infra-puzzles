#!/usr/bin/env python3
"""Audit reader-facing quality for Chapter 01 Lessons 02–30."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from build_chapter01_lessons import CHAPTER, LESSONS
from chapter01_delivery_content import DELIVERY, result_table
from tutorial_guides import CHAPTER_01_GUIDES


ROOT = CHAPTER.parents[1]


REQUIRED_HEADINGS = (
    "## Why this puzzle matters",
    "## Predict before reading the result",
    "## 1. Start from concrete tensors and state",
    "## 2. Derive the mechanism",
    "## 3. Translate the theory into an experiment",
    "## 4. Read the checked-in RTX 5090 result",
    "## 5. Solve the puzzle and make a decision",
    "## Reproduce",
    "## Extend the experiment",
    "## Evidence boundary",
    "## References",
)


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def check_relative_links(path: Path, text: str, issues: list[str]) -> None:
    """Require every reader-facing relative Markdown link to resolve inside the repo."""

    for raw_target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        target = unquote(raw_target.strip().split("#", 1)[0])
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I):
            continue
        resolved = (path.parent / target).resolve()
        if ROOT not in resolved.parents and resolved != ROOT:
            issues.append(f"{path.relative_to(ROOT)}: link escapes repository: {raw_target}")
        elif not resolved.exists():
            issues.append(f"{path.relative_to(ROOT)}: broken relative link: {raw_target}")


def main() -> int:
    issues: list[str] = []
    notes: dict[str, str] = {}

    for lesson in LESSONS:
        no = lesson["no"]
        directory = CHAPTER / f"{no:02d}-{lesson['slug']}"
        note_path = directory / "README.md"
        notebook_path = directory / "lab.ipynb"
        artifact_path = directory / "artifacts" / "rtx5090-result.json"
        note = note_path.read_text(encoding="utf-8")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        notes[directory.name] = note
        check_relative_links(note_path, note, issues)

        if words(note) < 850:
            issues.append(f"{directory.name}: README has only {words(note)} words")
        if sum(line.startswith("|---") for line in note.splitlines()) < 3:
            issues.append(f"{directory.name}: README needs three explanatory tables")
        if len(re.findall(r"^- \[", note, flags=re.M)) < 2:
            issues.append(f"{directory.name}: README needs at least two primary/official references")
        for heading in REQUIRED_HEADINGS:
            if heading not in note:
                issues.append(f"{directory.name}: missing heading {heading}")
        if no in CHAPTER_01_GUIDES:
            for signal in ("```mermaid", "### Walk it step by step"):
                if signal not in note:
                    issues.append(f"{directory.name}: missing curated visual guide signal {signal}")
        if result_table(no, artifact) not in note:
            issues.append(f"{directory.name}: README result table is not synchronized")
        delivery = DELIVERY[no]
        normalized_note = re.sub(r"\s+", " ", note)
        for field in ("hook", "derivation", "code_walk", "result_reading", "failure", "next"):
            anchor = re.sub(r"\s+", " ", str(delivery[field])).split(".", 1)[0]
            if anchor not in normalized_note:
                issues.append(f"{directory.name}: missing lesson-specific {field} narrative")

        markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        markdown_text = "\n".join(source(cell) for cell in markdown_cells)
        check_relative_links(notebook_path, markdown_text, issues)
        if len(markdown_cells) < 12:
            issues.append(f"{directory.name}: only {len(markdown_cells)} theory cells")
        if words(markdown_text) < 800:
            issues.append(f"{directory.name}: notebook theory has only {words(markdown_text)} words")
        if result_table(no, artifact) not in markdown_text:
            issues.append(f"{directory.name}: notebook result table is not synchronized")
        if no in CHAPTER_01_GUIDES:
            for signal in ("```mermaid", "### Walk it step by step"):
                if signal not in markdown_text:
                    issues.append(f"{directory.name}: notebook missing curated visual guide signal {signal}")
        if len(code_cells) != 3:
            issues.append(f"{directory.name}: expected three readable code stages")
        if any(cell.get("execution_count") is None for cell in code_cells):
            issues.append(f"{directory.name}: notebook contains unexecuted code")
        if any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ):
            issues.append(f"{directory.name}: notebook retains an error output")

    line_counts: Counter[str] = Counter()
    for note in notes.values():
        line_counts.update(set(line.strip() for line in note.splitlines() if line.strip()))
    mass_shared = {line for line, count in line_counts.items() if count >= 25}
    for name, note in notes.items():
        lines = [line.strip() for line in note.splitlines() if line.strip()]
        ratio = sum(line in mass_shared for line in lines) / len(lines)
        if ratio > 0.40:
            issues.append(f"{name}: {ratio:.1%} of non-empty lines remain mass-shared")

    if len(set(notes.values())) != len(notes):
        issues.append("duplicate lesson README bodies detected")

    if issues:
        print("Chapter 01 delivery audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(
        "Chapter 01 delivery audit passed: 29 enriched READMEs, synchronized result "
        "tables, and theory-integrated executed notebooks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
