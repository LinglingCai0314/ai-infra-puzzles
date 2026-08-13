#!/usr/bin/env python3
"""Audit Chapter 02's theory, executed notebooks, and retained RTX 5090 evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from build_chapter02_lessons import CHAPTER, LESSONS, notebook, read_artifact, result_table
from markdown_header import strip_markdown_header
from tutorial_guides import CHAPTER_02_GUIDES


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

BANNED_BOILERPLATE = (
    "engineering question is not whether a definition can be repeated",
    "Before opening Lesson",
    "tracks three layers through",
    "The inspectable invariant for",
    "comparison is deliberately small enough",
    "The reader loop for",
)


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def output_text(cell: dict) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        value = output.get("text", "")
        chunks.append("".join(value) if isinstance(value, list) else str(value))
        for item in output.get("data", {}).values():
            chunks.append("".join(item) if isinstance(item, list) else str(item))
    return "\n".join(chunks)


def check_relative_links(path: Path, text: str, issues: list[str]) -> None:
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
    expected_dirs = [f"{spec['no']:02d}-{spec['slug']}" for spec in LESSONS]
    actual_dirs = sorted(path.name for path in CHAPTER.glob("[0-9][0-9]-*") if path.is_dir())
    if actual_dirs != expected_dirs:
        issues.append(f"lesson directory sequence mismatch: {actual_dirs}")

    for spec in LESSONS:
        no = spec["no"]
        directory = CHAPTER / f"{no:02d}-{spec['slug']}"
        note_path = directory / "README.md"
        notebook_path = directory / "lab.ipynb"
        artifact_path = directory / "artifacts" / "rtx5090-result.json"
        if not all(path.exists() for path in (note_path, notebook_path, artifact_path)):
            issues.append(f"{directory.name}: missing README, notebook, or artifact")
            continue

        note = note_path.read_text(encoding="utf-8")
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        note_body = strip_markdown_header(note)
        notes[directory.name] = note_body
        check_relative_links(note_path, note, issues)

        # Concise, lesson-specific prose is preferable to padding every lesson
        # with the same delivery-language paragraphs.
        if words(note_body) < 750:
            issues.append(f"{directory.name}: README has only {words(note_body)} words")
        if sum(line.startswith("|---") for line in note_body.splitlines()) < 3:
            issues.append(f"{directory.name}: README needs three explanatory tables")
        if len(re.findall(r"^- \[", note_body, flags=re.M)) < 2:
            issues.append(f"{directory.name}: README needs at least two official/primary references")
        for heading in REQUIRED_HEADINGS:
            if heading not in note_body:
                issues.append(f"{directory.name}: missing heading {heading}")
        if no in CHAPTER_02_GUIDES:
            for signal in ("```mermaid", "### Walk it step by step"):
                if signal not in note_body:
                    issues.append(f"{directory.name}: missing curated visual guide signal {signal}")
        if result_table(spec, artifact) not in note_body:
            issues.append(f"{directory.name}: README result table is not synchronized")
        normalized_note = re.sub(r"\s+", " ", note_body)
        for field in ("hook", "mechanism", "code_walk", "failure", "next_step"):
            anchor = re.sub(r"\s+", " ", str(spec[field])).split(".", 1)[0]
            if anchor not in normalized_note:
                issues.append(f"{directory.name}: missing lesson-specific {field} narrative")

        if artifact.get("lesson") != no or artifact.get("title") != spec["title"]:
            issues.append(f"{directory.name}: artifact identity mismatch")
        if artifact.get("evidence_label") != spec["evidence_label"]:
            issues.append(f"{directory.name}: artifact evidence label mismatch")
        if not artifact.get("metrics") or not artifact.get("analysis") or not artifact.get("conclusion"):
            issues.append(f"{directory.name}: incomplete structured evidence")
        env = artifact.get("environment", {})
        if "RTX 5090" not in str(env.get("gpu")):
            issues.append(f"{directory.name}: artifact is not an RTX 5090 run")
        for field in ("compute_capability", "torch", "cuda_runtime", "python", "seed"):
            if env.get(field) in (None, ""):
                issues.append(f"{directory.name}: environment missing {field}")

        markdown_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        markdown_text = "\n".join(source(cell) for cell in markdown_cells)
        check_relative_links(notebook_path, markdown_text, issues)
        if len(markdown_cells) < 13:
            issues.append(f"{directory.name}: only {len(markdown_cells)} theory cells")
        if words(markdown_text) < 700:
            issues.append(f"{directory.name}: notebook theory has only {words(markdown_text)} words")
        if result_table(spec, artifact) not in markdown_text:
            issues.append(f"{directory.name}: notebook result table is not synchronized")
        if no in CHAPTER_02_GUIDES:
            for signal in ("```mermaid", "### Walk it step by step"):
                if signal not in markdown_text:
                    issues.append(f"{directory.name}: notebook missing curated visual guide signal {signal}")
        if len(code_cells) != 3:
            issues.append(f"{directory.name}: expected three readable code stages")
        if any(cell.get("execution_count") is None for cell in code_cells):
            issues.append(f"{directory.name}: notebook contains unexecuted code")
        if any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])):
            issues.append(f"{directory.name}: notebook retains an error output")
        retained = "\n".join(output_text(cell) for cell in code_cells)
        if "RTX 5090" not in retained:
            issues.append(f"{directory.name}: retained output does not identify RTX 5090")

        expected = notebook(spec, artifact, nb)
        expected_code = [source(cell) for cell in expected["cells"] if cell["cell_type"] == "code"]
        actual_code = [source(cell) for cell in code_cells]
        if actual_code != expected_code:
            issues.append(f"{directory.name}: notebook code drifted from reviewed source")

    lesson19 = read_artifact(LESSONS[18])
    if lesson19:
        required_native = ("export_succeeded", "checker_passed", "shape_inference_passed", "ort_executed")
        for field in required_native:
            if lesson19["metrics"].get(field) is not True:
                issues.append(f"19-onnx-shape-consistency: native gate {field} did not pass")

    line_counts: Counter[str] = Counter()
    for note in notes.values():
        line_counts.update(set(line.strip() for line in note.splitlines() if line.strip()))
    mass_shared = {line for line, count in line_counts.items() if count >= 24}
    for name, note in notes.items():
        lines = [line.strip() for line in note.splitlines() if line.strip()]
        ratio = sum(line in mass_shared for line in lines) / len(lines)
        if ratio > 0.42:
            issues.append(f"{name}: {ratio:.1%} of non-empty lines remain mass-shared")
    if len(set(notes.values())) != len(notes):
        issues.append("duplicate lesson README bodies detected")
    combined_notes = "\n".join(notes.values())
    for phrase in BANNED_BOILERPLATE:
        if phrase in combined_notes:
            issues.append(f"repetitive template phrase returned: {phrase}")

    if issues:
        print("Chapter 02 delivery audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(
        "Chapter 02 delivery audit passed: 28 concise theory READMEs with curated "
        "visual guides, executed RTX 5090 notebooks, and synchronized artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
