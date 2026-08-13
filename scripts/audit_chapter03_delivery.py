#!/usr/bin/env python3
"""Audit Chapter 03 theory, retained notebooks, and bounded vLLM evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from build_chapter03_lessons import CHAPTER, LESSONS, notebook, result_table
from markdown_header import strip_markdown_header


ROOT = CHAPTER.parents[1]
REQUIRED_HEADINGS = (
    "## Why this puzzle matters", "## Predict before reading the result",
    "## 1. Start from concrete requests and state", "## 2. Derive the mechanism",
    "## 3. Translate the theory into an experiment", "## 4. Read the checked-in RTX 5090 result",
    "## 5. Solve the puzzle and make a decision", "## Reproduce", "## Extend the experiment",
    "## Evidence boundary", "## References",
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
    for raw in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        target = unquote(raw.strip().split("#", 1)[0])
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, flags=re.I): continue
        resolved = (path.parent / target).resolve()
        if ROOT not in resolved.parents and resolved != ROOT:
            issues.append(f"{path.relative_to(ROOT)}: link escapes repository: {raw}")
        elif not resolved.exists():
            issues.append(f"{path.relative_to(ROOT)}: broken relative link: {raw}")


def main() -> int:
    issues: list[str] = []; notes: dict[str, str] = {}; artifacts: dict[int, dict] = {}
    expected = [f"{spec['no']:02d}-{spec['slug']}" for spec in LESSONS]
    actual = sorted(path.name for path in CHAPTER.glob("[0-9][0-9]-*") if path.is_dir())
    if actual != expected: issues.append(f"lesson directory sequence mismatch: {actual}")

    for spec in LESSONS:
        number = spec["no"]; directory = CHAPTER / f"{number:02d}-{spec['slug']}"
        note_path = directory / "README.md"; notebook_path = directory / "lab.ipynb"
        artifact_path = directory / "artifacts" / "rtx5090-result.json"
        if not all(path.exists() for path in (note_path, notebook_path, artifact_path)):
            issues.append(f"{directory.name}: missing README, notebook, or artifact"); continue
        note = note_path.read_text(encoding="utf-8"); nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8")); artifacts[number] = artifact
        note_body = strip_markdown_header(note)
        notes[directory.name] = note_body; check_relative_links(note_path, note, issues)
        if words(note_body) < 640: issues.append(f"{directory.name}: README has only {words(note_body)} words")
        if sum(line.startswith("|---") for line in note_body.splitlines()) < 3:
            issues.append(f"{directory.name}: README needs three explanatory tables")
        if len(re.findall(r"^- \[", note_body, flags=re.M)) < 2:
            issues.append(f"{directory.name}: README needs two official/primary references")
        for heading in REQUIRED_HEADINGS:
            if heading not in note_body: issues.append(f"{directory.name}: missing heading {heading}")
        for signal in ("```mermaid", "### Walk it step by step"):
            if signal not in note_body: issues.append(f"{directory.name}: missing visual guide {signal}")
        if result_table(spec, artifact) not in note_body:
            issues.append(f"{directory.name}: result table is not synchronized")
        normalized = re.sub(r"\s+", " ", note_body)
        for field in ("hook", "mechanism", "code_walk", "failure", "next_step"):
            anchor = re.sub(r"\s+", " ", str(spec[field])).split(".", 1)[0]
            if anchor not in normalized: issues.append(f"{directory.name}: missing {field} narrative")

        if artifact.get("lesson") != number or artifact.get("title") != spec["title"]:
            issues.append(f"{directory.name}: artifact identity mismatch")
        if artifact.get("evidence_label") != spec["evidence_label"]:
            issues.append(f"{directory.name}: evidence label mismatch")
        if not artifact.get("metrics") or not artifact.get("analysis") or not artifact.get("conclusion"):
            issues.append(f"{directory.name}: incomplete structured evidence")
        env = artifact.get("environment", {})
        if "RTX 5090" not in str(env.get("gpu")): issues.append(f"{directory.name}: not an RTX 5090 artifact")
        for field in ("compute_capability", "torch", "cuda_runtime", "python", "vllm", "model_path", "seed"):
            if env.get(field) in (None, ""): issues.append(f"{directory.name}: environment missing {field}")

        markdown_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        markdown_text = "\n".join(source(cell) for cell in markdown_cells)
        check_relative_links(notebook_path, markdown_text, issues)
        if len(markdown_cells) < 13: issues.append(f"{directory.name}: only {len(markdown_cells)} theory cells")
        if words(markdown_text) < 600: issues.append(f"{directory.name}: notebook theory has only {words(markdown_text)} words")
        if result_table(spec, artifact) not in markdown_text:
            issues.append(f"{directory.name}: notebook result table is not synchronized")
        if len(code_cells) != 3: issues.append(f"{directory.name}: expected three code stages")
        if any(cell.get("execution_count") is None for cell in code_cells):
            issues.append(f"{directory.name}: unexecuted code")
        if any(out.get("output_type") == "error" for cell in code_cells for out in cell.get("outputs", [])):
            issues.append(f"{directory.name}: retained error output")
        if "RTX 5090" not in "\n".join(output_text(cell) for cell in code_cells):
            issues.append(f"{directory.name}: retained output lacks RTX 5090")
        expected_nb = notebook(spec, artifact, nb)
        expected_code = [source(cell) for cell in expected_nb["cells"] if cell["cell_type"] == "code"]
        if [source(cell) for cell in code_cells] != expected_code:
            issues.append(f"{directory.name}: notebook code drifted from reviewed source")

    native_gates = {
        7: lambda m: m.get("requests", 0) == 3,
        8: lambda m: m.get("chat_status") == 200 and m.get("schema_valid") is True,
        10: lambda m: m.get("native_load") is True,
        12: lambda m: m.get("warm", {}).get("cached_tokens", 0) > 0,
        18: lambda m: m.get("structured", {}).get("schema_valid") is True,
        21: lambda m: m.get("metrics_status") == 200,
    }
    for number, gate in native_gates.items():
        if number in artifacts and not gate(artifacts[number].get("metrics", {})):
            issues.append(f"lesson {number:02d}: required native gate did not pass")
    if 30 in artifacts:
        final = artifacts[30]["metrics"]
        if final.get("release_ready") is not False or "rollback_rehearsed" not in final.get("blocker_names", []):
            issues.append("lesson 30: lab release must remain blocked on unperformed rollback rehearsal")

    line_counts: Counter[str] = Counter()
    for note in notes.values():
        line_counts.update(set(line.strip() for line in note.splitlines() if line.strip()))
    mass_shared = {line for line, count in line_counts.items() if count >= 25}
    for name, note in notes.items():
        lines = [line.strip() for line in note.splitlines() if line.strip()]
        if sum(line in mass_shared for line in lines) / len(lines) > 0.45:
            issues.append(f"{name}: excessive mass-shared prose")
    if len(set(notes.values())) != len(notes): issues.append("duplicate lesson README bodies")

    if issues:
        print("Chapter 03 delivery audit failed:")
        for issue in issues: print(f"- {issue}")
        return 1
    print("Chapter 03 delivery audit passed: 30 original lessons, diagrams, executed RTX 5090 notebooks, and synchronized artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
