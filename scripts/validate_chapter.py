#!/usr/bin/env python3
"""Validate lesson structure, executed notebooks, artifacts, and navigation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_LABELS = {
    "native-backend",
    "pytorch-gpu",
    "numerical-model",
    "capacity-model",
    "compatibility-probe",
}


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def output_text(cell: dict) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        text = output.get("text", "")
        chunks.append("".join(text) if isinstance(text, list) else str(text))
        data = output.get("data", {})
        for value in data.values():
            chunks.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(chunks)


def main() -> int:
    number = sys.argv[1] if len(sys.argv) > 1 else "01"
    matches = sorted((ROOT / "chapters").glob(f"{number}-*"))
    issues: list[str] = []
    if len(matches) != 1:
        print(f"expected one chapter matching {number}-*, found {len(matches)}")
        return 1
    chapter = matches[0]
    lessons = sorted(path for path in chapter.glob("[0-9][0-9]-*") if path.is_dir())
    expected_counts = {"01": 30, "02": 28, "03": 30, "04": 17, "05": 30}
    expected = list(range(1, expected_counts[number] + 1)) if number in expected_counts else None
    actual = [int(path.name.split("-", 1)[0]) for path in lessons]
    if expected and actual != expected:
        issues.append(f"lesson sequence mismatch: {actual}")

    chapter_text = (chapter / "README.md").read_text(encoding="utf-8")
    for lesson in lessons:
        rel = lesson.relative_to(ROOT)
        note = lesson / "README.md"
        notebook = lesson / "lab.ipynb"
        artifact = lesson / "artifacts" / "rtx5090-result.json"
        if not note.exists() or note.stat().st_size < 3000:
            issues.append(f"missing or short note: {rel}")
        else:
            note_text = note.read_text(encoding="utf-8")
            note_lower = note_text.lower()
            required_note_signals = {
                "prediction": ("predict", "prediction"),
                "experiment": ("experiment", "measurement protocol"),
                "reproduction": ("reproduce",),
                "evidence boundary": ("evidence boundary",),
                "references": ("references",),
                "notebook link": ("lab.ipynb",),
            }
            for signal, alternatives in required_note_signals.items():
                if not any(value in note_lower for value in alternatives):
                    issues.append(f"note missing {signal}: {rel}")
        if not notebook.exists():
            issues.append(f"missing notebook: {rel}")
            continue
        if not artifact.exists():
            issues.append(f"missing canonical JSON artifact: {rel}")

        try:
            nb = json.loads(notebook.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"invalid notebook JSON: {rel}: {exc}")
            continue
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        markdown_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "markdown"]
        markdown_text = "\n".join(cell_source(cell) for cell in markdown_cells).lower()
        if not code_cells:
            issues.append(f"no code cells: {rel}")
        notebook_signals = {
            "prediction": ("predict",),
            "theory bridge": ("theory", "mechanism"),
            "inspection": ("inspect",),
            "bounded explanation": ("explain", "decision", "conclusion"),
        }
        for signal, alternatives in notebook_signals.items():
            if not any(value in markdown_text for value in alternatives):
                issues.append(f"notebook missing {signal}: {rel}")
        for i, cell in enumerate(code_cells):
            if cell.get("execution_count") is None:
                issues.append(f"unexecuted code cell {i}: {rel}")
            if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                issues.append(f"error output in code cell {i}: {rel}")
        retained_output = "\n".join(output_text(cell) for cell in code_cells)
        if not retained_output.strip():
            issues.append(f"notebook has no retained output: {rel}")
        if "RTX 5090" not in retained_output and "GeForce RTX 5090" not in retained_output:
            issues.append(f"notebook output does not identify RTX 5090: {rel}")

        if artifact.exists():
            try:
                data = json.loads(artifact.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"invalid artifact {artifact.relative_to(ROOT)}: {exc}")
            else:
                serialized = json.dumps(data)
                lesson_number = int(lesson.name.split("-", 1)[0])
                if data.get("lesson") != lesson_number:
                    issues.append(f"artifact lesson number mismatch: {artifact.relative_to(ROOT)}")
                if data.get("evidence_label") not in EVIDENCE_LABELS:
                    issues.append(f"artifact has invalid evidence label: {artifact.relative_to(ROOT)}")
                if not str(data.get("conclusion", "")).strip():
                    issues.append(f"artifact has no bounded conclusion: {artifact.relative_to(ROOT)}")
                environment = data.get("environment", {})
                for field in ("gpu", "compute_capability", "torch", "cuda_runtime"):
                    if not environment.get(field):
                        issues.append(f"artifact environment missing {field}: {artifact.relative_to(ROOT)}")
                if "RTX 5090" not in serialized and "GeForce RTX 5090" not in serialized:
                    issues.append(f"artifact does not identify RTX 5090: {artifact.relative_to(ROOT)}")

        if f"{lesson.name}/README.md" not in chapter_text:
            issues.append(f"chapter navigation missing lesson: {lesson.name}")

    project_text = (ROOT / "README.md").read_text(encoding="utf-8")
    chapter_link = f"{chapter.relative_to(ROOT)}/README.md"
    if chapter_link not in project_text:
        issues.append(f"project README does not link Chapter {number}")

    if issues:
        print("Chapter validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Chapter validation passed: {chapter.relative_to(ROOT)} ({len(lessons)} lessons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
