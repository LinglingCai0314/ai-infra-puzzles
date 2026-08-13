#!/usr/bin/env python3
"""Add or refresh the shared project header in every Markdown document."""

from __future__ import annotations

from pathlib import Path

from markdown_header import HEADER_END, HEADER_START, ROOT, render_markdown_header


def markdown_files(root: Path = ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".git" not in path.relative_to(root).parts
    )


def apply_header(path: Path, root: Path = ROOT) -> bool:
    text = path.read_text(encoding="utf-8")
    expected = render_markdown_header(path, root)

    if text.startswith(HEADER_START):
        marker_end = text.find(HEADER_END)
        if marker_end < 0:
            raise ValueError(f"header start without end marker: {path}")
        remainder = text[marker_end + len(HEADER_END) :].lstrip("\n")
        updated = expected + remainder
    else:
        updated = expected + text.lstrip("\n")

    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    paths = markdown_files()
    changed = sum(apply_header(path) for path in paths)
    print(f"Markdown headers synchronized: {changed} changed, {len(paths)} checked")


if __name__ == "__main__":
    main()
