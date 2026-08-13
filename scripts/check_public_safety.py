#!/usr/bin/env python3
"""Fail when a public-repository draft contains common private artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from markdown_header import render_markdown_header

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "models",
    "outputs",
    "reports",
    "venv",
}
FORBIDDEN_TEXT = (
    "/Users/",
    "/root/",
    "Library/Mobile Documents",
    "seetacloud.com",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
)
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024
SELF = Path(__file__).resolve()


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        yield path


def check_markdown_links(path: Path, text: str, issues: list[str]) -> None:
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path = (path.parent / unquote(target)).resolve()
        if not target_path.exists():
            issues.append(f"broken relative link: {path} -> {raw_target}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    issues: list[str] = []
    checked = 0

    for path in iter_files(root):
        checked += 1
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            issues.append(f"file exceeds 10 MiB: {path} ({size} bytes)")

        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"text-like file is not UTF-8: {path}")
            continue

        if path.suffix.lower() == ".md":
            expected_header = render_markdown_header(path, root)
            if not text.startswith(expected_header):
                issues.append(f"missing or stale Markdown header: {path}")

        if path.resolve() != SELF:
            for token in FORBIDDEN_TEXT:
                if token in text:
                    issues.append(f"private token {token!r}: {path}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    issues.append(f"possible secret matching {pattern.pattern!r}: {path}")

        if path.suffix.lower() == ".md":
            check_markdown_links(path, text, issues)
        if path.suffix.lower() in {".ipynb", ".json"}:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid JSON: {path}: {exc}")

    if issues:
        print("Public-safety validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Public-safety validation passed for {checked} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
