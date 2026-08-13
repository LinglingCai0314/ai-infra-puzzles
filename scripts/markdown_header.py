#!/usr/bin/env python3
"""Render the shared AI Infra Puzzles header for repository Markdown files."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_START = "<!-- ai-infra-puzzles-header:start -->"
HEADER_END = "<!-- ai-infra-puzzles-header:end -->"


def render_markdown_header(path: Path, root: Path = ROOT) -> str:
    """Return the shared header with a logo path relative to *path*."""

    logo = root / "assets" / "branding" / "logo.png"
    logo_href = os.path.relpath(logo, path.parent).replace(os.sep, "/")
    return f"""{HEADER_START}
<p align="center">
  <img src="{logo_href}" alt="AI Infra Puzzles logo" width="170">
</p>

<h1 align="center">AI Infra Puzzles</h1>

<p align="center">
  <strong>Learn CUDA optimization and LLM inference through hands-on puzzles.</strong>
</p>
{HEADER_END}

"""


def strip_markdown_header(text: str) -> str:
    """Return Markdown body text without the shared header."""

    if not text.startswith(HEADER_START):
        return text
    marker_end = text.find(HEADER_END)
    if marker_end < 0:
        raise ValueError("header start without end marker")
    return text[marker_end + len(HEADER_END) :].lstrip("\n")
