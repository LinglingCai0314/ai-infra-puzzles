#!/usr/bin/env python3
"""Execute a chapter's notebooks in lesson order and retain their outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", default="01")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    matches = sorted((ROOT / "chapters").glob(f"{args.chapter}-*"))
    if len(matches) != 1:
        raise SystemExit(f"expected one chapter matching {args.chapter}-*, found {len(matches)}")
    chapter = matches[0]
    notebooks = []
    for path in sorted(chapter.glob("[0-9][0-9]-*/lab.ipynb")):
        lesson = int(path.parent.name.split("-", 1)[0])
        if args.start <= lesson <= args.end:
            notebooks.append(path)

    if not notebooks:
        raise SystemExit("no notebooks selected")

    for index, path in enumerate(notebooks, 1):
        print(f"[{index}/{len(notebooks)}] executing {path.relative_to(ROOT)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            allow_errors=False,
            record_timing=True,
        )
        client.execute(cwd=str(path.parent))
        nbformat.write(notebook, path)
    if args.chapter == "01":
        # Refresh tutorial result tables from the artifacts written by the run.
        from enrich_chapter01_delivery import main as enrich_delivery

        enrich_delivery()
    elif args.chapter == "02":
        # Refresh notes and theory cells from the newly written artifacts while
        # retaining outputs for code cells whose reviewed source is unchanged.
        from build_chapter02_lessons import build_chapter

        build_chapter(refresh_chapter_readme=False)
    print(f"Executed {len(notebooks)} notebooks successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
