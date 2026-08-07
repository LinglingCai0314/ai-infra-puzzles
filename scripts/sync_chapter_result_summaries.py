#!/usr/bin/env python3
"""Insert compact checked-in result summaries into generated lesson notes."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "01-mixed-precision-int4"
START = "<!-- rtx5090-result:start -->"
END = "<!-- rtx5090-result:end -->"


def artifact_for(lesson: Path) -> Path | None:
    preferred = lesson / "artifacts" / "rtx5090-result.json"
    if preferred.exists():
        return preferred
    candidates = sorted((lesson / "artifacts").glob("*.json"))
    return candidates[0] if candidates else None


def main() -> int:
    updated = 0
    for lesson in sorted(CHAPTER.glob("[0-9][0-9]-*")):
        if lesson.name.startswith("01-"):
            continue
        artifact = artifact_for(lesson)
        note = lesson / "README.md"
        if artifact is None or not note.exists():
            continue
        data = json.loads(artifact.read_text(encoding="utf-8"))
        env = data["environment"]
        block = (
            f"{START}\n"
            "## Checked-in RTX 5090 result\n\n"
            f"- **Environment:** {env['gpu']}, compute capability {env['compute_capability']}, "
            f"PyTorch {env['torch']}, CUDA runtime {env['cuda_runtime']}\n"
            f"- **Evidence label:** `{data['evidence_label']}`\n"
            f"- **Recorded outcome:** {data['conclusion']}\n\n"
            "The exact shapes, repeated samples, errors, compatibility fields, and "
            "units are preserved in the [JSON artifact](artifacts/rtx5090-result.json) "
            "and the executed notebook output.\n"
            f"{END}"
        )
        text = note.read_text(encoding="utf-8")
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
        else:
            text = text.replace("\n## Explain\n", f"\n{block}\n\n## Explain\n", 1)
        note.write_text(text, encoding="utf-8")
        updated += 1
    print(f"Updated {updated} lesson result summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
