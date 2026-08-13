#!/usr/bin/env python3
"""Audit the bilingual Chapter 05 notes, kernels, notebooks, and RTX 5090 evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from build_chapter05_lessons import CHAPTER, CHAPTER_ZH, LESSONS, notebook, result_table
from markdown_header import strip_markdown_header


ROOT = CHAPTER.parents[1]


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


def main() -> int:
    issues: list[str] = []
    expected = [f"{spec['no']:02d}-{spec['slug']}" for spec in LESSONS]
    actual = sorted(path.name for path in CHAPTER.glob("[0-9][0-9]-*") if path.is_dir())
    if actual != expected: issues.append(f"lesson directory sequence mismatch: {actual}")

    runtime = (ROOT / "scripts" / "chapter05_runtime.py").read_text(encoding="utf-8")
    for token in ("@triton.jit", "softmax_kernel", "matmul_kernel", "rmsnorm_kernel",
                  "paged_gather_kernel", "persistent_affine_kernel", "gelu_kernel"):
        if token not in runtime: issues.append(f"shared runtime missing {token}")
    cuda_source = CHAPTER / "05-explicit-cuda-control" / "vector_affine.cu"
    if not cuda_source.exists() or "cudaGetLastError" not in cuda_source.read_text(encoding="utf-8"):
        issues.append("explicit CUDA source or launch error check is missing")

    artifacts: dict[int, dict] = {}
    english_bodies: set[str] = set()
    for spec in LESSONS:
        lesson_name = f"{spec['no']:02d}-{spec['slug']}"
        path = CHAPTER / lesson_name
        path_zh = CHAPTER_ZH / lesson_name
        note_path = path / "README.md"
        note_zh_path = path_zh / "README.md"
        nb_path = path / "lab.ipynb"
        artifact_path = path / "artifacts" / "rtx5090-result.json"
        required = (note_path, note_zh_path, nb_path, artifact_path)
        for item in required:
            if not item.exists(): issues.append(f"{lesson_name}: missing {item.name}")
        if not all(item.exists() for item in required): continue

        note = note_path.read_text(encoding="utf-8")
        note_zh = note_zh_path.read_text(encoding="utf-8")
        body = strip_markdown_header(note); body_zh = strip_markdown_header(note_zh)
        english_bodies.add(body)
        if len(re.findall(r"[A-Za-z0-9_]+", body)) < 850: issues.append(f"{lesson_name}: English note is too short")
        if len(body_zh.encode("utf-8")) < 6000: issues.append(f"{lesson_name}: Chinese note is too short")
        for heading in ("## Why this puzzle matters", "## Predict before running", "## 1. Build the mechanism",
                        "## 2. Compare Triton with CUDA or the library path", "## 3. Turn theory into an experiment",
                        "## 4. Read the checked-in RTX 5090 result", "## 5. Make the bounded decision",
                        "## Reproduce", "## Evidence boundary", "## References"):
            if heading not in body: issues.append(f"{lesson_name}: missing English heading {heading}")
        for heading in ("## 为什么值得研究", "## 运行前先预测", "## 1. 建立机制",
                        "## 2. 对比 Triton 与 CUDA 或库函数路径", "## 3. 把理论变成实验",
                        "## 4. 阅读仓库中的 RTX 5090 结果", "## 5. 得出有边界的结论",
                        "## 复现", "## 证据边界", "## 参考资料"):
            if heading not in body_zh: issues.append(f"{lesson_name}: missing Chinese heading {heading}")

        artifact = json.loads(artifact_path.read_text(encoding="utf-8")); artifacts[spec["no"]] = artifact
        if artifact.get("lesson") != spec["no"] or artifact.get("title") != spec["title_en"]:
            issues.append(f"{lesson_name}: artifact identity mismatch")
        if artifact.get("evidence_label") != spec["evidence_label"]:
            issues.append(f"{lesson_name}: evidence label mismatch")
        if not artifact.get("analysis_en") or not artifact.get("analysis_zh") or not artifact.get("conclusion"):
            issues.append(f"{lesson_name}: bilingual analysis or conclusion missing")
        metrics = artifact.get("metrics", {})
        for key in ("primary", "secondary", "max_abs_error", "passed", "details"):
            if key not in metrics: issues.append(f"{lesson_name}: metric {key} missing")
        env = artifact.get("environment", {})
        for key in ("gpu", "compute_capability", "torch", "cuda_runtime", "triton", "triton_target", "python", "seed"):
            if env.get(key) in (None, ""): issues.append(f"{lesson_name}: environment {key} missing")
        if "RTX 5090" not in str(env.get("gpu")): issues.append(f"{lesson_name}: artifact is not RTX 5090")
        if result_table(spec, artifact, "en") not in note: issues.append(f"{lesson_name}: English result table is stale")
        if result_table(spec, artifact, "zh") not in note_zh: issues.append(f"{lesson_name}: Chinese result table is stale")

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        markdown_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        if len(markdown_cells) < 14: issues.append(f"{lesson_name}: notebook theory is too thin")
        if len(code_cells) != 3: issues.append(f"{lesson_name}: expected three code stages")
        if any(cell.get("execution_count") is None for cell in code_cells): issues.append(f"{lesson_name}: unexecuted code cell")
        if any(out.get("output_type") == "error" for cell in code_cells for out in cell.get("outputs", [])):
            issues.append(f"{lesson_name}: retained error output")
        if "RTX 5090" not in "\n".join(output_text(cell) for cell in code_cells):
            issues.append(f"{lesson_name}: output lacks RTX 5090 identity")
        rebuilt = notebook(spec, artifact, nb)
        expected_code = [source(cell) for cell in rebuilt["cells"] if cell["cell_type"] == "code"]
        if [source(cell) for cell in code_cells] != expected_code:
            issues.append(f"{lesson_name}: notebook code drifted from reviewed source")

    if len(english_bodies) != len(LESSONS): issues.append("duplicate English lesson bodies")
    for path, token in ((ROOT / "README.md", "chapters/05-triton-gpu-programming/README.md"),
                        (ROOT / "README_ZH.md", "chapters-zh/05-triton-gpu-programming/README.md"),
                        (ROOT / "chapters-zh" / "README.md", "05-triton-gpu-programming/README.md")):
        if token not in path.read_text(encoding="utf-8"): issues.append(f"navigation missing from {path.name}")

    if len(artifacts) == len(LESSONS):
        gates = [
            (artifacts[4]["metrics"]["details"]["tail"] != 0, "lesson 04 awkward tail"),
            (artifacts[5]["metrics"]["primary"] is False, "lesson 05 nvcc boundary"),
            (artifacts[7]["metrics"]["secondary"] > artifacts[7]["metrics"]["primary"], "lesson 07 mask identity"),
            (artifacts[11]["metrics"]["max_abs_error"] < 1e-4, "lesson 11 softmax correctness"),
            (artifacts[19]["metrics"]["primary"] > 0, "lesson 19 overflow probe"),
            (artifacts[23]["metrics"]["max_abs_error"] == 0, "lesson 23 paged gather"),
            (artifacts[24]["metrics"]["details"]["rocm_executed"] is False, "lesson 24 ROCm boundary"),
            (artifacts[26]["metrics"]["details"]["tma_used"] is False, "lesson 26 TMA boundary"),
            (isinstance(artifacts[30]["metrics"]["passed"], bool), "lesson 30 delivery gate recorded"),
        ]
        for passed, label in gates:
            if not passed: issues.append(f"failed mechanism gate: {label}")

    if issues:
        print("Chapter 05 delivery audit failed:")
        for issue in issues: print(f"- {issue}")
        return 1
    print("Chapter 05 delivery audit passed: 30 bilingual lessons, reviewed Triton/CUDA source, executed RTX 5090 notebooks, and synchronized artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
