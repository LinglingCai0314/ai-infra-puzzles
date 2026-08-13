#!/usr/bin/env python3
"""Audit Chapter 04 bilingual theory, visuals, notebooks, and RTX 5090 evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from build_chapter04_lessons import (
    CHAPTER,
    CHAPTER_ZH,
    LESSONS,
    notebook,
    result_table,
)
from markdown_header import strip_markdown_header


ROOT = CHAPTER.parents[1]
EXPECTED_ASSETS = {
    "1T1C_DRAM_Cell.png",
    "GPU_circuit_structures_from_L2_A4_landscape.pdf",
    "GPU_on_chip_structures_attention_acceleration.png",
    "HBM_circuit_to_gpu_connection.png",
    "HBM_circuit_to_gpu_connection_A4_portrait.pdf",
    "L2_cache_slice_circuit_structure.png",
    "L2_cache_slice_circuit_structure_A4_portrait.pdf",
    "NVIDIA_GPU_parameter_quick_table.png",
    "NoC_and_SM_circuit_structures_A4_portrait.pdf",
    "NoC_on_chip_network_circuit_structure.png",
    "NoC_on_chip_network_circuit_structure_A4_portrait.pdf",
    "SM_compute_partition_circuit_structure.png",
    "visualizations/cmos-inverter.html",
    "visualizations/cmos-inverter.png",
    "visualizations/dram-1t1c-read-mechanism.html",
    "visualizations/dram-1t1c-read-mechanism.png",
    "visualizations/gpu-memory-spatial-layout.html",
    "visualizations/gpu-memory-spatial-layout.png",
}


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def cell_source(cell: dict) -> str:
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
    expected_names = [f"{spec['no']:02d}-{spec['slug']}" for spec in LESSONS]
    actual_names = sorted(path.name for path in CHAPTER.glob("[0-9][0-9]-*") if path.is_dir())
    if actual_names != expected_names:
        issues.append(f"lesson directory sequence mismatch: {actual_names}")

    actual_assets = {
        str(path.relative_to(CHAPTER / "assets"))
        for path in (CHAPTER / "assets").rglob("*") if path.is_file()
    }
    if actual_assets != EXPECTED_ASSETS:
        issues.append(f"asset set mismatch: missing={sorted(EXPECTED_ASSETS - actual_assets)}, extra={sorted(actual_assets - EXPECTED_ASSETS)}")

    reference_corpus = []
    artifacts: dict[int, dict] = {}
    bodies: set[str] = set()
    for spec in LESSONS:
        name = f"{spec['no']:02d}-{spec['slug']}"
        directory = CHAPTER / name
        directory_zh = CHAPTER_ZH / name
        note_path = directory / "README.md"
        note_zh_path = directory_zh / "README.md"
        notebook_path = directory / "lab.ipynb"
        artifact_path = directory / "artifacts" / "rtx5090-result.json"
        for path in (note_path, note_zh_path, notebook_path, artifact_path):
            if not path.exists(): issues.append(f"{name}: missing {path.name}")
        if issues and not all(path.exists() for path in (note_path, note_zh_path, notebook_path, artifact_path)):
            continue

        note = note_path.read_text(encoding="utf-8")
        note_zh = note_zh_path.read_text(encoding="utf-8")
        body = strip_markdown_header(note)
        body_zh = strip_markdown_header(note_zh)
        bodies.add(body)
        reference_corpus.extend((note, note_zh))
        if words(body) < 580: issues.append(f"{name}: English README has only {words(body)} words")
        if len(body_zh.encode("utf-8")) < 3500: issues.append(f"{name}: Chinese README is too short")
        for heading in ("## Why this puzzle matters", "## Predict before running",
                        "## 1. Put the mechanism in physical space", "## 3. Turn theory into an experiment",
                        "## 4. Read the checked-in RTX 5090 result", "## 5. Make the bounded decision",
                        "## Reproduce", "## Evidence boundary", "## References"):
            if heading not in body: issues.append(f"{name}: missing English heading {heading}")
        for heading in ("## 为什么值得研究", "## 运行前先预测", "## 1. 把机制放回物理空间",
                        "## 3. 把理论变成实验", "## 4. 阅读仓库中的 RTX 5090 结果",
                        "## 5. 得出有边界的结论", "## 复现", "## 证据边界", "## 参考资料"):
            if heading not in body_zh: issues.append(f"{name}: missing Chinese heading {heading}")

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifacts[spec["no"]] = artifact
        if artifact.get("lesson") != spec["no"] or artifact.get("title") != spec["title_en"]:
            issues.append(f"{name}: artifact identity mismatch")
        if artifact.get("evidence_label") != spec["evidence_label"]:
            issues.append(f"{name}: artifact evidence label mismatch")
        if not artifact.get("metrics") or not artifact.get("analysis") or not artifact.get("conclusion"):
            issues.append(f"{name}: incomplete artifact")
        env = artifact.get("environment", {})
        for field in ("gpu", "compute_capability", "torch", "cuda_runtime", "python", "seed"):
            if env.get(field) in (None, ""): issues.append(f"{name}: environment missing {field}")
        if "RTX 5090" not in str(env.get("gpu")): issues.append(f"{name}: artifact is not from RTX 5090")
        if result_table(spec, artifact, "en") not in note:
            issues.append(f"{name}: English result table is not synchronized")
        if result_table(spec, artifact, "zh") not in note_zh:
            issues.append(f"{name}: Chinese result table is not synchronized")

        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        reference_corpus.append("\n".join(cell_source(cell) for cell in markdown_cells))
        if len(markdown_cells) < 13: issues.append(f"{name}: notebook theory is too thin")
        if len(code_cells) != 3: issues.append(f"{name}: expected three code stages")
        if any(cell.get("execution_count") is None for cell in code_cells):
            issues.append(f"{name}: unexecuted code cell")
        if any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])):
            issues.append(f"{name}: retained error output")
        if "RTX 5090" not in "\n".join(output_text(cell) for cell in code_cells):
            issues.append(f"{name}: retained output lacks RTX 5090 identity")
        expected_nb = notebook(spec, artifact, nb)
        expected_code = [cell_source(cell) for cell in expected_nb["cells"] if cell["cell_type"] == "code"]
        if [cell_source(cell) for cell in code_cells] != expected_code:
            issues.append(f"{name}: notebook code drifted from reviewed source")

    corpus = "\n".join(reference_corpus + [
        (CHAPTER / "README.md").read_text(encoding="utf-8"),
        (CHAPTER_ZH / "README.md").read_text(encoding="utf-8"),
    ])
    for asset in EXPECTED_ASSETS:
        if asset not in corpus:
            issues.append(f"asset is never referenced: {asset}")
    if len(bodies) != len(LESSONS): issues.append("duplicate English lesson bodies")

    if artifacts:
        checks = [
            (abs(artifacts[1]["metrics"]["voltage_energy_ratio"] - 1.5625) < 1e-9, "lesson 01 voltage ratio"),
            (artifacts[2]["metrics"]["fresh_margin_mv"] > artifacts[2]["metrics"]["leaked_margin_mv"], "lesson 02 margin ordering"),
            (artifacts[7]["metrics"]["memory_technology"] == "GDDR7", "lesson 07 memory identity"),
            (artifacts[9]["metrics"]["hotspot"]["mean_latency_ticks"] > artifacts[9]["metrics"]["balanced"]["mean_latency_ticks"], "lesson 09 hotspot latency"),
            (artifacts[10]["metrics"]["bank_multiplicity"]["stride_32"] == 32, "lesson 10 bank model"),
            (artifacts[12]["metrics"]["patterns"]["uniform"] > artifacts[12]["metrics"]["patterns"]["half_warp"], "lesson 12 divergence model"),
            (artifacts[13]["metrics"]["output_equivalent"] is True, "lesson 13 output equivalence"),
            (artifacts[16]["metrics"]["complete_artifacts"] == 15, "lesson 16 portfolio completeness"),
            (artifacts[17]["metrics"]["bandwidth_formula_error"] < 1e-12, "lesson 17 bandwidth audit"),
        ]
        for passed, label in checks:
            if not passed: issues.append(f"failed mechanism gate: {label}")

    if issues:
        print("Chapter 04 delivery audit failed:")
        for issue in issues: print(f"- {issue}")
        return 1
    print("Chapter 04 delivery audit passed: 17 bilingual lessons, 18 source visuals, executed RTX 5090 notebooks, and synchronized artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
