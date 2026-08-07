from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


LESSON_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = Path(
    os.environ.get("CH1_OUTPUT_DIR", str(LESSON_DIR / "outputs" / "raw"))
).expanduser()
ARTIFACT_DIR = Path(
    os.environ.get("CH1_REPORT_DIR", str(LESSON_DIR / "outputs" / "reports"))
).expanduser()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bf16 = load_json(RAW_DIR / "bf16.json")
    int4 = load_json(RAW_DIR / "int4.json")
    if bf16.get("status") != "complete" or int4.get("status") != "complete":
        raise RuntimeError("Both bf16.json and int4.json must be complete")

    bf16_logits_payload = torch.load(RAW_DIR / "bf16_last_logits.pt", map_location="cpu", weights_only=True)
    int4_logits_payload = torch.load(RAW_DIR / "int4_last_logits.pt", map_location="cpu", weights_only=True)
    if bf16_logits_payload["case_ids"] != int4_logits_payload["case_ids"]:
        raise RuntimeError("Quality case ordering differs")
    bf16_logits = bf16_logits_payload["logits"].float()
    int4_logits = int4_logits_payload["logits"].float()
    cosine = F.cosine_similarity(bf16_logits, int4_logits, dim=-1)
    bf16_top1 = bf16_logits.argmax(dim=-1)
    int4_top1 = int4_logits.argmax(dim=-1)

    bf16_rows = {row["id"]: row for row in bf16["quality"]["rows"]}
    int4_rows = {row["id"]: row for row in int4["quality"]["rows"]}
    prediction_agreement = sum(
        bf16_rows[case_id]["prediction"] == int4_rows[case_id]["prediction"]
        for case_id in bf16_logits_payload["case_ids"]
    ) / len(bf16_logits_payload["case_ids"])

    prefill = {}
    for length in sorted(bf16["prefill"], key=int):
        base_ms = bf16["prefill"][length]["median_ms"]
        candidate_ms = int4["prefill"][length]["median_ms"]
        prefill[length] = {
            "bf16_median_ms": base_ms,
            "int4_median_ms": candidate_ms,
            "latency_speedup_bf16_over_int4": ratio(base_ms, candidate_ms),
            "int4_latency_change_pct": pct_change(candidate_ms, base_ms),
        }

    profiler_events = int4["profiler"]["selected_events"]
    int4_event_keys = [event["key"] for event in profiler_events if "int4" in event["key"].lower()]
    comparison = {
        "status": "complete",
        "scope": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware": int4["environment"]["nvidia_smi"],
            "baseline": "BF16 weights and compute",
            "candidate": "TorchAO weight-only INT4, group_size=128, BF16 input/compute",
            "decode_note": "Decode is an approximate decomposition: median generate time minus separately measured median prefill time.",
        },
        "memory": {
            "bf16_storage_gib": bf16["load"]["tensor_storage"]["unique_storage_gib"],
            "int4_storage_gib": int4["load"]["tensor_storage"]["unique_storage_gib"],
            "storage_reduction_pct": 100.0
            * (1.0 - ratio(int4["load"]["tensor_storage"]["unique_storage_gib"], bf16["load"]["tensor_storage"]["unique_storage_gib"])),
            "bf16_allocated_after_load_gib": bf16["load"]["allocated_after_load_gib"],
            "int4_allocated_after_load_gib": int4["load"]["allocated_after_load_gib"],
            "allocated_after_load_reduction_pct": 100.0
            * (1.0 - ratio(int4["load"]["allocated_after_load_gib"], bf16["load"]["allocated_after_load_gib"])),
            "bf16_peak_runtime_allocated_gib": bf16["runtime_memory"]["peak_allocated_gib"],
            "int4_peak_runtime_allocated_gib": int4["runtime_memory"]["peak_allocated_gib"],
        },
        "prefill": prefill,
        "generation": {
            "bf16_total_median_ms": bf16["generate"]["median_ms"],
            "int4_total_median_ms": int4["generate"]["median_ms"],
            "total_latency_speedup_bf16_over_int4": ratio(bf16["generate"]["median_ms"], int4["generate"]["median_ms"]),
            "bf16_approx_decode_tokens_per_s": bf16["generate"]["approximate_decode_tokens_per_s"],
            "int4_approx_decode_tokens_per_s": int4["generate"]["approximate_decode_tokens_per_s"],
            "approx_decode_throughput_ratio_int4_over_bf16": ratio(
                int4["generate"]["approximate_decode_tokens_per_s"],
                bf16["generate"]["approximate_decode_tokens_per_s"],
            ),
        },
        "quality": {
            "count": len(bf16_logits_payload["case_ids"]),
            "bf16_accuracy": bf16["quality"]["accuracy"],
            "int4_accuracy": int4["quality"]["accuracy"],
            "accuracy_delta_int4_minus_bf16": int4["quality"]["accuracy"] - bf16["quality"]["accuracy"],
            "generated_answer_agreement": prediction_agreement,
            "last_logit_top1_agreement": float((bf16_top1 == int4_top1).float().mean()),
            "last_logit_cosine_mean": float(cosine.mean()),
            "last_logit_cosine_min": float(cosine.min()),
            "per_case_logit_cosine": {
                case_id: float(value)
                for case_id, value in zip(bf16_logits_payload["case_ids"], cosine)
            },
        },
        "operator_evidence": {
            "int4_module_count": int4["load"]["int4_module_count"],
            "bf16_int4_module_count": bf16["load"]["int4_module_count"],
            "profiler_int4_event_keys": int4_event_keys,
            "profile_bf16": bf16["profiler"]["table_path"],
            "profile_int4": int4["profiler"]["table_path"],
        },
    }
    (ARTIFACT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# BF16 vs TorchAO INT4 on RTX 5090",
        "",
        f"- Hardware: `{comparison['scope']['hardware']}`",
        f"- Model: `{comparison['scope']['model']}`",
        f"- Candidate: {comparison['scope']['candidate']}",
        "",
        "## Memory",
        "",
        f"- Unique model tensor storage: BF16 `{comparison['memory']['bf16_storage_gib']:.3f} GiB`, INT4 `{comparison['memory']['int4_storage_gib']:.3f} GiB`, reduction `{comparison['memory']['storage_reduction_pct']:.2f}%`.",
        f"- CUDA allocated after load: BF16 `{comparison['memory']['bf16_allocated_after_load_gib']:.3f} GiB`, INT4 `{comparison['memory']['int4_allocated_after_load_gib']:.3f} GiB`, reduction `{comparison['memory']['allocated_after_load_reduction_pct']:.2f}%`.",
        "",
        "## Prefill",
        "",
        "| Sequence | BF16 median ms | INT4 median ms | BF16/INT4 speedup |",
        "|---:|---:|---:|---:|",
    ]
    for length, row in comparison["prefill"].items():
        lines.append(
            f"| {length} | {row['bf16_median_ms']:.3f} | {row['int4_median_ms']:.3f} | {row['latency_speedup_bf16_over_int4']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Generation / approximate Decode",
            "",
            f"- Total generation median: BF16 `{comparison['generation']['bf16_total_median_ms']:.3f} ms`, INT4 `{comparison['generation']['int4_total_median_ms']:.3f} ms`.",
            f"- Approximate Decode throughput: BF16 `{comparison['generation']['bf16_approx_decode_tokens_per_s']:.3f} tok/s`, INT4 `{comparison['generation']['int4_approx_decode_tokens_per_s']:.3f} tok/s`, ratio `{comparison['generation']['approx_decode_throughput_ratio_int4_over_bf16']:.3f}×`.",
            "- Caveat: Decode is approximated by subtracting a separately measured Prefill median from total generation time.",
            "",
            "## Quality sample",
            "",
            f"- Multiple-choice accuracy: BF16 `{comparison['quality']['bf16_accuracy']:.3f}`, INT4 `{comparison['quality']['int4_accuracy']:.3f}` over `{comparison['quality']['count']}` fixed questions.",
            f"- Generated answer agreement: `{comparison['quality']['generated_answer_agreement']:.3f}`.",
            f"- Last-logit top-1 agreement: `{comparison['quality']['last_logit_top1_agreement']:.3f}`; cosine mean `{comparison['quality']['last_logit_cosine_mean']:.6f}`, minimum `{comparison['quality']['last_logit_cosine_min']:.6f}`.",
            "- This small fixed sample does not establish general model quality.",
            "",
            "## Operator evidence",
            "",
            f"- TorchAO INT4 modules: `{comparison['operator_evidence']['int4_module_count']}` (BF16 baseline: `{comparison['operator_evidence']['bf16_int4_module_count']}`).",
            f"- Profiler INT4 keys: `{comparison['operator_evidence']['profiler_int4_event_keys']}`.",
            "",
        ]
    )
    (ARTIFACT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "comparison": str(ARTIFACT_DIR / "comparison.json")}))


if __name__ == "__main__":
    main()
