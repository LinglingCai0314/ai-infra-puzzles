from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchao
import transformers
from torch.profiler import ProfilerActivity, profile
from torchao.quantization.linear_quant_modules import (
    Int4WeightOnlyQuantizer,
    WeightOnlyInt4Linear,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


LESSON_DIR = Path(__file__).resolve().parents[1]
MODEL_SOURCE = os.environ.get("CH1_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
RAW_DIR = Path(
    os.environ.get("CH1_OUTPUT_DIR", str(LESSON_DIR / "outputs" / "raw"))
).expanduser()
MANIFEST_PATH = Path(
    os.environ.get("CH1_MODEL_MANIFEST", str(RAW_DIR / "model_manifest.json"))
).expanduser()
LOCAL_FILES_ONLY = os.environ.get("CH1_LOCAL_FILES_ONLY", "0") == "1"
SEED = 20260807


QUALITY_CASES = [
    {"id": "q01", "q": "2 + 3 等于多少？", "choices": ["A. 4", "B. 5", "C. 6", "D. 7"], "answer": "B"},
    {"id": "q02", "q": "法国的首都是哪座城市？", "choices": ["A. 伦敦", "B. 柏林", "C. 巴黎", "D. 罗马"], "answer": "C"},
    {"id": "q03", "q": "在标准大气压下，水的冰点是多少摄氏度？", "choices": ["A. 0", "B. 50", "C. 100", "D. -100"], "answer": "A"},
    {"id": "q04", "q": "12 ÷ 3 等于多少？", "choices": ["A. 2", "B. 3", "C. 5", "D. 4"], "answer": "D"},
    {"id": "q05", "q": "Python 列表末尾添加一个元素最常用的方法是？", "choices": ["A. append", "B. push", "C. add", "D. insertEnd"], "answer": "A"},
    {"id": "q06", "q": "二进制 1010 对应的十进制数是多少？", "choices": ["A. 8", "B. 10", "C. 12", "D. 14"], "answer": "B"},
    {"id": "q07", "q": "地球的天然卫星是？", "choices": ["A. 火星", "B. 金星", "C. 月球", "D. 太阳"], "answer": "C"},
    {"id": "q08", "q": "在常见二进制计量中，1 KiB 等于多少字节？", "choices": ["A. 100", "B. 512", "C. 1000", "D. 1024"], "answer": "D"},
    {"id": "q09", "q": "逻辑与 AND 的两个输入在什么情况下输出真？", "choices": ["A. 至少一个为真", "B. 两个都为假", "C. 两个不同", "D. 两个都为真"], "answer": "D"},
    {"id": "q10", "q": "9 × 7 等于多少？", "choices": ["A. 63", "B. 56", "C. 72", "D. 49"], "answer": "A"},
    {"id": "q11", "q": "中国的首都是哪座城市？", "choices": ["A. 上海", "B. 北京", "C. 广州", "D. 深圳"], "answer": "B"},
    {"id": "q12", "q": "下面哪个是速度的国际单位？", "choices": ["A. kg", "B. s", "C. m", "D. m/s"], "answer": "D"},
    {"id": "q13", "q": "均匀硬币抛一次出现正面的概率是？", "choices": ["A. 0", "B. 1", "C. 1/2", "D. 1/4"], "answer": "C"},
    {"id": "q14", "q": "81 的算术平方根是多少？", "choices": ["A. 9", "B. 8", "C. 7", "D. 6"], "answer": "A"},
    {"id": "q15", "q": "HTML 中创建超链接使用哪个标签？", "choices": ["A. <p>", "B. <a>", "C. <img>", "D. <div>"], "answer": "B"},
    {"id": "q16", "q": "函数 f(x)=x² 的导数是？", "choices": ["A. x", "B. x²", "C. 2", "D. 2x"], "answer": "D"},
    {"id": "q17", "q": "下面哪个数是质数？", "choices": ["A. 21", "B. 27", "C. 29", "D. 33"], "answer": "C"},
    {"id": "q18", "q": "RGB 属于哪一种颜色混合模型？", "choices": ["A. 加色模型", "B. 减色模型", "C. 灰度模型", "D. 索引模型"], "answer": "A"},
    {"id": "q19", "q": "GPU 中 HBM 的英文全称最接近哪一项？", "choices": ["A. High Bus Module", "B. High Bandwidth Memory", "C. Hybrid Binary Matrix", "D. Hardware Buffer Map"], "answer": "B"},
    {"id": "q20", "q": "有符号对称 INT4 常用的整数范围是？", "choices": ["A. 0 到 15", "B. -16 到 15", "C. -7 到 8", "D. -8 到 7"], "answer": "D"},
]


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stats_ms(values_s: list[float]) -> dict[str, Any]:
    values_ms = [v * 1000.0 for v in values_s]
    return {
        "samples_ms": values_ms,
        "median_ms": statistics.median(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "p05_ms": percentile(values_ms, 0.05),
        "p95_ms": percentile(values_ms, 0.95),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def tensor_storage_bytes(model: torch.nn.Module) -> dict[str, Any]:
    seen: set[tuple[int, int]] = set()
    total = 0
    tensors = []
    for kind, named in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in named:
            try:
                storage = tensor.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                nbytes = storage.nbytes()
            except Exception:
                key = (id(tensor), tensor.numel() * tensor.element_size())
                nbytes = tensor.numel() * tensor.element_size()
            if key in seen:
                continue
            seen.add(key)
            total += int(nbytes)
            tensors.append({"kind": kind, "name": name, "dtype": str(tensor.dtype), "nbytes": int(nbytes)})
    return {"unique_storage_bytes": total, "unique_storage_gib": total / (1024**3), "tensor_count": len(tensors)}


def dtype_histogram(model: torch.nn.Module) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for tensor in list(model.parameters()) + list(model.buffers()):
        hist[str(tensor.dtype)] += tensor.numel()
    return dict(sorted(hist.items()))


def make_ids(tokenizer: Any, length: int, batch: int = 1) -> torch.Tensor:
    text = (
        "低精度推理需要同时记录权重格式、激活格式、KV Cache、目标 kernel、"
        "显存峰值、Prefill、Decode 和输出质量。"
    )
    base = tokenizer(text, add_special_tokens=False)["input_ids"]
    repeats = math.ceil(length / len(base))
    ids = torch.tensor((base * repeats)[:length], dtype=torch.long)
    return ids.unsqueeze(0).repeat(batch, 1).cuda()


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def timed_forward(model: torch.nn.Module, ids: torch.Tensor, warmups: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmups):
        out = model(input_ids=ids, use_cache=True)
        del out
    synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        synchronize()
        samples.append(time.perf_counter() - start)
        del out
    result = stats_ms(samples)
    result["batch"] = ids.shape[0]
    result["sequence_length"] = ids.shape[1]
    result["tokens_per_s_from_median"] = ids.numel() / (result["median_ms"] / 1000.0)
    return result


@torch.inference_mode()
def timed_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    ids: torch.Tensor,
    new_tokens: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for _ in range(warmups):
        out = model.generate(ids, **kwargs)
        del out
    synchronize()
    samples = []
    generated_counts = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = model.generate(ids, **kwargs)
        synchronize()
        samples.append(time.perf_counter() - start)
        generated_counts.append(int(out.shape[-1] - ids.shape[-1]))
        del out
    result = stats_ms(samples)
    result.update(
        {
            "batch": ids.shape[0],
            "input_length": ids.shape[1],
            "requested_new_tokens": new_tokens,
            "generated_counts": generated_counts,
            "average_new_tokens_per_s_from_median": statistics.median(generated_counts)
            / (result["median_ms"] / 1000.0),
        }
    )
    return result


def quality_prompt(case: dict[str, Any]) -> str:
    return (
        f"问题：{case['q']}\n"
        + "\n".join(case["choices"])
        + "\n只输出 A、B、C、D 中的一个大写字母，不要解释。"
    )


def chat_ids(tokenizer: Any, prompt_text: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": "你是严谨的选择题助手，严格遵守输出格式。"},
        {"role": "user", "content": prompt_text},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    return ids.cuda()


@torch.inference_mode()
def run_quality(model: torch.nn.Module, tokenizer: Any, cases: list[dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor]:
    rows = []
    logits_rows = []
    for case in cases:
        ids = chat_ids(tokenizer, quality_prompt(case))
        forward = model(input_ids=ids, use_cache=False)
        last_logits = forward.logits[0, -1].float().cpu()
        logits_rows.append(last_logits)
        top1_token_id = int(last_logits.argmax())
        top1_token = tokenizer.decode([top1_token_id])
        del forward

        generated = model.generate(
            ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0, ids.shape[-1] :], skip_special_tokens=True).strip()
        match = re.search(r"[ABCD]", text.upper())
        prediction = match.group(0) if match else None
        rows.append(
            {
                "id": case["id"],
                "question": case["q"],
                "answer": case["answer"],
                "prediction": prediction,
                "correct": prediction == case["answer"],
                "generated_text": text,
                "last_logit_top1_token_id": top1_token_id,
                "last_logit_top1_token": top1_token,
            }
        )
        del generated, ids, last_logits
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    return {"count": len(rows), "accuracy": accuracy, "rows": rows}, torch.stack(logits_rows)


@torch.inference_mode()
def run_profiler(model: torch.nn.Module, ids: torch.Tensor, mode: str) -> dict[str, Any]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = model(input_ids=ids, use_cache=False)
        synchronize()
        del out
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
    (RAW_DIR / f"profile_{mode}.txt").write_text(table + "\n", encoding="utf-8")
    selected = []
    for event in prof.key_averages():
        key = str(event.key)
        if any(token in key.lower() for token in ("int4", "pack", "linear", "matmul", "mm")):
            selected.append(
                {
                    "key": key,
                    "count": int(event.count),
                    "self_cuda_time_total_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                    "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
                }
            )
    selected.sort(key=lambda x: x["self_cuda_time_total_us"], reverse=True)
    return {"selected_events": selected[:80], "table_path": str(RAW_DIR / f"profile_{mode}.txt")}


def environment_payload() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "torchao": torchao.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_properties_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,pstate",
                "--format=csv,noheader",
            ]
        ),
        "model_manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else None,
    }


def load_model(mode: str) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SOURCE,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=LOCAL_FILES_ONLY,
    ).cuda().eval()
    synchronize()
    bf16_load_s = time.perf_counter() - start
    quantize_s = None
    if mode == "int4":
        start = time.perf_counter()
        quantizer = Int4WeightOnlyQuantizer(
            groupsize=128,
            padding_allowed=True,
            inner_k_tiles=8,
            device=torch.device("cuda"),
            precision=torch.bfloat16,
        )
        model = quantizer.quantize(model).eval()
        synchronize()
        quantize_s = time.perf_counter() - start
    int4_module_count = sum(isinstance(module, WeightOnlyInt4Linear) for module in model.modules())
    metadata = {
        "bf16_load_s": bf16_load_s,
        "int4_quantize_s": quantize_s,
        "allocated_after_load_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_after_load_gib": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated_during_load_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_during_load_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "tensor_storage": tensor_storage_bytes(model),
        "dtype_histogram_numel": dtype_histogram(model),
        "int4_module_count": int4_module_count,
        "module_class_histogram": dict(Counter(type(module).__name__ for module in model.modules()).most_common()),
    }
    return model, tokenizer, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "int4"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"smoke_{args.mode}" if args.smoke else args.mode
    out_path = RAW_DIR / f"{suffix}.json"
    logits_path = RAW_DIR / f"{suffix}_last_logits.pt"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    payload: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "smoke": args.smoke,
        "seed": SEED,
        "started_at_unix": time.time(),
    }
    try:
        payload["environment"] = environment_payload()
        model, tokenizer, load_metadata = load_model(args.mode)
        payload["load"] = load_metadata

        # Keep the runtime peak separate from the temporary peak incurred while
        # loading BF16 weights and (for the candidate) packing INT4 weights.
        torch.cuda.reset_peak_memory_stats()
        payload["runtime_memory_baseline"] = {
            "allocated_before_inference_gib": torch.cuda.memory_allocated() / (1024**3),
            "reserved_before_inference_gib": torch.cuda.memory_reserved() / (1024**3),
        }

        if args.smoke:
            lengths, warmups, repeats = [128], 1, 2
            decode_new_tokens, decode_warmups, decode_repeats = 8, 1, 1
            cases = QUALITY_CASES[:2]
        else:
            lengths, warmups, repeats = [128, 512, 1024], 3, 10
            decode_new_tokens, decode_warmups, decode_repeats = 64, 2, 5
            cases = QUALITY_CASES

        payload["prefill"] = {}
        for length in lengths:
            ids = make_ids(tokenizer, length)
            payload["prefill"][str(length)] = timed_forward(model, ids, warmups, repeats)
            del ids

        decode_ids = make_ids(tokenizer, 128)
        payload["generate"] = timed_generate(
            model,
            tokenizer,
            decode_ids,
            decode_new_tokens,
            decode_warmups,
            decode_repeats,
        )
        prefill_median_ms = payload["prefill"]["128"]["median_ms"]
        generate_median_ms = payload["generate"]["median_ms"]
        approximate_decode_ms = max(generate_median_ms - prefill_median_ms, 1e-9)
        payload["generate"]["approximate_decode_ms_after_subtracting_prefill"] = approximate_decode_ms
        payload["generate"]["approximate_decode_tokens_per_s"] = (
            decode_new_tokens / (approximate_decode_ms / 1000.0)
        )

        quality, logits = run_quality(model, tokenizer, cases)
        payload["quality"] = quality
        torch.save({"case_ids": [case["id"] for case in cases], "logits": logits}, logits_path)
        payload["last_logits_path"] = str(logits_path)

        payload["profiler"] = run_profiler(model, decode_ids, suffix)
        del decode_ids
        synchronize()
        payload["runtime_memory"] = {
            "allocated_end_gib": torch.cuda.memory_allocated() / (1024**3),
            "reserved_end_gib": torch.cuda.memory_reserved() / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        }
        payload["status"] = "complete"
        payload["finished_at_unix"] = time.time()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "mode": args.mode, "smoke": args.smoke, "output": str(out_path)}))
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        payload["finished_at_unix"] = time.time()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(payload["traceback"])
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
