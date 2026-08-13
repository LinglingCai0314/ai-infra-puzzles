#!/usr/bin/env python3
"""Reviewed experiment cells for the Chapter 04 GPU foundations labs."""

from __future__ import annotations


ENV_CODE = r'''
from pathlib import Path
from collections import Counter, deque
import json, math, platform, statistics, sys, time

import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "Chapter 04 retained runs require a CUDA-capable GPU."
DEVICE = torch.device("cuda")
SEED = 20260813 + LESSON_NO
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

major, minor = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
ENV = {
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": f"{major}.{minor}",
    "torch": torch.__version__,
    "cuda_runtime": str(torch.version.cuda),
    "python": sys.version.split()[0],
    "seed": SEED,
}
print(json.dumps(ENV, indent=2))

def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

def cuda_samples(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples

def summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "samples_ms": samples,
    }
'''.strip()


EXPERIMENTS: dict[int, str] = {
    1: r'''
C = 80e-15
alpha = 0.18
frequency = 1e9

def transition_energy(capacitance_f, voltage_v):
    return capacitance_f * voltage_v**2

def dynamic_power(capacitance_f, voltage_v, activity, frequency_hz):
    return activity * capacitance_f * voltage_v**2 * frequency_hz

truth_table = {0: 1, 1: 0}
e_1v = transition_energy(C, 1.0)
e_08v = transition_energy(C, 0.8)
voltage_sweep = {
    str(v): dynamic_power(C, v, alpha, frequency) * 1e3
    for v in (0.6, 0.7, 0.8, 0.9, 1.0)
}
activity_sweep = {
    str(a): dynamic_power(C, 1.0, a, frequency) * 1e3
    for a in (0.05, 0.1, 0.18, 0.3, 0.5)
}
metrics = {
    "truth_table": truth_table,
    "capacitance_f": C,
    "activity": alpha,
    "frequency_hz": frequency,
    "energy_1v_fj": e_1v * 1e15,
    "energy_08v_fj": e_08v * 1e15,
    "voltage_energy_ratio": e_1v / e_08v,
    "power_1v_mw": dynamic_power(C, 1.0, alpha, frequency) * 1e3,
    "voltage_sweep_mw": voltage_sweep,
    "activity_sweep_mw": activity_sweep,
}
analysis = (
    f"At fixed C, activity, and frequency, lowering voltage from 1.0 V to 0.8 V "
    f"reduced modeled transition energy from {metrics['energy_1v_fj']:.1f} to "
    f"{metrics['energy_08v_fj']:.1f} fJ, a {metrics['voltage_energy_ratio']:.3f}x ratio. "
    "This is a sensitivity model, not board-power telemetry."
)
print(json.dumps(metrics, indent=2))
''',
    2: r'''
VDD = 1.0
VPRE = VDD / 2
C_CELL = 30e-15

def shared_voltage(v_cell, c_cell, v_bit, c_bit):
    return (v_cell * c_cell + v_bit * c_bit) / (c_cell + c_bit)

def margin_mv(v_cell, bitline_ratio):
    shared = shared_voltage(v_cell, C_CELL, VPRE, C_CELL * bitline_ratio)
    return abs(shared - VPRE) * 1e3

fresh = margin_mv(1.0, 10)
leaked = margin_mv(0.72, 10)
ratios = {str(r): margin_mv(1.0, r) for r in (5, 10, 20, 40, 80)}
retention = {str(v): margin_mv(v, 10) for v in (1.0, 0.9, 0.8, 0.72, 0.6)}
metrics = {
    "vdd_v": VDD,
    "precharge_v": VPRE,
    "cell_capacitance_f": C_CELL,
    "fresh_margin_mv": fresh,
    "leaked_margin_mv": leaked,
    "margin_retained": leaked / fresh,
    "restore_target_v": VDD,
    "bitline_ratio_sweep_margin_mv": ratios,
    "retention_sweep_margin_mv": retention,
}
analysis = (
    f"With a 10:1 bitline/cell capacitance ratio, the ideal fresh-cell deviation was "
    f"{fresh:.3f} mV and fell to {leaked:.3f} mV when retained cell voltage was 0.72 V. "
    "The sense amplifier and restore step are therefore part of the read contract."
)
print(json.dumps(metrics, indent=2))
''',
    3: r'''
hierarchy = [
    {"level": "register", "scope": "thread", "technology": "flip-flop/register file", "latency_ratio": 1},
    {"level": "shared/L1", "scope": "block/SM", "technology": "SRAM", "latency_ratio": 5},
    {"level": "L2", "scope": "all SMs", "technology": "SRAM", "latency_ratio": 25},
    {"level": "external memory", "scope": "device", "technology": "GDDR7 on this GPU", "latency_ratio": 120},
]
working_set = 64 * 2**20
reuse_counts = (1, 2, 4, 8, 16, 32)
bytes_per_use = {str(r): working_set / r for r in reuse_counts}
device_gib = props.total_memory / 2**30
metrics = {
    "hierarchy": hierarchy,
    "working_set_bytes": working_set,
    "device_memory_gib": device_gib,
    "bytes_per_use": bytes_per_use,
    "bytes_per_use_streaming": int(bytes_per_use["1"]),
    "bytes_per_use_reuse32": int(bytes_per_use["32"]),
    "reuse32_reduction": bytes_per_use["1"] / bytes_per_use["32"],
}
analysis = (
    f"A {working_set / 2**20:.0f} MiB working set costs {working_set:,} external bytes per use "
    f"when streamed once, but {bytes_per_use['32']:,.0f} bytes per logical use when one load is "
    "amortized across 32 on-chip uses. Latency ratios are illustrative."
)
print(json.dumps(metrics, indent=2))
''',
    4: r'''
dtype = torch.bfloat16
n = 2**26
a = torch.randn(n, device=DEVICE, dtype=dtype)
b = torch.randn(n, device=DEVICE, dtype=dtype)
c = torch.empty_like(a)
vector_samples = cuda_samples(lambda: torch.add(a, b, out=c), repeats=25)
vector_median = statistics.median(vector_samples)
vector_bytes = 3 * n * a.element_size()
vector_gbps = vector_bytes / (vector_median / 1e3) / 1e9
vector_intensity = n / vector_bytes

m = 2048
x = torch.randn((m, m), device=DEVICE, dtype=dtype)
y = torch.randn((m, m), device=DEVICE, dtype=dtype)
z = torch.empty((m, m), device=DEVICE, dtype=dtype)
gemm_samples = cuda_samples(lambda: torch.mm(x, y, out=z), repeats=20)
gemm_median = statistics.median(gemm_samples)
gemm_flops = 2 * m**3
gemm_tflops = gemm_flops / (gemm_median / 1e3) / 1e12
gemm_bytes = (x.numel() + y.numel() + z.numel()) * x.element_size()
metrics = {
    "vector_median_ms": vector_median,
    "vector_effective_gbps": vector_gbps,
    "vector_intensity_flop_byte": vector_intensity,
    "gemm_median_ms": gemm_median,
    "gemm_tflops": gemm_tflops,
    "gemm_intensity_flop_byte": gemm_flops / gemm_bytes,
    "vector_samples_ms": vector_samples,
    "gemm_samples_ms": gemm_samples,
}
analysis = (
    f"Vector addition delivered {vector_gbps:.1f} requested GB/s at only "
    f"{vector_intensity:.4f} FLOP/byte; the BF16 GEMM delivered {gemm_tflops:.1f} TFLOP/s "
    f"with an algorithmic intensity of {metrics['gemm_intensity_flop_byte']:.1f} FLOP/byte."
)
print(json.dumps(metrics, indent=2))
''',
    5: r'''
dtype = torch.bfloat16

def make_case(m, n, k):
    a = torch.randn((m, k), device=DEVICE, dtype=dtype)
    b = torch.randn((k, n), device=DEVICE, dtype=dtype)
    out = torch.empty((m, n), device=DEVICE, dtype=dtype)
    samples = cuda_samples(lambda: torch.mm(a, b, out=out), repeats=20)
    median = statistics.median(samples)
    return {"shape": [m, n, k], "median_ms": median,
            "tflops": (2 * m * n * k) / (median / 1e3) / 1e12,
            "samples_ms": samples, "checksum": float(out.float().mean().item())}

aligned = make_case(2048, 2048, 2048)
awkward = make_case(2039, 2053, 2041)
metrics = {
    "compute_capability": ENV["compute_capability"],
    "aligned": aligned,
    "awkward": awkward,
    "aligned_median_ms": aligned["median_ms"],
    "awkward_median_ms": awkward["median_ms"],
    "aligned_tflops": aligned["tflops"],
    "awkward_tflops": awkward["tflops"],
    "throughput_ratio": aligned["tflops"] / awkward["tflops"],
}
analysis = (
    f"The aligned and awkward BF16 shapes reached {aligned['tflops']:.1f} and "
    f"{awkward['tflops']:.1f} TFLOP/s. This establishes a library-shape effect on this stack; "
    "it does not identify internal instructions without a profiler trace."
)
print(json.dumps(metrics, indent=2))
''',
    6: r'''
dtype = torch.bfloat16
B, H, N, D = 1, 16, 1024, 64
q = torch.randn((B, H, N, D), device=DEVICE, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)
scale = 1 / math.sqrt(D)

def eager_attention():
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(dtype)
    return torch.matmul(probs, v)

def sdpa_attention():
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

eager_samples = cuda_samples(eager_attention, warmup=3, repeats=15)
sdpa_samples = cuda_samples(sdpa_attention, warmup=3, repeats=15)

def peak_delta(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / 2**20

eager_out, eager_peak = peak_delta(eager_attention)
sdpa_out, sdpa_peak = peak_delta(sdpa_attention)
max_error = float((eager_out.float() - sdpa_out.float()).abs().max().item())
score_mib = B * H * N * N * q.element_size() / 2**20
metrics = {
    "shape": [B, H, N, D], "dtype": str(dtype),
    "score_tensor_mib": score_mib,
    "eager_median_ms": statistics.median(eager_samples),
    "sdpa_median_ms": statistics.median(sdpa_samples),
    "eager_peak_mib": eager_peak,
    "sdpa_peak_mib": sdpa_peak,
    "max_abs_error": max_error,
    "eager_samples_ms": eager_samples, "sdpa_samples_ms": sdpa_samples,
}
analysis = (
    f"The explicit score tensor is {score_mib:.1f} MiB. Eager and SDPA medians were "
    f"{metrics['eager_median_ms']:.3f} and {metrics['sdpa_median_ms']:.3f} ms, with "
    f"{max_error:.6f} maximum absolute output difference. Backend identity is not inferred."
)
print(json.dumps(metrics, indent=2))
''',
    7: r'''
interface_bits = 512
pin_rate_gbps = 28.0
theoretical_gbps = interface_bits * pin_rate_gbps / 8
n = 2**26
src = torch.randn(n, device=DEVICE, dtype=torch.float32)
dst = torch.empty_like(src)
samples = cuda_samples(lambda: dst.copy_(src), repeats=25)
median = statistics.median(samples)
requested_bytes = 2 * src.numel() * src.element_size()
effective = requested_bytes / (median / 1e3) / 1e9
metrics = {
    "memory_technology": "GDDR7",
    "interface_bits": interface_bits,
    "pin_rate_gbps": pin_rate_gbps,
    "theoretical_gbps": theoretical_gbps,
    "tensor_mib": src.numel() * src.element_size() / 2**20,
    "copy_median_ms": median,
    "effective_copy_gbps": effective,
    "achieved_fraction": effective / theoretical_gbps,
    "samples_ms": samples,
    "checksum": float(dst[:4096].sum().item()),
}
analysis = (
    f"A 512-bit interface at 28 Gb/s per pin yields {theoretical_gbps:.0f} GB/s. The "
    f"device-copy probe reported {effective:.1f} requested GB/s ({effective/theoretical_gbps:.1%} "
    "of that interface number) on the GDDR7 RTX 5090."
)
print(json.dumps(metrics, indent=2))
''',
    8: r'''
n = 2**26
x = torch.randn(n, device=DEVICE, dtype=torch.float32)
rows = {}
for stride in (1, 2, 4, 8, 16, 32):
    view = x[::stride]
    samples = cuda_samples(lambda view=view: (view * 1.0001).sum(), warmup=3, repeats=15)
    median = statistics.median(samples)
    useful_bytes = view.numel() * view.element_size()
    rows[f"stride_{stride}"] = {
        "elements": view.numel(), "median_ms": median,
        "requested_gbps": useful_bytes / (median / 1e3) / 1e9,
        "samples_ms": samples,
    }

line_bytes, sets = 128, 4096
address = 0x1234ABCD
offset_bits = int(math.log2(line_bytes)); set_bits = int(math.log2(sets))
address_example = {
    "address_hex": hex(address),
    "offset": address & (line_bytes - 1),
    "set": (address >> offset_bits) & (sets - 1),
    "tag": address >> (offset_bits + set_bits),
    "assumed_line_bytes": line_bytes, "assumed_sets": sets,
}
metrics = {**rows, "address_example": address_example}
analysis = (
    f"Requested bandwidth fell from {rows['stride_1']['requested_gbps']:.1f} GB/s at stride 1 "
    f"to {rows['stride_32']['requested_gbps']:.1f} GB/s at stride 32. This is a locality probe; "
    "no L2 hit-rate counter was collected."
)
print(json.dumps(metrics, indent=2))
''',
    9: r'''
def simulate(hotspot, arrival_ticks=160, sources=4, outputs=4, service_per_output=1):
    queues = [deque() for _ in range(outputs)]
    latencies = []
    max_queue = 0
    queue_area = 0
    delivered = 0
    tick = 0
    while tick < arrival_ticks or any(queues):
        if tick < arrival_ticks:
            for source in range(sources):
                destination = 0 if hotspot else source % outputs
                queues[destination].append((source, tick))
        for output in range(outputs):
            for _ in range(service_per_output):
                if queues[output]:
                    _, created = queues[output].popleft()
                    latencies.append(tick - created + 1)
                    delivered += 1
        total_queued = sum(len(q) for q in queues)
        queue_area += total_queued
        max_queue = max(max_queue, max(len(q) for q in queues))
        tick += 1
    return {
        "delivered_flits": delivered, "drain_ticks": tick,
        "mean_latency_ticks": statistics.mean(latencies),
        "p95_latency_ticks": percentile(latencies, 0.95),
        "mean_queue": queue_area / tick, "max_queue": max_queue,
    }

balanced = simulate(False)
hotspot = simulate(True)
metrics = {
    "balanced": balanced, "hotspot": hotspot,
    "hotspot_latency_ratio": hotspot["mean_latency_ticks"] / balanced["mean_latency_ticks"],
}
analysis = (
    f"Balanced and hotspot traffic delivered the same {balanced['delivered_flits']} flits, "
    f"but mean latency changed from {balanced['mean_latency_ticks']:.2f} to "
    f"{hotspot['mean_latency_ticks']:.2f} ticks as one output became oversubscribed."
)
print(json.dumps(metrics, indent=2))
''',
    10: r'''
limits = {
    "max_threads_sm": 2048, "max_warps_sm": 64, "max_blocks_sm": 16,
    "registers_sm": 65536, "shared_bytes_sm": 64 * 1024,
}
cases = {
    "balanced": {"threads": 256, "registers_thread": 32, "shared_bytes": 8 * 1024},
    "register_heavy": {"threads": 256, "registers_thread": 112, "shared_bytes": 8 * 1024},
    "shared_heavy": {"threads": 256, "registers_thread": 32, "shared_bytes": 40 * 1024},
}

def occupancy(case):
    warps = math.ceil(case["threads"] / 32)
    block_limits = {
        "threads": limits["max_threads_sm"] // case["threads"],
        "warps": limits["max_warps_sm"] // warps,
        "registers": limits["registers_sm"] // (case["threads"] * case["registers_thread"]),
        "shared": limits["shared_bytes_sm"] // case["shared_bytes"],
        "blocks": limits["max_blocks_sm"],
    }
    resident_blocks = min(block_limits.values())
    return {"resident_blocks": resident_blocks, "resident_warps": resident_blocks * warps,
            "occupancy": resident_blocks * warps / limits["max_warps_sm"],
            "limiting_resources": [k for k, v in block_limits.items() if v == resident_blocks],
            "block_limits": block_limits}

occ = {name: occupancy(case) for name, case in cases.items()}
bank_multiplicity = {}
for stride in (1, 2, 4, 8, 16, 32):
    counts = Counter((lane * stride) % 32 for lane in range(32))
    bank_multiplicity[f"stride_{stride}"] = max(counts.values())
metrics = {
    "model_limits": limits, "device_name": ENV["gpu"],
    "occupancy": {name: value["occupancy"] for name, value in occ.items()},
    "occupancy_details": occ, "bank_multiplicity": bank_multiplicity,
}
analysis = (
    f"Modeled occupancy was {occ['balanced']['occupancy']:.1%}, "
    f"{occ['register_heavy']['occupancy']:.1%}, and {occ['shared_heavy']['occupancy']:.1%}; "
    f"stride 32 mapped all lanes to one illustrative bank (multiplicity "
    f"{bank_multiplicity['stride_32']})."
)
print(json.dumps(metrics, indent=2))
''',
    11: r'''
updates = 2**22
bins = 2**18
values = torch.ones(updates, device=DEVICE, dtype=torch.float32)
dispersed_idx = torch.arange(updates, device=DEVICE, dtype=torch.int64) % bins
hotspot_bins = 64
hotspot_idx = torch.arange(updates, device=DEVICE, dtype=torch.int64) % hotspot_bins
destination = torch.zeros(bins, device=DEVICE, dtype=torch.float32)

def run(indices):
    destination.zero_()
    destination.scatter_add_(0, indices, values)

dispersed_samples = cuda_samples(lambda: run(dispersed_idx), repeats=20)
run(dispersed_idx); dispersed_checksum = float(destination.sum().item())
hotspot_samples = cuda_samples(lambda: run(hotspot_idx), repeats=20)
run(hotspot_idx); hotspot_checksum = float(destination.sum().item())
dispersed_median = statistics.median(dispersed_samples)
hotspot_median = statistics.median(hotspot_samples)
metrics = {
    "updates": updates, "bins": bins, "hotspot_bins": hotspot_bins,
    "dispersed_median_ms": dispersed_median, "hotspot_median_ms": hotspot_median,
    "hotspot_slowdown": hotspot_median / dispersed_median,
    "dispersed_collision_ratio": 1 - bins / updates,
    "hotspot_collision_ratio": 1 - hotspot_bins / updates,
    "dispersed_checksum": dispersed_checksum, "hotspot_checksum": hotspot_checksum,
    "dispersed_samples_ms": dispersed_samples, "hotspot_samples_ms": hotspot_samples,
}
analysis = (
    f"Concentrating {updates:,} updates into {hotspot_bins} bins changed median scatter-add "
    f"latency by {metrics['hotspot_slowdown']:.3f}x versus spreading them across {bins:,} bins. "
    "Both routes preserved the total update checksum."
)
print(json.dumps(metrics, indent=2))
''',
    12: r'''
problem_size = 1000
block_size = 256
warp_size = 32
grid_blocks = math.ceil(problem_size / block_size)
tail_active = problem_size - (grid_blocks - 1) * block_size

def two_path_efficiency(mask):
    true_count = sum(mask); false_count = len(mask) - true_count
    issued_paths = int(true_count > 0) + int(false_count > 0)
    return len(mask) / (len(mask) * issued_paths)

patterns = {
    "uniform": [True] * warp_size,
    "half_warp": [lane < 16 for lane in range(warp_size)],
    "alternating": [lane % 2 == 0 for lane in range(warp_size)],
}
pattern_efficiency = {name: two_path_efficiency(mask) for name, mask in patterns.items()}
metrics = {
    "problem_size": problem_size, "block_size": block_size, "warp_size": warp_size,
    "grid_blocks": grid_blocks, "tail_active_threads": tail_active,
    "patterns": pattern_efficiency,
    "active_masks_hex": {
        name: hex(sum((1 << lane) for lane, active in enumerate(mask) if active))
        for name, mask in patterns.items()
    },
}
analysis = (
    f"The launch needs {grid_blocks} blocks and the final block has {tail_active} active threads. "
    f"Under the explicit equal-cost two-path model, uniform efficiency is "
    f"{pattern_efficiency['uniform']:.1%} and both mixed patterns are "
    f"{pattern_efficiency['half_warp']:.1%}; compiler behavior is not modeled."
)
print(json.dumps(metrics, indent=2))
''',
    13: r'''
n = 8192
base = torch.randn((n, n), device=DEVICE, dtype=torch.float32)
transposed = base.t()
out_contiguous = torch.empty_like(base)
out_transposed = torch.empty_like(base)
contiguous_samples = cuda_samples(lambda: out_contiguous.copy_(base), repeats=20)
transposed_samples = cuda_samples(lambda: out_transposed.copy_(transposed), repeats=20)
contiguous_median = statistics.median(contiguous_samples)
transposed_median = statistics.median(transposed_samples)
requested_bytes = 2 * base.numel() * base.element_size()
contiguous_gbps = requested_bytes / (contiguous_median / 1e3) / 1e9
transposed_gbps = requested_bytes / (transposed_median / 1e3) / 1e9
equivalent = bool(torch.allclose(out_transposed[:64, :64], base.t()[:64, :64]))
metrics = {
    "shape": list(base.shape), "base_stride": list(base.stride()),
    "transposed_stride": list(transposed.stride()),
    "contiguous_median_ms": contiguous_median,
    "transposed_median_ms": transposed_median,
    "contiguous_gbps": contiguous_gbps, "transposed_gbps": transposed_gbps,
    "transposed_slowdown": transposed_median / contiguous_median,
    "output_equivalent": equivalent,
    "contiguous_samples_ms": contiguous_samples, "transposed_samples_ms": transposed_samples,
}
analysis = (
    f"The contiguous and transposed views requested the same logical bytes, but their strides "
    f"were {base.stride()} and {transposed.stride()}; copy latency changed by "
    f"{metrics['transposed_slowdown']:.3f}x. Physical transaction counters were not collected."
)
print(json.dumps(metrics, indent=2))
''',
    14: r'''
n = 2**24
values = torch.randn(n, device=DEVICE, dtype=torch.float32)
one_idx = torch.zeros(n, device=DEVICE, dtype=torch.int64)
many_bins = 2**16
many_idx = torch.arange(n, device=DEVICE, dtype=torch.int64) % many_bins
one_out = torch.zeros(1, device=DEVICE, dtype=torch.float32)
many_out = torch.zeros(many_bins, device=DEVICE, dtype=torch.float32)

def one_bin():
    one_out.zero_(); one_out.scatter_add_(0, one_idx, values)

def many_bin():
    many_out.zero_(); many_out.scatter_add_(0, many_idx, values)

sum_samples = cuda_samples(lambda: values.sum(), repeats=20)
one_samples = cuda_samples(one_bin, repeats=20)
many_samples = cuda_samples(many_bin, repeats=20)
reference = float(values.sum().item())
one_bin(); many_bin()
one_value = float(one_out.item()); many_value = float(many_out.sum().item())
sum_median = statistics.median(sum_samples)
one_median = statistics.median(one_samples)
many_median = statistics.median(many_samples)
metrics = {
    "sum_median_ms": sum_median, "one_bin_median_ms": one_median,
    "many_bin_median_ms": many_median, "one_bin_slowdown": one_median / sum_median,
    "many_bin_slowdown": many_median / sum_median,
    "checksum_error": max(abs(reference - one_value), abs(reference - many_value)),
    "reference_sum": reference, "one_bin_sum": one_value, "many_bin_sum": many_value,
    "sum_samples_ms": sum_samples, "one_bin_samples_ms": one_samples,
    "many_bin_samples_ms": many_samples,
}
analysis = (
    f"Library sum, one-bin scatter, and many-bin scatter medians were {sum_median:.3f}, "
    f"{one_median:.3f}, and {many_median:.3f} ms. The routes are a hierarchy/contention "
    "probe and may accumulate in different floating-point orders."
)
print(json.dumps(metrics, indent=2))
''',
    15: r'''
n = 4096
a = torch.randn((n, n), device=DEVICE, dtype=torch.bfloat16)
b = torch.randn((n, n), device=DEVICE, dtype=torch.bfloat16)
out = torch.empty((n, n), device=DEVICE, dtype=torch.bfloat16)
for _ in range(5):
    torch.mm(a, b, out=out)
torch.cuda.synchronize()

host_samples_us = []
for _ in range(20):
    tick = time.perf_counter()
    torch.mm(a, b, out=out)
    host_samples_us.append((time.perf_counter() - tick) * 1e6)
torch.cuda.synchronize()
event_samples = cuda_samples(lambda: torch.mm(a, b, out=out), warmup=0, repeats=20)
host_median = statistics.median(host_samples_us)
event_median = statistics.median(event_samples)
flops = 2 * n**3
metrics = {
    "shape": [n, n, n], "dtype": "bfloat16",
    "host_enqueue_us": host_median, "event_median_ms": event_median,
    "timing_illusion_ratio": (event_median * 1000) / host_median,
    "library_tflops": flops / (event_median / 1e3) / 1e12,
    "host_samples_us": host_samples_us, "event_samples_ms": event_samples,
    "checksum": float(out[:64, :64].float().mean().item()),
}
analysis = (
    f"Unsynchronized host enqueue took {host_median:.2f} µs while CUDA events measured "
    f"{event_median:.3f} ms of device work, a {metrics['timing_illusion_ratio']:.1f}x unit-normalized "
    "gap. Host end-to-end timing remains valid when synchronized."
)
print(json.dumps(metrics, indent=2))
''',
    16: r'''
chapter = Path.cwd().parent
required_top = ("lesson", "title", "environment", "evidence_label", "metrics", "analysis", "conclusion")
layer_map = {
    "hardware-model": set(range(1, 4)) | {7, 9, 10},
    "kernel-measurement": {4, 5, 6, 8, 11, 13, 14, 15},
    "systems-decision": {3, 4, 6, 7, 11, 15},
}
records = []
for lesson in range(1, 16):
    matches = sorted(chapter.glob(f"{lesson:02d}-*/artifacts/rtx5090-result.json"))
    if not matches:
        records.append({"lesson": lesson, "found": False, "complete": False,
                        "missing": list(required_top), "evidence_label": None})
        continue
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    missing = [field for field in required_top if data.get(field) in (None, "", {})]
    env_missing = [field for field in ("gpu", "compute_capability", "torch", "cuda_runtime")
                   if not data.get("environment", {}).get(field)]
    records.append({"lesson": lesson, "found": True, "complete": not missing and not env_missing,
                    "missing": missing, "environment_missing": env_missing,
                    "evidence_label": data.get("evidence_label")})

found = sum(r["found"] for r in records)
complete = sum(r["complete"] for r in records)
labels = sorted({r["evidence_label"] for r in records if r["evidence_label"]})
represented_layers = sorted(name for name, lessons in layer_map.items()
                            if any(r["complete"] and r["lesson"] in lessons for r in records))
metrics = {
    "artifacts_expected": 15, "artifacts_found": found,
    "complete_artifacts": complete, "completion_rate": complete / 15,
    "evidence_labels": labels, "evidence_labels_represented": len(labels),
    "layers": represented_layers, "layers_represented": len(represented_layers),
    "records": records,
}
analysis = (
    f"The portfolio audit found {found}/15 artifacts and {complete}/15 complete records, "
    f"covering {len(labels)} evidence labels and {len(represented_layers)} project layers. "
    "Schema completeness is necessary but does not validate experimental causality."
)
print(json.dumps(metrics, indent=2))
''',
    17: r'''
official = {
    "source": "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
    "memory": {"value": 32.0, "unit": "GiB", "technology": "GDDR7"},
    "interface": {"value": 512.0, "unit": "bit"},
    "pin_rate": {"value": 28.0, "unit": "Gb/s per pin", "derived_from_bandwidth": True},
    "bandwidth": {"value": 1792.0, "unit": "GB/s"},
    "compute_capability": {"value": "12.0", "unit": "major.minor"},
}
calculated_bandwidth = official["interface"]["value"] * official["pin_rate"]["value"] / 8
bandwidth_error = abs(calculated_bandwidth - official["bandwidth"]["value"]) / official["bandwidth"]["value"]
device_gib = props.total_memory / 2**30

lesson07_path = chapter = Path.cwd().parent
lesson07_matches = sorted(chapter.glob("07-*/artifacts/rtx5090-result.json"))
lesson07 = json.loads(lesson07_matches[0].read_text(encoding="utf-8")) if lesson07_matches else None
lesson07_fraction = (lesson07 or {}).get("metrics", {}).get("achieved_fraction")

illustrative_compute_tflops = 100.0
intensities = (0.25, 1, 4, 16, 64, 256)
roofline = {
    str(ai): min(illustrative_compute_tflops, official["bandwidth"]["value"] * ai / 1000)
    for ai in intensities
}
fields_with_units = sum(
    1 for key, value in official.items()
    if isinstance(value, dict) and value.get("unit")
)
assert official["memory"]["technology"] == "GDDR7"
assert bandwidth_error < 1e-12
metrics = {
    "official_snapshot": official,
    "device_memory_gib": device_gib,
    "official_memory_gib": official["memory"]["value"],
    "capacity_difference_gib": device_gib - official["memory"]["value"],
    "calculated_bandwidth_gbps": calculated_bandwidth,
    "bandwidth_formula_error": bandwidth_error,
    "lesson07_achieved_fraction": lesson07_fraction,
    "fields_with_units": fields_with_units,
    "illustrative_compute_roof_tflops": illustrative_compute_tflops,
    "roofline_tflops_by_intensity": roofline,
}
fraction_text = "not available" if lesson07_fraction is None else f"{lesson07_fraction:.1%}"
analysis = (
    f"The width/rate formula reproduced {calculated_bandwidth:.0f} GB/s with zero arithmetic "
    f"error; the device reported {device_gib:.2f} GiB and Lesson 07 achieved {fraction_text} of "
    "the interface figure. The compute roof in the table is explicitly illustrative."
)
print(json.dumps(metrics, indent=2))
''',
}
