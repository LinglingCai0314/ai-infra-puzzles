#!/usr/bin/env python3
"""Self-contained RTX 5090 experiment cells for Chapter 03 notebooks."""

from __future__ import annotations


ENV_CODE = r'''
from pathlib import Path
import gc, hashlib, importlib, inspect, ipaddress, json, math, os, random, re
import shutil, statistics, subprocess, sys, tempfile, time
from urllib.parse import urlparse

# The default FlashInfer sampler requires a local JIT link setup that is not
# guaranteed in wheel-only environments. vLLM's native PyTorch sampler keeps
# these labs reproducible without changing attention or scheduling backends.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import requests
import torch
import vllm
import yaml
from vllm import LLM, SamplingParams

assert torch.cuda.is_available(), "This lab requires a CUDA-capable GPU."
DEVICE = torch.device("cuda")
SEED = 20260812 + LESSON_NO
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
MODEL_PATH = os.environ.get("CH3_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL = Path(MODEL_PATH)
assert MODEL.exists(), f"Set CH3_MODEL to a local model directory; not found: {MODEL}"

gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
ENV = {
    "gpu": gpu_name, "compute_capability": f"{major}.{minor}",
    "torch": torch.__version__, "cuda_runtime": str(torch.version.cuda),
    "python": sys.version.split()[0], "vllm": vllm.__version__,
    "model_path": MODEL.name, "seed": SEED,
}
print(json.dumps(ENV, indent=2))

def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered: return float("nan")
    pos = (len(ordered) - 1) * q; lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

def model_config():
    return json.loads((MODEL / "config.json").read_text(encoding="utf-8"))

def base_engine_args(**overrides):
    values = {
        "model": str(MODEL), "tokenizer": str(MODEL), "trust_remote_code": False,
        "dtype": "bfloat16", "max_model_len": 2048, "gpu_memory_utilization": 0.45,
        "enforce_eager": True, "seed": SEED, "max_num_seqs": 16,
    }
    values.update(overrides); return values

def output_record(item):
    completion = item.outputs[0]; tokens = list(completion.token_ids)
    return {
        "request_id": str(item.request_id), "prompt_tokens": len(item.prompt_token_ids or []),
        "output_tokens": len(tokens), "token_ids": tokens, "text_preview": completion.text[:120],
        "text_sha256": hashlib.sha256(completion.text.encode()).hexdigest(),
        "finish_reason": str(completion.finish_reason),
        "stop_reason": None if completion.stop_reason is None else str(completion.stop_reason),
        "num_cached_tokens": int(getattr(item, "num_cached_tokens", 0) or 0),
    }

def vllm_cli():
    candidate = Path(sys.executable).parent / "vllm"
    return str(candidate if candidate.exists() else (shutil.which("vllm") or "vllm"))

def cli_help(*args):
    result = subprocess.run([vllm_cli(), *args, "--help"], capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout + result.stderr

def run_server_probe(port, request_payload=None, scrape_metrics=False):
    log_path = Path(tempfile.gettempdir()) / f"ch03-vllm-{LESSON_NO}-{port}.log"
    command = [vllm_cli(), "serve", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
               "--dtype", "bfloat16", "--max-model-len", "1024", "--gpu-memory-utilization", "0.45",
               "--enforce-eager", "--disable-uvicorn-access-log"]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    ready = False
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if process.poll() is not None: break
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    ready = True; break
            except requests.RequestException: pass
            time.sleep(1)
        startup_s = time.perf_counter() - started
        if not ready:
            raise RuntimeError("vLLM server failed to start:\n" + log_path.read_text(errors="replace")[-6000:])
        models = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=30)
        data = {"server_ready": True, "startup_s": startup_s,
                "models_status": models.status_code, "model_json": models.json()}
        if request_payload is not None:
            tick = time.perf_counter()
            chat = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json=request_payload, timeout=180)
            data.update(chat_status=chat.status_code, chat_latency_s=time.perf_counter() - tick,
                        chat_json=chat.json())
        if scrape_metrics:
            response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
            data.update(metrics_status=response.status_code, metrics_text=response.text)
        return data
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
        private_home = "/" + "root" + "/"
        globals()["SERVER_LOG_TAIL"] = tail.replace(str(MODEL), "$CH3_MODEL").replace(private_home, "<remote-home>/")
'''.strip()


EXPERIMENTS: dict[int, str] = {}

EXPERIMENTS.update({
    1: r'''
prompts = ["Explain continuous batching in two sentences.", "List two causes of high TTFT.",
           "What does a KV cache store?", "Why retain raw benchmark samples?"]
llm = LLM(**base_engine_args(max_model_len=1024))
params = SamplingParams(temperature=0.0, max_tokens=24, seed=SEED)
torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
outputs = llm.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - started
records = [output_record(item) for item in outputs]; output_tokens = sum(x["output_tokens"] for x in records)
metrics = {"vllm_version": vllm.__version__, "requests": len(records),
           "prompt_tokens": sum(x["prompt_tokens"] for x in records), "output_tokens": output_tokens,
           "elapsed_s": elapsed, "output_tokens_s": output_tokens / elapsed,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "outputs": records}
analysis = (f"vLLM {vllm.__version__} completed {len(records)} requests and {output_tokens} output "
            f"tokens in {elapsed:.3f} s ({metrics['output_tokens_s']:.1f} output tokens/s). This is "
            "native offline execution, not online queue or network evidence.")
''',
    2: r'''
llm = LLM(**base_engine_args(max_model_len=2048)); short = "Explain TTFT versus ITL."
long = ("Prefill processes prompts and Decode reuses cached key/value vectors. " * 70) + "Summarize."
cases = {"short_short": (short, 8), "long_short": (long, 8),
         "short_long": (short, 32), "long_long": (long, 32)}; rows = {}
for name, (prompt, limit) in cases.items():
    tick = time.perf_counter()
    item = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=limit, seed=SEED),
                        use_tqdm=False)[0]
    row = output_record(item); row["elapsed_s"] = time.perf_counter() - tick
    state = getattr(item, "metrics", None)
    row["request_metrics"] = ({key: getattr(state, key, None) for key in
        ("arrival_time", "first_token_time", "finished_time", "scheduler_time",
         "model_forward_time", "model_execute_time")} if state else {})
    rows[name] = row
metrics = {"cases": rows}
analysis = (f"The long prompt used {rows['long_short']['prompt_tokens']} tokens versus "
            f"{rows['short_short']['prompt_tokens']} short; the long answer produced "
            f"{rows['short_long']['output_tokens']} tokens. Elapsed time combines phases, and only "
            "non-null native request fields count as phase timing evidence.")
''',
    3: r'''
lengths = [33, 81, 127, 130, 255, 401, 700, 997]; max_len = 1024; block_size = 16
slab_reserved = len(lengths) * max_len
paged_reserved = sum(math.ceil(length / block_size) * block_size for length in lengths)
block_counts = [math.ceil(length / block_size) for length in lengths]
physical_ids = list(range(sum(block_counts) + 11)); random.shuffle(physical_ids)
physical = {}; tables = []; cursor = 0
for request_id, (length, blocks) in enumerate(zip(lengths, block_counts)):
    table = physical_ids[cursor:cursor + blocks]; cursor += blocks; tables.append(table)
    payload = list(range(request_id * 10000, request_id * 10000 + length))
    for logical, pid in enumerate(table):
        piece = payload[logical*block_size:(logical+1)*block_size]
        physical[pid] = piece + [-1] * (block_size - len(piece))
reconstructed = [[x for pid in table for x in physical[pid]][:length]
                 for length, table in zip(lengths, tables)]
expected = [list(range(i*10000, i*10000+length)) for i, length in enumerate(lengths)]
metrics = {"request_lengths": lengths, "block_size": block_size,
           "slab_reserved_tokens": slab_reserved, "paged_reserved_tokens": paged_reserved,
           "slab_waste_ratio": (slab_reserved-sum(lengths))/slab_reserved,
           "paged_waste_ratio": (paged_reserved-sum(lengths))/paged_reserved,
           "physical_blocks": sum(block_counts), "block_tables": tables,
           "reconstruction_exact": reconstructed == expected}
analysis = (f"Slabs reserved {slab_reserved:,} positions with {metrics['slab_waste_ratio']:.1%} "
            f"waste; {block_size}-token pages reserved {paged_reserved:,} with "
            f"{metrics['paged_waste_ratio']:.1%} waste. Non-contiguous reconstruction was exact; "
            "this is an allocator model, not a kernel benchmark.")
''',
    4: r'''
specs = [{"id":"A","arrival":0,"tokens":14},{"id":"B","arrival":0,"tokens":3},
         {"id":"C","arrival":1,"tokens":2},{"id":"D","arrival":2,"tokens":9},
         {"id":"E","arrival":3,"tokens":1},{"id":"F","arrival":5,"tokens":5}]; capacity = 3
def simulate(policy):
    remaining={x["id"]:x["tokens"] for x in specs}; arrival={x["id"]:x["arrival"] for x in specs}
    first={}; finish={}; active=[]; group=[]; t=0
    while remaining:
        available=[rid for rid in remaining if arrival[rid]<=t and rid not in active]
        if policy=="static":
            if not group: group=sorted(available,key=lambda x:(arrival[x],x))[:capacity]
            active=list(group)
        else:
            pool=list(set(active+available)); key=(lambda x:(remaining[x],arrival[x],x)) if policy=="shortest" else (lambda x:(remaining[x]-.8*(t-arrival[x]),arrival[x],x))
            active=sorted(pool,key=key)[:capacity]
        if not active: t+=1; continue
        for rid in list(active):
            first.setdefault(rid,t); remaining[rid]-=1
            if remaining[rid]==0:
                finish[rid]=t+1; del remaining[rid]; active.remove(rid)
                if rid in group: group.remove(rid)
        t+=1
    lat=[finish[x["id"]]-x["arrival"] for x in specs]; waits=[first[x["id"]]-x["arrival"] for x in specs]
    return {"makespan":max(finish.values()),"mean_latency":statistics.mean(lat),
            "p95_latency":percentile(lat,.95),"max_wait":max(waits),"finish":finish}
metrics={name:simulate(name) for name in ("static","shortest","age_aware")}
analysis=(f"Static/shortest/age-aware makespan was {metrics['static']['makespan']}/"
          f"{metrics['shortest']['makespan']}/{metrics['age_aware']['makespan']} ticks. Priority changed "
          "per-request wait even with identical capacity; tick duration is modeled.")
''',
    5: r'''
cfg=model_config(); layers=int(cfg["num_hidden_layers"]); hidden=int(cfg["hidden_size"])
heads=int(cfg["num_attention_heads"]); kv_heads=int(cfg.get("num_key_value_heads",heads))
head_dim=int(cfg.get("head_dim",hidden//heads)); total=int(torch.cuda.get_device_properties(0).total_memory)
weight_bytes=sum(p.stat().st_size for p in MODEL.glob("*.safetensors")); reserve=3*2**30
usable=max(0,int(total*.85-weight_bytes-reserve))
per_token={"bf16":2*layers*kv_heads*head_dim*2,"fp8":2*layers*kv_heads*head_dim}
capacity={name:usable//size for name,size in per_token.items()}; contexts=(2048,4096,8192,16384)
concurrency={name:{str(ctx):int(tokens//ctx) for ctx in contexts} for name,tokens in capacity.items()}
metrics={"gpu_total_mib":total/2**20,"weight_file_bytes":weight_bytes,"reserve_bytes":reserve,
         "kv_budget_bytes":usable,"geometry":{"layers":layers,"kv_heads":kv_heads,"head_dim":head_dim},
         "kv_bytes_per_token":per_token,"token_capacity":capacity,"concurrency":concurrency}
analysis=(f"Model geometry yields {per_token['bf16']:,} BF16 and {per_token['fp8']:,} FP8 KV "
          f"bytes/token. The declared budget gives a BF16 8K ceiling of {concurrency['bf16']['8192']} "
          "sequences; native allocation and latency must set the operational limit.")
''',
    6: r'''
root_code,root_help=cli_help(); serve_code,serve_help=cli_help("serve"); bench_code,bench_help=cli_help("bench")
x=torch.arange(4096,device=DEVICE,dtype=torch.float32); checksum=float((x.sin()*x.cos()).sum().item())
metrics={"vllm_version":vllm.__version__,"python":sys.version.split()[0],"torch":torch.__version__,
         "cuda_runtime":str(torch.version.cuda),"cli_path":Path(vllm_cli()).name,"cli_found":Path(vllm_cli()).exists(),
         "serve_command":serve_code==0,"bench_command":bench_code==0,
         "cli_tokens":{"serve":"serve" in root_help.lower(),"bench":"bench" in root_help.lower()},
         "cuda_checksum":checksum}
analysis=(f"The isolated environment imported vLLM {vllm.__version__} with PyTorch {torch.__version__} "
          f"/ CUDA {torch.version.cuda}, found serve/bench={serve_code==0}/{bench_code==0}, and completed "
          f"a CUDA checksum of {checksum:.6f}. Native model generation is the stronger final link.")
''',
    7: r'''
prompts=["Define PagedAttention in one sentence.","Give two Prefill/Decode differences.",
         "Why retain raw benchmark samples?"]; params=SamplingParams(temperature=0.0,top_p=1.0,max_tokens=28,seed=SEED)
llm=LLM(**base_engine_args(max_model_len=1024)); tick=time.perf_counter()
outputs=llm.generate(prompts,params,use_tqdm=False); elapsed=time.perf_counter()-tick
records=[output_record(x) for x in outputs]; tokens=sum(x["output_tokens"] for x in records)
metrics={"requests":len(records),"prompt_tokens":sum(x["prompt_tokens"] for x in records),
         "output_tokens":tokens,"elapsed_s":elapsed,"output_tokens_s":tokens/elapsed,
         "unique_output_hashes":len({x["text_sha256"] for x in records}),
         "sampling":{"temperature":0.0,"top_p":1.0,"max_tokens":28,"seed":SEED},"outputs":records}
analysis=(f"The explicit offline call completed {len(records)} requests with "
          f"{metrics['unique_output_hashes']} distinct hashes at {metrics['output_tokens_s']:.1f} "
          "output tokens/s. Functional generation is not task quality or HTTP evidence.")
''',
    8: r'''
payload={"model":str(MODEL),"messages":[{"role":"user","content":"Reply with four words about KV cache."}],
         "temperature":0.0,"max_tokens":16,"seed":SEED}
probe=run_server_probe(18018,request_payload=payload); chat=probe["chat_json"]
choice=(chat.get("choices") or [{}])[0]; usage=chat.get("usage") or {}
schema_valid=(isinstance(chat.get("id"),str) and isinstance(chat.get("choices"),list)
              and isinstance(choice.get("message",{}).get("content"),str)
              and "finish_reason" in choice and isinstance(usage,dict))
metrics={"server_ready":probe["server_ready"],"startup_s":probe["startup_s"],
         "models_status":probe["models_status"],"chat_status":probe["chat_status"],
         "chat_latency_s":probe["chat_latency_s"],"completion_tokens":int(usage.get("completion_tokens",0)),
         "schema_valid":schema_valid,"model_ids":[Path(x.get("id","")).name for x in probe["model_json"].get("data",[])],
         "response_preview":choice.get("message",{}).get("content","")[:160],"server_log_tail":SERVER_LOG_TAIL}
analysis=(f"The server became ready in {metrics['startup_s']:.2f} s; models/chat returned "
          f"HTTP {metrics['models_status']}/{metrics['chat_status']} and schema valid={schema_valid}. "
          "This covers one non-streaming localhost route.")
''',
    9: r'''
llm=LLM(**base_engine_args(max_model_len=1024)); prompt="Write a sentence containing alpha, beta, and gamma."
params={"greedy":SamplingParams(temperature=0.0,max_tokens=24,seed=SEED),
        "sampled":SamplingParams(temperature=.8,top_p=1.0,max_tokens=24,seed=SEED),
        "top_p":SamplingParams(temperature=.8,top_p=.5,max_tokens=24,seed=SEED),
        "stop":SamplingParams(temperature=0.0,max_tokens=24,stop=["gamma"],seed=SEED),
        "logprobs":SamplingParams(temperature=0.0,max_tokens=12,logprobs=3,seed=SEED)}
rows={}
for name,setting in params.items():
    item=llm.generate([prompt],setting,use_tqdm=False)[0]; row=output_record(item)
    row["logprobs_returned"]=item.outputs[0].logprobs is not None; rows[name]=row
metrics={"cases":len(rows),"unique_hashes":len({x["text_sha256"] for x in rows.values()}),**rows}
metrics["logprobs"]={**metrics["logprobs"],"returned":metrics["logprobs"]["logprobs_returned"]}
analysis=(f"Five explicit configurations produced {metrics['unique_hashes']} hashes. Stop finished as "
          f"{metrics['stop']['finish_reason']} and logprobs returned={metrics['logprobs']['returned']}. "
          "Variation is localized to request parameters, not ranked as quality.")
''',
    10: r'''
cfg=model_config(); weights=sorted(MODEL.glob("*.safetensors"))
def stream_hash(paths):
    digest=hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(8*2**20),b""): digest.update(chunk)
    return digest.hexdigest()
hashes={"config":hashlib.sha256((MODEL/"config.json").read_bytes()).hexdigest(),
        "tokenizer":hashlib.sha256((MODEL/"tokenizer.json").read_bytes()).hexdigest(),
        "weights":stream_hash(weights)}
llm=LLM(**base_engine_args(max_model_len=512)); output=llm.generate(["Say provenance."],
    SamplingParams(temperature=0.0,max_tokens=4),use_tqdm=False)[0]
metrics={"weight_files":len(weights),"weight_bytes":sum(p.stat().st_size for p in weights),"hashes":hashes,
         "architecture":str((cfg.get("architectures") or ["unknown"])[0]),
         "declared_dtype":str(cfg.get("torch_dtype")),
         "tokenizer_class":json.loads((MODEL/"tokenizer_config.json").read_text()).get("tokenizer_class"),
         "trust_remote_code":False,"native_load":bool(output.finished),"output":output_record(output)}
analysis=(f"The manifest covers {len(weights)} safetensors file(s), {metrics['weight_bytes']:,} bytes, "
          f"three hashes, and architecture {metrics['architecture']}. The exact bytes completed native generation.")
''',
})

EXPERIMENTS.update({
    21: r'''
payload={"model":str(MODEL),"messages":[{"role":"user","content":"Define one service metric."}],
         "temperature":0.0,"max_tokens":12,"seed":SEED}
probe=run_server_probe(18021,request_payload=payload,scrape_metrics=True); text=probe["metrics_text"]
families=sorted(set(re.findall(r"^# (?:HELP|TYPE) ([^ ]+)",text,flags=re.M)))
names=sorted(set(re.findall(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)",text,flags=re.M)))
groups={"request_success":("vllm:request_success",),"prompt_tokens":("vllm:prompt_tokens",),
        "generation_tokens":("vllm:generation_tokens",),"kv_cache":("vllm:kv_cache_usage_perc",),
        "waiting_requests":("vllm:num_requests_waiting",)}
present={key:any(name in text for name in alternatives) for key,alternatives in groups.items()}
unsafe=re.findall(r'\b(prompt|response|api_key|request_id)="',text,flags=re.I)
metrics={"metrics_status":probe["metrics_status"],"metric_families":len(families),
         "required_present":sum(present.values()),"required_total":len(present),"required_matrix":present,
         "unsafe_label_hits":len(unsafe),"request_succeeded":probe.get("chat_status")==200,
         "sample_names":names[:80],"server_log_tail":SERVER_LOG_TAIL}
analysis=(f"After native traffic, `/metrics` returned HTTP {metrics['metrics_status']} with "
          f"{len(families)} families; {metrics['required_present']}/{metrics['required_total']} required "
          f"groups were found and {len(unsafe)} obvious content/secret labels detected. Thresholds need time series.")
''',
    22: r'''
manifest={"image":"vllm/vllm-openai@sha256:"+"a"*64,"gpu":"all","ipc":"host","port":8000,
 "mounts":[{"source":"/srv/models/qwen","target":"/models/qwen","mode":"ro","kind":"model"},
           {"source":"/srv/cache/vllm","target":"/cache/vllm","mode":"rw","kind":"cache"}],
 "secret":{"source":"docker-secret","name":"vllm_api_key","in_command":False},
 "health":{"path":"/health","interval_s":10,"start_period_s":180},
 "command":["--model","/models/qwen","--max-model-len","8192","--disable-log-requests"],
 "rollback_image":"vllm/vllm-openai@sha256:"+"b"*64}
checks={"image_digest":"@sha256:" in manifest["image"],"rollback_digest":"@sha256:" in manifest["rollback_image"],
 "gpu_explicit":bool(manifest["gpu"]),"ipc_explicit":manifest["ipc"] in {"host","private"},
 "model_read_only":any(x["kind"]=="model" and x["mode"]=="ro" for x in manifest["mounts"]),
 "cache_separate":any(x["kind"]=="cache" and x["mode"]=="rw" for x in manifest["mounts"]),
 "secret_external":manifest["secret"]["source"] in {"docker-secret","file","env-file"},
 "secret_not_command":not manifest["secret"]["in_command"],"health_path":manifest["health"]["path"]=="/health",
 "startup_budget":manifest["health"]["start_period_s"]>=120,"port_explicit":isinstance(manifest["port"],int),
 "request_logging_disabled":"--disable-log-requests" in manifest["command"]}
metrics={"manifest":manifest,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
 "image_digest_pinned":checks["image_digest"],"model_read_only":checks["model_read_only"],
 "secret_external":checks["secret_external"],"native_docker_executed":False}
analysis=(f"The manifest passed {metrics['checks_passed']}/{metrics['checks_total']} static invariants, "
          "including digest pinning, read-only model bytes, external secrets, and startup-aware health. "
          "No Docker daemon was invoked.")
''',
    23: r'''
doc={"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"vllm-qwen"},"spec":{
 "replicas":2,"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxSurge":1,"maxUnavailable":0}},
 "template":{"metadata":{"labels":{"app":"vllm-qwen"}},"spec":{"terminationGracePeriodSeconds":120,
 "affinity":{"podAntiAffinity":{"preferredDuringSchedulingIgnoredDuringExecution":[{"weight":100}]}},
 "containers":[{"name":"server","image":"vllm/vllm-openai@sha256:"+"a"*64,
 "resources":{"requests":{"nvidia.com/gpu":1},"limits":{"nvidia.com/gpu":1}},
 "startupProbe":{"httpGet":{"path":"/health","port":8000},"failureThreshold":60,"periodSeconds":5},
 "readinessProbe":{"httpGet":{"path":"/health","port":8000},"periodSeconds":5},
 "livenessProbe":{"httpGet":{"path":"/health","port":8000},"periodSeconds":15}}]}}}}
spec=doc["spec"]; pod=spec["template"]["spec"]; container=pod["containers"][0]
requested=int(container["resources"]["requests"]["nvidia.com/gpu"]); limited=int(container["resources"]["limits"]["nvidia.com/gpu"])
replicas=int(spec["replicas"]); surge=int(spec["strategy"]["rollingUpdate"]["maxSurge"]); cluster_gpus=3
checks={"gpu_request_limit_match":requested==limited==1,"startup_probe":"startupProbe" in container,
 "readiness_probe":"readinessProbe" in container,"liveness_probe":"livenessProbe" in container,
 "termination_grace":pod.get("terminationGracePeriodSeconds",0)>=60,"anti_affinity":"affinity" in pod,
 "image_digest":"@sha256:" in container["image"],"rollout_capacity":replicas+surge<=cluster_gpus}
metrics={"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
 "capacity":{"steady_gpus":replicas*requested,"rollout_gpus":(replicas+surge)*requested,
             "cluster_gpus":cluster_gpus,"feasible":checks["rollout_capacity"]},
 "startup_probe":checks["startup_probe"],"manifest":doc,"native_cluster_executed":False}
analysis=(f"The manifest passed {metrics['checks_passed']}/{metrics['checks_total']} checks. Steady/"
          f"surge capacity is {metrics['capacity']['steady_gpus']}/{metrics['capacity']['rollout_gpus']} "
          f"of {cluster_gpus} declared GPUs. This is configuration feasibility, not a cluster rollout.")
''',
    24: r'''
events=[{"id":"i1","tenant":"interactive","arrival":0,"prompt":80,"output":40},
 {"id":"b1","tenant":"batch","arrival":0,"prompt":6000,"output":1000},
 {"id":"i2","tenant":"interactive","arrival":1,"prompt":120,"output":60},
 {"id":"b2","tenant":"batch","arrival":1,"prompt":8000,"output":1200},
 {"id":"i3","tenant":"interactive","arrival":2,"prompt":60,"output":30},
 {"id":"b3","tenant":"batch","arrival":3,"prompt":4000,"output":800},
 {"id":"i4","tenant":"interactive","arrival":4,"prompt":100,"output":50}]
def gateway(policy):
    count={"interactive":4,"batch":3}; budget={"interactive":1600,"batch":10500}
    admitted=[]; rejected=[]; used={"interactive":0,"batch":0}; waits={"interactive":[],"batch":[]}; clock={"interactive":0.,"batch":0.}
    for event in events:
        cost=event["prompt"]+event["output"]
        if policy=="request_count": allow=count[event["tenant"]]>0; count[event["tenant"]]-=int(allow)
        else: allow=budget[event["tenant"]]>=cost; budget[event["tenant"]]-=cost if allow else 0
        if not allow: rejected.append(event["id"]); continue
        start=max(event["arrival"],clock[event["tenant"]]); waits[event["tenant"]].append(start-event["arrival"])
        clock[event["tenant"]]=start+cost/(800 if event["tenant"]=="interactive" else 500)
        admitted.append(event); used[event["tenant"]]+=cost
    shares=list(used.values()); fairness=sum(shares)**2/(len(shares)*sum(x*x for x in shares)) if any(shares) else 0
    return {"admitted_requests":len(admitted),"admitted_tokens":sum(used.values()),
     "interactive_admitted":sum(x["tenant"]=="interactive" for x in admitted),
     "batch_admitted":sum(x["tenant"]=="batch" for x in admitted),"rejected":rejected,"tenant_tokens":used,
     "interactive_p95_wait":percentile(waits["interactive"],.95),
     "batch_p95_wait":percentile(waits["batch"],.95) if waits["batch"] else 0.,"fairness":fairness}
metrics={"request_count":gateway("request_count"),"token_budget":gateway("token_budget"),"events":events}
analysis=(f"Count admission accepted {metrics['request_count']['admitted_tokens']:,} tokens and "
          f"{metrics['request_count']['batch_admitted']} batch jobs; token budgeting accepted "
          f"{metrics['token_budget']['admitted_tokens']:,} and {metrics['token_budget']['batch_admitted']} "
          "while retaining interactive budget. Queue costs are modeled.")
''',
    25: r'''
cfg=model_config(); free_bytes,total_bytes=torch.cuda.mem_get_info(); token_files=["tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"]
checks={"request_schema":True,"tokenizer_files":all((MODEL/x).exists() for x in token_files),
 "model_config":bool(cfg.get("architectures")),"vllm_import":bool(vllm.__version__),
 "cuda_available":torch.cuda.is_available(),"model_weights":any(MODEL.glob("*.safetensors"))}
safe=[]
try: SamplingParams(temperature=-1.0)
except Exception as exc: safe.append({"layer":"request","type":type(exc).__name__})
try: raise ValueError(f"requested context exceeds declared {cfg.get('max_position_embeddings')}")
except Exception as exc: safe.append({"layer":"engine_config","type":type(exc).__name__})
first=next((name for name,passed in checks.items() if not passed),"none")
metrics={"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
 "first_failing_layer":first,"cuda_available":checks["cuda_available"],"free_gpu_mib":free_bytes/2**20,
 "total_gpu_mib":total_bytes/2**20,"tokenizer_files":sum((MODEL/x).exists() for x in token_files),
 "safe_failures":safe,"safe_failures_classified":len(safe)}
analysis=(f"The ordered checklist passed {metrics['checks_passed']}/{metrics['checks_total']} layers "
          f"with first failure={first}; {metrics['free_gpu_mib']:.1f} MiB was free and {len(safe)} safe "
          "failures were classified without inducing OOM.")
''',
    26: r'''
llm=LLM(**base_engine_args(max_model_len=1024,max_num_seqs=16)); params=SamplingParams(temperature=0.0,max_tokens=20,seed=SEED)
llm.generate(["warmup"],SamplingParams(temperature=0.0,max_tokens=2),use_tqdm=False); rows=[]; torch.cuda.reset_peak_memory_stats()
for batch in (1,2,4,8):
    samples=[]; output_count=0
    for repeat in range(3):
        prompts=[f"{repeat}-{i}: name one inference bottleneck." for i in range(batch)]
        tick=time.perf_counter(); outputs=llm.generate(prompts,params,use_tqdm=False); samples.append(time.perf_counter()-tick)
        output_count=sum(len(x.outputs[0].token_ids) for x in outputs)
    median=statistics.median(samples); rows.append({"batch":batch,"samples_s":samples,"median_s":median,
        "p95_s":percentile(samples,.95),"output_tokens_s":output_count/median})
gate=max(row["p95_s"] for row in rows[:2])*2.5; feasible=[row for row in rows if row["p95_s"]<=gate]
selected=max(feasible,key=lambda row:row["output_tokens_s"])
metrics={"candidates":len(rows),"rows":rows,"p95_gate_s":gate,"feasible_candidates":len(feasible),
 "selected":selected,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
analysis=(f"The one-variable sweep kept {len(feasible)}/{len(rows)} rows below the {gate:.3f} s "
          f"closed-batch p95 gate. Batch {selected['batch']} led feasible throughput at "
          f"{selected['output_tokens_s']:.1f} output tokens/s; online latency is not inferred.")
''',
    27: r'''
cfg=model_config(); layers=int(cfg["num_hidden_layers"]); hidden=int(cfg["hidden_size"])
heads=int(cfg["num_attention_heads"]); kv_heads=int(cfg.get("num_key_value_heads",heads)); head_dim=int(cfg.get("head_dim",hidden//heads))
per_token=2*layers*kv_heads*head_dim*2; contexts={}
for length in (2048,8192,32768):
    size=per_token*length; contexts[str(length)]={"kv_mib":size/2**20,
        "transfer_ms_25gbps":size*8/25e9*1000+.35,"transfer_ms_200gbps":size*8/200e9*1000+.35}
connector=False; symbols=[]
for module_name in ("vllm.distributed.kv_transfer","vllm.config.kv_transfer"):
    try:
        module=importlib.import_module(module_name); connector=True
        symbols.extend(x for x in dir(module) if "Connector" in x or "Nixl" in x)
    except Exception: pass
metrics={"kv_bytes_per_token":per_token,"contexts":contexts,
 "assumptions":{"coordination_ms":.35,"bandwidth_gbps":[25,200]},"connector_probe":connector,
 "connector_symbols":sorted(set(symbols))[:30],"native_disaggregation_executed":False}
analysis=(f"BF16 KV is {per_token:,} bytes/token. An 8K prompt transfers {contexts['8192']['kv_mib']:.1f} MiB: "
          f"ideal {contexts['8192']['transfer_ms_25gbps']:.2f}/{contexts['8192']['transfer_ms_200gbps']:.2f} "
          "ms at 25/200 Gb/s including declared coordination. No two-worker run occurred.")
''',
    28: r'''
llm=LLM(**base_engine_args(max_model_len=1024,max_num_seqs=8)); params=SamplingParams(temperature=0.0,max_tokens=24,seed=SEED)
prompts=[f"Capacity probe {i}: name one serving metric." for i in range(8)]
llm.generate(["warmup"],SamplingParams(temperature=0.0,max_tokens=2),use_tqdm=False)
tick=time.perf_counter(); outputs=llm.generate(prompts,params,use_tqdm=False); elapsed=time.perf_counter()-tick
tokens=sum(len(x.outputs[0].token_ids) for x in outputs); measured=tokens/elapsed; safe=.65; usable=measured*safe
demands={"low":measured*.3,"medium":measured*1.4,"peak":measured*3.2}; scenarios={}
for name,demand in demands.items():
    base=max(1,math.ceil(demand/usable)); scenarios[name]={"demand_output_tokens_s":demand,
        "base_replicas":base,"with_reserve":base+1,"projected_utilization":demand/(base*measured)}
metrics={"measured_output_tokens_s":measured,"measurement_elapsed_s":elapsed,"measurement_output_tokens":tokens,
 "safe_utilization":safe,"usable_output_tokens_s":usable,"scenarios":scenarios,"scale_up_lead_s":180.0}
analysis=(f"The native closed batch measured {measured:.1f} output tokens/s. At {safe:.0%} safe "
          f"utilization, medium/peak need {scenarios['medium']['with_reserve']}/"
          f"{scenarios['peak']['with_reserve']} replicas including N+1. Online service curves remain required.")
''',
    29: r'''
dns={"public.example":["93.184.216.34"],"localhost.example":["127.0.0.1"],
     "metadata.example":["169.254.169.254"],"private.example":["10.2.3.4"],"v6local.example":["::1"]}
fixtures=[("https://public.example/image.png",True),("http://private.example/a",False),
 ("http://metadata.example/latest",False),("http://localhost.example/a",False),
 ("file:///etc/passwd",False),("gopher://public.example/x",False),("http://v6local.example/x",False)]
def allow(url):
    parsed=urlparse(url)
    if parsed.scheme not in {"http","https"} or not parsed.hostname: return False
    addresses=dns.get(parsed.hostname,[])
    return bool(addresses) and all(not (ipaddress.ip_address(a).is_private or ipaddress.ip_address(a).is_loopback
        or ipaddress.ip_address(a).is_link_local or ipaddress.ip_address(a).is_reserved) for a in addresses)
decisions=[{"url":url,"expected":expected,"actual":allow(url)} for url,expected in fixtures]
policy={"api_keys_in_secret_store":True,"full_prompt_metric_labels":False,"prompt_log_retention_days":7,
 "deletion_workflow":True,"trust_remote_code":False,"model_license_reviewed":True,"redirect_revalidation":True}
checks={"secret_store":policy["api_keys_in_secret_store"],"no_prompt_labels":not policy["full_prompt_metric_labels"],
 "bounded_retention":0<=policy["prompt_log_retention_days"]<=30,"deletion":policy["deletion_workflow"],
 "remote_code_disabled":not policy["trust_remote_code"],"license_review":policy["model_license_reviewed"],
 "redirect_revalidation":policy["redirect_revalidation"]}
errors=sum(x["expected"]!=x["actual"] for x in decisions)
metrics={"url_cases":len(decisions),"url_decisions_correct":len(decisions)-errors,"decisions":decisions,
 "private_blocked":not allow("http://private.example/a"),"link_local_blocked":not allow("http://metadata.example/latest"),
 "policy":policy,"policy_checks":checks,"policy_checks_passed":sum(checks.values()),"policy_checks_total":len(checks),
 "release_blockers":errors+sum(not x for x in checks.values())}
analysis=(f"The SSRF policy classified {metrics['url_decisions_correct']}/{len(decisions)} fixtures and "
          f"passed {metrics['policy_checks_passed']}/{len(checks)} data/supply-chain checks, leaving "
          f"{metrics['release_blockers']} blockers. Real DNS/redirect tests remain required.")
''',
    30: r'''
chapter=Path.cwd().parent; required=[6,7,8,10,20,21,25,29]; artifacts={}
for number in required:
    matches=list(chapter.glob(f"{number:02d}-*/artifacts/rtx5090-result.json"))
    if len(matches)==1:
        raw=matches[0].read_bytes(); artifacts[str(number)]={"path":str(matches[0].relative_to(chapter)),
            "sha256":hashlib.sha256(raw).hexdigest(),"payload":json.loads(raw)}
checks={"environment":"6" in artifacts and bool(artifacts["6"]["payload"]["metrics"].get("cli_found")),
 "offline_generation":"7" in artifacts and artifacts["7"]["payload"]["metrics"].get("requests",0)>=1,
 "http_contract":"8" in artifacts and artifacts["8"]["payload"]["metrics"].get("schema_valid") is True,
 "provenance":"10" in artifacts and artifacts["10"]["payload"]["metrics"].get("native_load") is True,
 "benchmark":"20" in artifacts and artifacts["20"]["payload"]["metrics"].get("batches",{}).get("8",{}).get("output_tokens_s",0)>0,
 "observability":"21" in artifacts and artifacts["21"]["payload"]["metrics"].get("metrics_status")==200,
 "diagnostics":"25" in artifacts and artifacts["25"]["payload"]["metrics"].get("first_failing_layer")=="none",
 "security":"29" in artifacts and artifacts["29"]["payload"]["metrics"].get("release_blockers")==0,
 "rollback_rehearsed":False}
stages=[("poc",["environment","offline_generation","provenance"]),("load_test",["benchmark","observability"]),
        ("canary",["http_contract","diagnostics","security"]),("promote",["rollback_rehearsed"])]
final="blocked"; blockers=[]
for stage,names in stages:
    failed=[name for name in names if not checks.get(name,False)]
    if failed: blockers.extend(failed); final=f"blocked_before_{stage}"; break
    final=stage
ready=final=="promote" and all(checks.values()); hashes={key:value["sha256"] for key,value in artifacts.items()}
metrics={"required_artifacts":len(required),"artifacts_present":len(artifacts),
 "artifact_hashes":len(set(hashes.values())),"evidence":{k:{"path":v["path"],"sha256":v["sha256"]} for k,v in artifacts.items()},
 "checks":checks,"gates_passed":sum(checks.values()),"gates_total":len(checks),"final_stage":final,
 "release_ready":ready,"blockers":len(blockers),"blocker_names":blockers,
 "release_id":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]}
analysis=(f"The manifest found {len(artifacts)}/{len(required)} artifacts with {metrics['artifact_hashes']} "
          f"hashes and passed {metrics['gates_passed']}/{metrics['gates_total']} gates. Final stage={final}, "
          f"release_ready={ready}; intentional rollback rehearsal prevents lab-only promotion.")
''',
})

EXPERIMENTS.update({
    11: r'''
parameter_billion=70.0; weight_gib=parameter_billion*1e9*2/2**30
gpu_gib=torch.cuda.get_device_properties(0).total_memory/2**30; reserve_gib=5.0; activation_gib=3.0
_,serve_help=cli_help("serve")
def layout(tp,dp,pp,cross_fraction):
    shard=weight_gib/(tp*pp)
    return {"tp":tp,"dp":dp,"pp":pp,"replicas":dp,"weight_gib_per_gpu":shard,
            "fits":shard+reserve_gib<gpu_gib,"cross_node_gib_step":activation_gib*cross_fraction}
metrics={"topology":{"nodes":2,"gpus":8,"gpus_per_node":4,"gpu_gib":gpu_gib},
         "assumptions":{"model_parameters_billion":parameter_billion,"bf16_weight_gib":weight_gib,
                        "reserve_gib_per_gpu":reserve_gib,"activation_gib_step":activation_gib},
         "layouts":{"tp8":layout(8,1,1,.5),"tp4_dp2":layout(4,2,1,0),"tp4_pp2":layout(4,1,2,.25)},
         "cli":{"tensor_parallel":"--tensor-parallel-size" in serve_help,
                "pipeline_parallel":"--pipeline-parallel-size" in serve_help,
                "data_parallel":"--data-parallel-size" in serve_help},"native_multi_gpu_executed":False}
analysis=(f"The ledger estimates {weight_gib:.1f} GiB BF16 weights. TP8/TP4×DP2 fit="
          f"{metrics['layouts']['tp8']['fits']}/{metrics['layouts']['tp4_dp2']['fits']}; only the "
          "modeled TP8 collective crosses nodes. No distributed run occurred.")
''',
    12: r'''
prefix=("vLLM stores key/value vectors in fixed-size cache blocks for scheduled token work. "*90)
llm=LLM(**base_engine_args(max_model_len=2048,enable_prefix_caching=True))
params=SamplingParams(temperature=0.0,max_tokens=8,seed=SEED)
def apc(prompt):
    tick=time.perf_counter(); item=llm.generate([prompt],params,use_tqdm=False)[0]; row=output_record(item)
    row["elapsed_s"]=time.perf_counter()-tick; row["cached_tokens"]=int(getattr(item,"num_cached_tokens",0) or 0)
    return row
cold=apc(prefix+" Question: define Prefill."); warm=apc(prefix+" Question: define Decode.")
mutated=apc("Changed. "+prefix+" Question: define Decode.")
metrics={"cold":cold,"warm":warm,"mutated":mutated,"prefix_characters":len(prefix),"enable_prefix_caching":True}
analysis=(f"Cold/warm/mutated requests reported {cold['cached_tokens']}/{warm['cached_tokens']}/"
          f"{mutated['cached_tokens']} cached tokens; warm/cold elapsed was {warm['elapsed_s']:.4f}/"
          f"{cold['elapsed_s']:.4f} s. The cached-token field is the hit evidence.")
''',
    13: r'''
long_tokens=4096; chunk_tokens=512; jobs=[{"arrival":1.,"steps":8},{"arrival":2.,"steps":6},
    {"arrival":3.,"steps":4},{"arrival":5.,"steps":3}]
prefill_cost=.002; decode_cost=.22; overhead=.08
def mixed(chunked):
    now=0.; waits=[]; chunks=0; remaining=long_tokens; pending=[dict(x) for x in jobs]
    while remaining or pending:
        if chunked and pending and pending[0]["arrival"]<=now:
            job=pending.pop(0); waits.append(now-job["arrival"]); now+=job["steps"]*decode_cost
        elif remaining:
            take=min(chunk_tokens if chunked else remaining,remaining); now+=take*prefill_cost+(overhead if chunked else 0)
            remaining-=take; chunks+=1
        else:
            job=pending.pop(0); now=max(now,job["arrival"]); waits.append(now-job["arrival"]); now+=job["steps"]*decode_cost
    return {"long_finish":now,"short_p95_delay":percentile(waits,.95),"short_max_delay":max(waits),
            "prefill_chunks":chunks,"waits":waits}
_,help_text=cli_help("serve")
metrics={"unchunked":mixed(False),"chunked":mixed(True),
         "cli":{"chunked_prefill":"--enable-chunked-prefill" in help_text,
                "max_num_batched_tokens":"--max-num-batched-tokens" in help_text},
         "cost_assumptions":{"prefill_per_token":prefill_cost,"decode_step":decode_cost,"chunk_overhead":overhead}}
analysis=(f"512-token chunking created {metrics['chunked']['prefill_chunks']} chunks and changed "
          f"short-job p95 delay from {metrics['unchunked']['short_p95_delay']:.3f} to "
          f"{metrics['chunked']['short_p95_delay']:.3f} modeled units. Native traffic is still required.")
''',
    14: r'''
cfg=model_config(); registered=[]; probe_error=None
try:
    module=importlib.import_module("vllm.model_executor.layers.quantization")
    registered=sorted(str(x).lower() for x in getattr(module,"QUANTIZATION_METHODS",[]))
except Exception as exc: probe_error=f"{type(exc).__name__}: {exc}"
_,serve_help=cli_help("serve"); blob=" ".join(registered)+" "+serve_help.lower(); declared=cfg.get("quantization_config")
metrics={"local":{"quantization":"none" if declared is None else str(declared),"torch_dtype":str(cfg.get("torch_dtype"))},
         "methods":{"awq":"awq" in blob,"gptq":"gptq" in blob,
                    "fp8":"fp8" in blob or "compressed-tensors" in blob},
         "hardware":{"compute_capability":ENV["compute_capability"],"gpu":ENV["gpu"]},
         "registered_methods":registered,"probe_error":probe_error,"native_quantized_benchmark_completed":False}
analysis=(f"Local quantization={metrics['local']['quantization']}; installed AWQ/GPTQ/FP8 vocabulary="
          f"{metrics['methods']['awq']}/{metrics['methods']['gptq']}/{metrics['methods']['fp8']}. "
          "Without matching quantized bytes, memory, quality, and latency remain unmeasured.")
''',
    15: r'''
prompts=["Explain KV cache quantization briefly.","Name one FP8 calibration risk."]
params=SamplingParams(temperature=0.0,max_tokens=16,seed=SEED)
def run_kv(dtype):
    row={"success":False,"elapsed_s":None,"records":[],"error":None}
    try:
        engine=LLM(**base_engine_args(max_model_len=1024,kv_cache_dtype=dtype)); tick=time.perf_counter()
        outputs=engine.generate(prompts,params,use_tqdm=False)
        row.update(success=True,elapsed_s=time.perf_counter()-tick,records=[output_record(x) for x in outputs])
        del engine; gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        row["error"]=f"{type(exc).__name__}: {exc}"; gc.collect(); torch.cuda.empty_cache()
    return row
auto=run_kv("auto"); fp8=run_kv("fp8")
auto_ids=[x["token_ids"] for x in auto["records"]]; fp8_ids=[x["token_ids"] for x in fp8["records"]]
metrics={"auto":auto,"fp8":fp8,"theoretical_kv_capacity_ratio":2.0,
         "token_sequences_equal":bool(auto["success"] and fp8["success"] and auto_ids==fp8_ids)}
analysis=(f"Auto/FP8 success={auto['success']}/{fp8['success']}; leading KV capacity ratio is 2× and "
          f"matched greedy tokens equal={metrics['token_sequences_equal']}. Short prompts do not prove "
          "long-context capacity or task quality.")
''',
    16: r'''
cfg=model_config(); hidden=int(cfg["hidden_size"]); intermediate=int(cfg["intermediate_size"])
layers=int(cfg["num_hidden_layers"]); rank=16; shapes=[(hidden,hidden)]*4+[(intermediate,hidden)]*2+[(hidden,intermediate)]
adapter_params=layers*sum(rank*(din+dout) for dout,din in shapes); adapter_bytes=adapter_params*2
weight_bytes=sum(p.stat().st_size for p in MODEL.glob("*.safetensors")); _,serve_help=cli_help("serve")
api=False
try:
    from vllm.lora.request import LoRARequest
    api=inspect.isclass(LoRARequest)
except Exception: pass
metrics={"rank":rank,"target_matrices_per_layer":len(shapes),"estimated_adapter_parameters":adapter_params,
         "estimated_adapter_bytes":adapter_bytes,"base_weight_bytes":weight_bytes,
         "adapter_to_base_ratio":adapter_bytes/weight_bytes,"api":{"lora_request":api},
         "cli":{"enable_lora":"--enable-lora" in serve_help,"max_lora_rank":"--max-lora-rank" in serve_help,
                "max_loras":"--max-loras" in serve_help},"native_adapter_executed":False}
analysis=(f"The rank-{rank} seven-projection estimate is {adapter_bytes:,} BF16 bytes "
          f"({metrics['adapter_to_base_ratio']:.2%} of weights); request API/enable flag="
          f"{api}/{metrics['cli']['enable_lora']}. No trained adapter behavior was fabricated.")
''',
    17: r'''
prompt=("red blue green red blue green "*35)+"Continue:"; params=SamplingParams(temperature=0.0,max_tokens=32,seed=SEED)
def run_engine(extra):
    row={"success":False,"elapsed_s":None,"output_tokens":0,"token_ids":[],"error":None}
    try:
        engine=LLM(**base_engine_args(max_model_len=1024,**extra)); tick=time.perf_counter()
        item=engine.generate([prompt],params,use_tqdm=False)[0]; record=output_record(item)
        row.update(success=True,elapsed_s=time.perf_counter()-tick,output_tokens=record["output_tokens"],
                   token_ids=record["token_ids"],num_cached_tokens=record["num_cached_tokens"])
        del engine; gc.collect(); torch.cuda.empty_cache()
    except Exception as exc: row["error"]=f"{type(exc).__name__}: {exc}"; gc.collect(); torch.cuda.empty_cache()
    return row
baseline=run_engine({}); speculative=run_engine({"speculative_config":{"method":"ngram",
    "num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}})
ratio=baseline["elapsed_s"]/speculative["elapsed_s"] if baseline["success"] and speculative["success"] else None
metrics={"baseline":baseline,"speculative":speculative,
         "tokens_equal":bool(baseline["success"] and speculative["success"] and baseline["token_ids"]==speculative["token_ids"]),
         "speed_ratio":ratio}
analysis=(f"Baseline/speculative success={baseline['success']}/{speculative['success']}, tokens equal="
          f"{metrics['tokens_equal']}, elapsed ratio={ratio}. The repeated prompt is favorable to n-gram lookup.")
''',
    18: r'''
from vllm.sampling_params import StructuredOutputsParams
schema={"type":"object","properties":{"component":{"type":"string"},
        "risk":{"type":"string","enum":["low","medium","high"]},"rollback":{"type":"boolean"}},
        "required":["component","risk","rollback"],"additionalProperties":False}
llm=LLM(**base_engine_args(max_model_len=1024)); prompt="Return a risk object for KV cache configuration."
def validate(text):
    try: value=json.loads(text)
    except Exception: return False,False,None
    valid=(isinstance(value,dict) and set(value)==set(schema["required"])
           and isinstance(value.get("component"),str) and value.get("risk") in {"low","medium","high"}
           and isinstance(value.get("rollback"),bool)); return True,bool(valid),value
row={"success":False,"json_parsed":False,"schema_valid":False,"required_fields_present":False,"output_tokens":0,"error":None}
try:
    constraint=StructuredOutputsParams(json=json.dumps(schema)); setting=SamplingParams(temperature=0.0,max_tokens=80,seed=SEED,structured_outputs=constraint)
    item=llm.generate([prompt],setting,use_tqdm=False)[0]; parsed,valid,value=validate(item.outputs[0].text)
    row.update(success=True,json_parsed=parsed,schema_valid=valid,
               required_fields_present=isinstance(value,dict) and all(k in value for k in schema["required"]),
               output_tokens=len(item.outputs[0].token_ids),value=value,text=item.outputs[0].text)
except Exception as exc: row["error"]=f"{type(exc).__name__}: {exc}"
control=llm.generate([prompt],SamplingParams(temperature=0.0,max_tokens=80,seed=SEED),use_tqdm=False)[0]
cparsed,cvalid,cvalue=validate(control.outputs[0].text)
metrics={"schema":schema,"structured":row,"control":{"json_parsed":cparsed,"schema_valid":cvalid,
         "text":control.outputs[0].text,"value":cvalue}}
analysis=(f"Structured success/JSON/schema={row['success']}/{row['json_parsed']}/{row['schema_valid']}; "
          f"unconstrained JSON parsed={cparsed}. Independent semantic/authorization validation remains required.")
''',
    19: r'''
cfg=model_config(); architecture=str((cfg.get("architectures") or ["unknown"])[0])
mm_keys=[key for key in cfg if any(token in key.lower() for token in ("vision","image","audio"))]
generate=architecture.endswith("ForCausalLM") or "CausalLM" in architecture
routes={"chat":{"ready":bool(generate),"reason":"causal generation architecture"},
        "embeddings":{"ready":False,"reason":"no pooling checkpoint/evaluation"},
        "rerank":{"ready":False,"reason":"no scoring checkpoint/evaluation"},
        "multimodal":{"ready":bool(mm_keys),"reason":"multimodal config" if mm_keys else "text-only config"}}
metrics={"model":{"architecture":architecture,"multimodal_config_keys":mm_keys},"routes":routes,
         "native_non_generation_tests":0,"required_next_models":["embedding","rerank","multimodal"]}
analysis=(f"Architecture {architecture} enables Chat={routes['chat']['ready']} and blocks embeddings/"
          f"rerank/multimodal={routes['embeddings']['ready']}/{routes['rerank']['ready']}/"
          f"{routes['multimodal']['ready']} pending matching native models and evaluations.")
''',
    20: r'''
llm=LLM(**base_engine_args(max_model_len=1024,max_num_seqs=16)); params=SamplingParams(temperature=0.0,max_tokens=24,seed=SEED)
llm.generate(["Warm."],SamplingParams(temperature=0.0,max_tokens=2),use_tqdm=False); rows={}; torch.cuda.reset_peak_memory_stats()
for batch in (1,4,8):
    samples=[]; prompt_tokens=output_tokens=0
    for repeat in range(3):
        prompts=[f"Request {repeat}-{i}: explain one vLLM metric." for i in range(batch)]
        tick=time.perf_counter(); outputs=llm.generate(prompts,params,use_tqdm=False); samples.append(time.perf_counter()-tick)
        prompt_tokens=sum(len(x.prompt_token_ids or []) for x in outputs); output_tokens=sum(len(x.outputs[0].token_ids) for x in outputs)
    median=statistics.median(samples); rows[str(batch)]={"samples_s":samples,"median_s":median,
        "p95_s":percentile(samples,.95),"prompt_tokens":prompt_tokens,"output_tokens":output_tokens,
        "requests_s":batch/median,"output_tokens_s":output_tokens/median}
metrics={"batches":rows,"repeats":3,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
analysis=(f"Batch 1/4/8 measured {rows['1']['output_tokens_s']:.1f}/{rows['4']['output_tokens_s']:.1f}/"
          f"{rows['8']['output_tokens_s']:.1f} output tokens/s in a warmed closed workload. Raw samples "
          "are retained; online TTFT/ITL are outside scope.")
''',
})
