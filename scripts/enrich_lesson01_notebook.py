#!/usr/bin/env python3
"""Idempotently add theory-to-code bridges to the executed Lesson 01 notebook."""

from __future__ import annotations

import json
from pathlib import Path

from tutorial_guides import CHAPTER_01_GUIDES, render_guide


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "chapters" / "01-mixed-precision-int4" / "01-precision-formats" / "lab.ipynb"


LESSON_01_GUIDE = render_guide(CHAPTER_01_GUIDES[1])


BLOCKS = {
    "## 1. Measurement protocol": ("l01-theory-formats", f"""## Theory bridge — a format is not a speed rank

A precision decision has at least five parts: **storage representation, scale or
packing metadata, compute input dtype, accumulation dtype, and the operator that
consumes the tensor**. TF32, for example, normally keeps FP32 storage while
changing eligible matrix-multiply execution; weight-only INT4 changes weight
storage but normally keeps activations and accumulation wider.

The first theoretical prediction is therefore about memory only:

```text
ideal weight bytes = parameter count × nominal bytes per stored weight
```

Real allocated memory adds unquantized layers, group scales, packed-layout
metadata, temporary conversion buffers, activations, cache, workspaces, and
allocator behavior. The notebook records several ledger lines instead of
pretending that nominal bit width equals runtime memory.

{LESSON_01_GUIDE}"""),
    "## 2. Run the BF16 baseline": ("l01-theory-baseline", """## Theory bridge — why the BF16 baseline comes first

The baseline freezes the checkpoint, prompt tokens, batch, sequence lengths,
warm-up, repetitions, decoding policy, and quality probe. BF16 is not assumed
perfect; it is the control needed to attribute any later change to the INT4
candidate. Loading it in a separate process also prevents the candidate's
packing and allocator state from contaminating baseline memory."""),
    "## 3. Run the TorchAO INT4 candidate": ("l01-theory-int4", """## Theory bridge — what weight-only INT4 actually executes

For each quantized weight group, a reference view is
`q = clamp(round(w / s), -8, 7)` and `w_hat = s·q`. A production weight-only
operator reads packed codes and scales while BF16 activations enter the linear
layer. Smaller storage can reduce memory traffic, but scale loads, unpack or
dequantization, unsupported layers, small-batch shapes, and kernel launch
overhead can outweigh that saving.

This is why the candidate must prove three independent facts: modules were
converted, an INT4 operator appeared in the profiler, and measured latency
changed under the same workload."""),
    "## 4. Build the comparison from this run": ("l01-theory-evidence", """## Theory bridge — read four evidence axes separately

The comparison is not one winner column. It asks:

1. **Storage/runtime memory:** did stable active bytes and peak bytes fall?
2. **Operator path:** did the intended packed INT4 operation execute?
3. **Performance:** did Prefill and approximate Decode improve at the tested shapes?
4. **Quality:** did outputs remain within the small frozen regression probe?

A pass on one axis does not imply a pass on another. In particular, a real INT4
kernel can execute and save memory while still losing latency."""),
    "## 6. Make a bounded decision": ("l01-theory-decision", """## Theory bridge — the decision and its reversal conditions

Keep INT4 as the default only if the required capacity gain is real, the native
path executes, quality stays within the predeclared gate, and the relevant
Prefill/Decode or service SLO improves. Otherwise keep BF16 or use INT4 only as
a capacity fallback. A different backend, offline-packed checkpoint, model
shape, batch, or software release is a valid reason to rerun—not a reason to
generalize this result away."""),
}


def main() -> int:
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    blocked_ids = {value[0] for value in BLOCKS.values()}
    notebook["cells"] = [cell for cell in notebook["cells"] if cell.get("id") not in blocked_ids]
    rebuilt = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        for heading, (cell_id, theory) in BLOCKS.items():
            if source.startswith(heading):
                rebuilt.append({"id":cell_id,"cell_type":"markdown","metadata":{},"source":theory})
                break
        rebuilt.append(cell)
    notebook["cells"] = rebuilt
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Added {len(BLOCKS)} theory bridges to {PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
