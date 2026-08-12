# Lesson 22 — Packaging an INT4 Inference Deliverable

> **Puzzle:** What files make a quantized model reproducible rather than merely loadable on one machine?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A quantized model is deployable only when its bytes and interpretation travel together.
Packed weights without scales are meaningless; correct weights with the wrong tokenizer
or base revision are unsafe; and an artifact without a checksum cannot be distinguished
from a partial copy. Packaging is therefore part of inference correctness, not
administrative cleanup.

## Predict before reading the result

1. List the minimum fields needed to load, validate, and roll back a quantized artifact.
2. Predict the packed payload size for the notebook's weight and scale tensors.
3. Explain what a SHA-256 digest proves and what semantic errors it cannot detect.

## 1. Start from concrete tensors and state

A deployable package binds tensor shards, scales/zero points, shapes and packing schema,
base/tokenizer revisions, runtime requirements, checksums, smoke vectors, and rollback
identity.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | A deliverable binds base revision, quantization recipe, tokenizer, tensor shapes, scales, packing, and runtime requirements. |
| 2 | Checksums detect corruption but do not validate semantics. |
| 3 | A smoke test and rollback pointer belong beside the artifact. |

## 2. Derive the mechanism

A cryptographic hash verifies bytes, while a schema verifies meaning. Both are needed:
identical shapes with the wrong scale axis can be semantically corrupt yet perfectly
hash-consistent.

A package contract binds schema version, base revision, quantization format, group
size/axis, tensor shapes, scale dtype, runtime/backend requirement, checksums, and
rollback target. The checksum establishes byte identity; the schema establishes how
those bytes should be decoded. Both are needed.

Production packages also include tokenizer/config files, special-token policy,
architecture code revision, licenses, model card, and quality/performance reports.
Keeping a minimal manifest in the lab makes the invariant testable without publishing
checkpoint data.

## 3. Translate the theory into an experiment

**Experiment:** Create an in-memory synthetic INT4 shard on CUDA, serialize only a tiny temporary payload, verify its checksum and manifest fields, then delete the temporary file.

| Experimental role | Frozen definition |
|---|---|
| Baseline | unversioned in-memory reference weights with no handoff contract |
| Candidate | temporary packed INT4 payload plus validated manifest and checksum |
| Held constant | fixed tensor shape, group size, serializer, required-field set, rollback ID |
| Measurements | payload bytes, SHA-256, required-field completeness, cleanup status |
| Evidence label | `pytorch-gpu` |

The lab creates a tiny temporary packed payload, hashes and validates its manifest, and
deletes it so no model checkpoint enters the repository.

### Code walk-through

The notebook quantizes a 256×512 matrix, serializes codes and scales into a temporary
file, computes SHA-256, records size and interpretation fields, validates required keys,
and lets the temporary payload disappear after the check. Only the small manifest
evidence remains public.

The exercise proves packaging logic without committing weights. It does not claim
compatibility with SafeTensors, Hugging Face quantization configs, or a named production
runtime.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Manifest complete | yes |
| Payload bytes | 139,264 bytes |
| Format | reference-int4 |
| Group size | 64 |
| SHA-256 | `bd46d808…6714de` |
| Temporary payload removed | yes |

### What the numbers mean

The generated reference payload was 139,264 bytes and received digest `bd46d8…714de`.
Every required manifest field was present, including base revision, format, group size,
shape, runtime, and BF16 rollback target. The temporary payload was deleted after
validation.

This is a reproducibility and safety result: another process can verify identity and
interpretation metadata. It is not a model export, engine load, or distribution license
decision.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Ship a versioned contract with hashes, schema, compatibility, smoke test, and rollback—not a loose weight file.

### Acceptance and rollback gate

Test fresh-environment load, hash verification, schema validation, deterministic smoke
output, memory budget, native operator, and rollback artifact before release.

### How this conclusion can fail

A digest cannot detect that the wrong scale axis was declared if both producer and
consumer share the same bad schema. Mutable model tags and missing tokenizer revisions
also break reproducibility. Never place secrets, local paths, proprietary weights, or
unlicensed datasets in a public package to make a tutorial appear complete.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/22-int4-inference-package/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Define a JSON Schema for the manifest, add per-file hashes and total-size checks, and
write a loader that rejects an incompatible runtime or base revision before allocating
GPU memory. Test truncation, swapped scale files, wrong group size, and rollback loading
as deliberate failure cases.

## Evidence boundary

**Evidence label:** [`pytorch-gpu`](../README.md#evidence-labels).

## References

- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [SafeTensors documentation](https://huggingface.co/docs/safetensors/index)
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
