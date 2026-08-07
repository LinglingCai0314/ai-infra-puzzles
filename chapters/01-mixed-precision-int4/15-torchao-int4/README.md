# Lesson 15 — TorchAO INT4 Weight-Only Quantization

> **Puzzle:** Can a PyTorch-native INT4 conversion reduce storage and still lose on latency?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A PyTorch-native quantization API still depends on a precise package, ABI, hardware, and
kernel combination. Conversion success, packed storage, numerical agreement, and latency
are separate gates. Preserving a failed compatibility attempt is more useful than
silently substituting a fake quantizer and calling it TorchAO.

## Predict before reading the result

1. Predict the evidence sequence required before comparing TorchAO INT4 latency with BF16.
2. Decide whether an installed `torchao` package is enough to claim native execution.
3. Explain how an ABI or auxiliary-kernel dependency can block an otherwise supported GPU.

## 1. Start from concrete tensors and state

TorchAO conversion replaces or wraps eligible `Linear` weights with a packed tensor
subclass/configuration. The Python module, packed storage, and selected matmul kernel
are three inspectable layers.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | TorchAO replaces eligible modules according to a quantization configuration. |
| 2 | Packed storage and executed operator evidence are distinct from a module label. |
| 3 | Small batch and shape-specific overhead can outweigh lower memory traffic. |

## 2. Derive the mechanism

INT4 weight-only compute conceptually reads packed codes and group scales while BF16
activations enter the linear operation. Modern TorchAO versions may choose among packing
formats and external kernel libraries such as MSLK.

A weight-only conversion replaces eligible linear modules with a representation that
stores packed low-bit weights and dispatches a compatible operator for floating-point
inputs. The theoretical bandwidth reduction appears only if conversion succeeds, packing
is retained, and the runtime avoids materializing full dequantized weights. Package
metadata alone proves none of those conditions.

The compatibility chain is `Python package → PyTorch ABI → auxiliary kernel package →
GPU architecture → quantization config → converted module → executed operator`. A break
near the beginning prevents meaningful memory, error, or latency comparison farther down
the chain.

## 3. Translate the theory into an experiment

**Experiment:** Convert a BF16 linear layer with TorchAO INT4, record the resulting module type, compare output error, and time both paths.

| Experimental role | Frozen definition |
|---|---|
| Baseline | BF16 linear module, reserved as the fallback path |
| Candidate | TorchAO `Int4WeightOnlyConfig` conversion on the same CUDA stack |
| Held constant | PyTorch 2.12/CUDA 13 environment, RTX 5090, layer/config intent |
| Measurements | package presence, conversion status, exact exception class/message; downstream metrics only on success |
| Evidence label | `compatibility-probe` |

The notebook attempts the documented native configuration inside an explicit
compatibility boundary and records the exact failure class when the path cannot execute.

### Code walk-through

The notebook imports TorchAO, constructs the intended conversion, and catches the exact
failure rather than replacing the candidate. The JSON result records both
`torchao_installed=true` and `conversion=failed`, which distinguishes package discovery
from backend readiness.

Because conversion stopped before a quantized module existed, the notebook correctly
omits storage, output-error, operator, and latency numbers for the candidate.
Fabricating those from a reference quantizer would answer a different lesson.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| TorchAO installed | yes |
| Conversion status | failed |
| Failure type | ImportError |
| Failure message | Requires mslk >= 1.0.0 |

### What the numbers mean

TorchAO was present, but conversion raised `ImportError: Requires mslk >= 1.0.0`. The
native INT4 operator did not execute, so the evidence label is `compatibility-probe`,
not `native-backend`. This negative result establishes the exact boundary of the saved
environment and a reproducible next action.

Lesson 01 used a full-model TorchAO path that did execute under its tested
configuration. The contrast is valuable: backend support can depend on API/configuration
and dependency versions even on the same GPU, so results must stay attached to their
exact path.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Treat TorchAO INT4 as a measured backend path, not a universal performance property of four-bit weights.

### Acceptance and rollback gate

Require successful import/conversion, quantized tensor/module identity, storage
accounting, operator evidence, output error, and repeated latency. Preserve dependency
failure rather than falling back silently.

### How this conclusion can fail

The worst response would be to catch the error, run a hand-written fake quantizer, and
leave the heading 'TorchAO benchmark'. Another failure is installing arbitrary nightly
wheels until import succeeds without checking ABI compatibility or whether the
environment was altered for other lessons.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map BF16 linear module, reserved as the fallback path
and TorchAO `Int4WeightOnlyConfig` conversion on the same CUDA stack back to the
derivation. Verify the printed environment, then check that PyTorch 2.12/CUDA 13
environment, RTX 5090, layer/config intent stayed fixed. Read package presence,
conversion status, exact exception class/message; downstream metrics only on success
before applying the acceptance gate; the artifact-writing cell retains the complete
structured result from the recorded run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/15-torchao-int4/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Create an isolated environment using the TorchAO compatibility matrix, install a
matching MSLK/PyTorch build, and rerun conversion. Only after success should the lab add
module type, packed storage, output error, operator trace, warm-up, repeated latency,
and a comparison with the BF16 baseline.

## Evidence boundary

The named optional backend did not complete a native run in this environment. Package
and failure evidence are retained; service or kernel performance is not inferred.

The checked-in observation belongs to Lesson 15's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
- [TorchAO quantization API](https://docs.pytorch.org/ao/stable/api_reference/index.html)
- [TorchAO repository](https://github.com/pytorch/ao)
