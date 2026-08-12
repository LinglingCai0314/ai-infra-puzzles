"""Curated visual and step-by-step explanations for high-leverage lessons.

The lesson generators keep experiments, measurements, and evidence labels as the
source of truth.  This module adds a visual only when a relationship is easier to
understand as a data flow, state machine, or dependency graph.  Lessons that do not
benefit from a diagram intentionally remain prose-first.
"""

from __future__ import annotations

from typing import Any


def G(steps: list[tuple[str, str]], mermaid: str) -> dict[str, Any]:
    return {"steps": steps, "mermaid": mermaid.strip()}


CHAPTER_01_MAP = """
flowchart LR
  A["Formats and dispatch<br/>Lessons 01–07"] --> B["Quantization mechanics<br/>Lessons 08–13"]
  B --> C["Framework and runtime paths<br/>Lessons 14–21"]
  C --> D["Validation and operations<br/>Lessons 22–30"]
  A -. "shape and dtype evidence" .-> C
  B -. "error and calibration evidence" .-> D
""".strip()


CHAPTER_02_MAP = """
flowchart LR
  A["Define the delivery target"] --> B["Choose a pruning granularity"]
  B --> C["Prune with dependency constraints"]
  C --> D["Recover quality"]
  D --> E["Export a supported representation"]
  E --> F["Prove runtime and product value"]
  F -->|"gate fails"| C
  F -->|"gate passes"| G["Release with rollback evidence"]
""".strip()


CHAPTER_01_GUIDES: dict[int, dict[str, Any]] = {
    1: G(
        [
            ("Start with the stored object", "Identify which weights are packed to INT4 and which layers remain BF16."),
            ("Follow the runtime data path", "Track packed codes, group scales, BF16 activations, accumulation dtype, and any unpack or dequantization work."),
            ("Read four evidence axes", "Evaluate memory, operator identity, latency, and quality independently under one frozen workload."),
            ("Make a workload-specific decision", "Use INT4 for the tested path only when its capacity benefit and service gates justify the added kernel work."),
        ],
        """
flowchart LR
  W["model weights"] --> P["packed INT4 codes + scales"]
  A["BF16 activations"] --> K["weight-only linear kernel"]
  P --> K
  K --> O["BF16/FP32 accumulation and output"]
  O --> C["next layer and KV-cache path"]
  K --> E["memory + operator + latency + quality evidence"]
""",
    ),
    2: G(
        [
            ("Write the exact GEMM", "Record M, N, K, dtype, layout, and strides; a model-level precision label is not enough."),
            ("Estimate the useful work", "Use 2MKN to see how little the awkward shape changes the mathematical workload."),
            ("Check dispatch constraints", "Ask whether alignment, tile boundaries, and the Decode or Prefill shape family fit an efficient kernel."),
            ("Separate observations", "Timing establishes application behavior; a native trace is required to name the instruction path."),
        ],
        """
flowchart LR
  A["A: M × K"] --> G["Requested GEMM"]
  B["B: K × N"] --> G
  G --> Q{"dtype, layout, and<br/>shape fit the fast path?"}
  Q -->|"yes"| T["Efficient tiled kernel"]
  Q -->|"tail or fallback"| F["Lower-utilization path"]
  T --> E["Timing + native trace"]
  F --> E
""",
    ),
    3: G(
        [
            ("Enter autocast for forward", "Eligible operators may use a lower compute dtype while master parameters stay FP32."),
            ("Scale before backward", "Backward sees S times the loss, moving small gradients into a representable range."),
            ("Unscale and inspect", "Gradient clipping and finiteness checks must use unscaled gradients."),
            ("Step conditionally", "The optimizer updates only when gradients are finite; then the scale policy is updated."),
        ],
        """
flowchart TD
  A["FP32 parameters"] --> B["autocast forward + loss"]
  B --> C["scale(loss).backward()"]
  C --> D["unscale gradients"]
  D --> Q{"all gradients finite?"}
  Q -->|"yes"| E["optimizer.step()"]
  Q -->|"no"| F["skip parameter update"]
  E --> U["scaler.update()"]
  F --> U
""",
    ),
    5: G(
        [
            ("Locate the first bad stage", "Check forward activations, the scaled loss, scaled gradients, unscaled gradients, and parameters in that order."),
            ("Classify the symptom", "Inf indicates overflow; excessive zeros can indicate underflow even though every value is finite."),
            ("Apply the matching intervention", "Lower the scale for scaled-gradient overflow, raise it for underflow, or change the forward dtype for activation overflow."),
            ("Replay the same batch", "A diagnosis is useful only when the intervention removes the original first failure without creating a new one."),
        ],
        """
flowchart TD
  A["Non-finite loss or bad update"] --> B{"first bad tensor?"}
  B -->|"forward activation"| C["Change forward dtype,<br/>normalization, or input range"]
  B -->|"scaled gradient is Inf"| D["Lower scale and skip step"]
  B -->|"tiny gradient became zero"| E["Raise scale or use wider dtype"]
  B -->|"after optimizer"| F["Inspect unscale, clipping,<br/>optimizer state, and LR"]
  C --> R["Replay the same batch"]
  D --> R
  E --> R
  F --> R
""",
    ),
    7: G(
        [
            ("Separate persistent from temporary state", "Weights persist for the model lifetime; activations and workspace live for an operator or layer."),
            ("Account for context state", "KV cache grows with layers, batch, sequence length, heads, and head dimension."),
            ("Name storage and compute dtypes", "A tensor stored in INT4 may be dequantized into FP16/BF16 before or inside the kernel."),
            ("Optimize the dominant term", "Choose weight, activation, or KV quantization only after the workload-specific memory ledger identifies the bottleneck."),
        ],
        """
flowchart LR
  W["Weights<br/>persistent"] --> K["Layer kernel"]
  A["Activations<br/>short-lived"] --> K
  C["KV cache<br/>grows with context"] <--> K
  K --> O["Output activations"]
  W -. "storage dtype may differ<br/>from compute dtype" .-> K
""",
    ),
    8: G(
        [
            ("Choose a quantization range", "Derive scale and, for asymmetric quantization, zero point from the calibration range."),
            ("Map to integer codes", "Round and clamp each value into the available codebook."),
            ("Reconstruct for comparison", "Dequantize with the same metadata and measure error against the original tensor."),
            ("Change one granularity at a time", "Sweep group size while retaining metadata bytes so accuracy and effective storage remain comparable."),
        ],
        """
flowchart LR
  X["Floating tensor x"] --> S["Choose scale s<br/>and zero point z"]
  S --> Q["q = clamp(round(x / s) + z)"]
  Q --> P["Pack integer codes + metadata"]
  P --> D["x_hat = s(q - z)"]
  D --> E["Error and storage ledger"]
""",
    ),
    10: G(
        [
            ("Measure channel ranges", "Collect activation and weight maxima on calibration data for matching input channels."),
            ("Choose reciprocal scales", "Use alpha to decide how much range moves from each activation channel into its weight channel."),
            ("Verify floating equivalence", "Before rounding, confirm that (X/s)(W·s)^T still equals XW^T."),
            ("Quantize and validate", "Select alpha by held-out output or task quality, then verify a named W8A8 runtime path."),
        ],
        """
flowchart LR
  X["Activation X<br/>channel outliers"] --> XS["X' = X / s<br/>smaller activation range"]
  W["Weight W"] --> WS["W' = W · s<br/>absorbs migrated range"]
  XS --> M["Quantized linear path"]
  WS --> M
  M --> Y["Compare with Y = XW^T"]
  A["alpha sweep"] --> S["choose s per channel"]
  S --> XS
  S --> WS
""",
    ),
    11: G(
        [
            ("Freeze one layer and its calibration inputs", "GPTQ reconstructs layer outputs under the distribution represented by those inputs."),
            ("Estimate input curvature", "The activation Gram or Hessian approximation weights errors in directions that matter for the layer output."),
            ("Quantize a block of columns", "Choose codes, compute the residual introduced by rounding, and keep the quantized columns fixed."),
            ("Compensate the remaining columns", "Propagate the residual through the inverse-curvature approximation before moving to the next block."),
        ],
        """
flowchart LR
  A["Calibration activations"] --> H["Approximate curvature"]
  W["FP weight block"] --> Q["Quantize current columns"]
  H --> Q
  Q --> R["Compute reconstruction residual"]
  R --> C["Compensate unquantized columns"]
  C --> N{"more columns?"}
  N -->|"yes"| Q
  N -->|"no"| O["Validate layer output"]
""",
    ),
    12: G(
        [
            ("Observe activation-aware salience", "A small weight can matter when it multiplies a consistently large activation channel."),
            ("Search a scaling strength", "Rescale selected channels so important weights occupy more useful quantization levels."),
            ("Fold the scale into adjacent operations", "Preserve the floating function before quantization and avoid adding an unexplained runtime transform."),
            ("Judge the quantized output", "Use held-out layer or task error, not weight reconstruction error alone, to select the candidate."),
        ],
        """
flowchart LR
  X["Activation statistics"] --> I["Rank salient channels"]
  W["FP weights"] --> S["Search channel scaling"]
  I --> S
  S --> Q["Quantize scaled weights to INT4"]
  Q --> V["Validate held-out output"]
  V -->|"gate fails"| S
  V -->|"gate passes"| P["Pack for W4A16 backend"]
""",
    ),
    13: G(
        [
            ("Freeze the quantized base", "The NF4 base weights are storage for forward computation, not trainable optimizer parameters."),
            ("Dequantize for compute", "Blocks are reconstructed into the configured compute dtype as the layer executes."),
            ("Train only adapters", "LoRA matrices, their gradients, and their optimizer states form the main trainable parameter budget."),
            ("Keep a complete memory ledger", "Add quantized weights, scales, adapters, gradients, optimizer state, activations, and temporary workspace."),
        ],
        """
flowchart LR
  N["NF4 base weights<br/>frozen"] --> D["blockwise dequantize"]
  D --> B["base linear output"]
  X["input activation"] --> B
  X --> L["trainable LoRA path"]
  B --> Y["combined output"]
  L --> Y
  Y --> G["gradients only for adapters"]
""",
    ),
    16: G(
        [
            ("Express quantization in the graph", "Q/DQ nodes and their axes, block sizes, and scales must describe the intended representation."),
            ("Build for a named target", "TensorRT validates dtype, shape, hardware, and tactic constraints during engine construction."),
            ("Inspect the selected implementation", "A successful build does not prove that the intended INT4 tactic was selected."),
            ("Validate numerics and performance", "Compare the engine with the frozen baseline under the same inputs and timing protocol."),
        ],
        """
flowchart LR
  W["FP weight"] --> Q["Quantize / pack INT4 blocks"]
  Q --> DQ["Q/DQ graph semantics"]
  X["FP16/BF16 activation"] --> B["TensorRT builder"]
  DQ --> B
  B --> T{"supported INT4 tactic?"}
  T -->|"yes"| E["WoQ / INT4 engine"]
  T -->|"no"| F["fallback or build failure"]
  E --> V["Numerical + latency validation"]
""",
    ),
    17: G(
        [
            ("Pin the source model", "Record model revision, tokenizer, and baseline quality before conversion."),
            ("Calibrate or optimize", "ModelOpt produces scales, recipes, or a quantized checkpoint tied to calibration data and target format."),
            ("Build the runtime engine", "TensorRT-LLM consumes the supported artifact for a named GPU, shape range, and parallel configuration."),
            ("Carry provenance into serving", "The final package must preserve every revision and command needed to reproduce quality and performance."),
        ],
        """
flowchart LR
  M["Pinned model + tokenizer"] --> O["ModelOpt calibration / quantization"]
  C["Calibration corpus + recipe"] --> O
  O --> A["Quantized checkpoint + metadata"]
  A --> B["TensorRT-LLM build"]
  H["Target GPU + build config"] --> B
  B --> E["Engine"]
  E --> V["Quality, latency, memory gates"]
  V --> P["Versioned serving package"]
""",
    ),
    19: G(
        [
            ("Write the cache shape", "Account for layers, batch, sequence length, KV heads, head dimension, K and V, and bytes per element."),
            ("Choose a scale lifetime", "Per-token, per-head, or per-block scales trade metadata and kernel work against error."),
            ("Quantize live cache tensors", "Include scale bytes and any staging buffers rather than reporting the nominal element width only."),
            ("Test attention and service behavior", "Validate attention-output error, long-context quality, latency, and concurrency capacity."),
        ],
        """
flowchart LR
  T["New token"] --> K["K projection"]
  T --> V["V projection"]
  K --> QK["quantize + store K"]
  V --> QV["quantize + store V"]
  QK --> C["growing KV cache"]
  QV --> C
  C --> D["dequantize or fused attention read"]
  D --> A["attention output + quality check"]
""",
    ),
    24: G(
        [
            ("Freeze the service workload", "Specify prompt/output lengths, arrival pattern, concurrency, batching policy, and cache state."),
            ("Measure latency as phases", "Separate queueing, Prefill, inter-token Decode, and total request latency."),
            ("Report distributions and throughput together", "A throughput gain is incomplete when p95 or p99 violates the service objective."),
            ("Attribute the result", "Pair end-to-end metrics with memory and operator evidence so a bottleneck shift is visible."),
        ],
        """
flowchart LR
  W["Frozen request distribution"] --> Q["Queueing"]
  Q --> P["Prefill"]
  P --> D["Decode loop"]
  D --> O["Completed requests"]
  Q --> M["p50 / p95 / p99 ledger"]
  P --> M
  D --> M
  O --> T["throughput + capacity"]
  T --> G{"all SLO and quality gates pass?"}
""",
    ),
    27: G(
        [
            ("Bind immutable artifacts", "Model, tokenizer, recipe, runtime, container, and GPU compatibility form one release unit."),
            ("Pass offline gates", "Quality, numerical, load, and performance checks run before any traffic exposure."),
            ("Increase exposure in stages", "Shadow and canary stages consume predeclared health and quality thresholds."),
            ("Make rollback executable", "Every stage points to a load-tested baseline and has an automatic or operator-triggered stop condition."),
        ],
        """
stateDiagram-v2
  [*] --> Offline
  Offline --> LoadSmoke: quality and performance pass
  Offline --> Rollback: gate fails
  LoadSmoke --> Shadow: load and compatibility pass
  LoadSmoke --> Rollback: gate fails
  Shadow --> Canary: shadow checks pass
  Shadow --> Rollback: drift or error
  Canary --> Rollout: SLO and quality pass
  Canary --> Rollback: threshold breached
  Rollout --> Rollback: production regression
""",
    ),
    30: G(
        [
            ("Reject impossible capacity plans early", "Estimate weight, KV-cache, runtime, and concurrency memory before choosing a quantizer."),
            ("Build a representative smaller proof", "Validate the exact format, backend, quality suite, and workload on a model that fits the available hardware."),
            ("Create the full-model artifact", "Quantize the pinned 70B checkpoint with reproducible calibration and packaging metadata."),
            ("Promote only with full-system evidence", "Require multi-GPU engine, load, quality, long-context, throughput, and rollback results from the target topology."),
        ],
        """
flowchart TD
  R["70B requirements + target SLO"] --> C["Capacity and topology model"]
  C --> Q{"fits target hardware?"}
  Q -->|"no"| X["revise format, parallelism,<br/>context, or concurrency"]
  X --> C
  Q -->|"yes"| P["small-model backend proof"]
  P --> A["quantize pinned 70B artifact"]
  A --> E["build full multi-GPU engine"]
  E --> G["quality + load + service gates"]
  G -->|"fail"| B["rollback or mixed-bit fallback"]
  G -->|"pass"| S["staged release"]
""",
    ),
}


CHAPTER_02_GUIDES: dict[int, dict[str, Any]] = {
    1: G(
        [
            ("Name the product objective", "Choose package size, memory, latency, throughput, energy, or cost and attach a measurable gate."),
            ("Locate the structural change", "Distinguish zeros in a tensor from a changed tensor shape or stored representation."),
            ("Locate the execution change", "Confirm whether the runtime dispatched a smaller dense operator or a supported sparse operator."),
            ("Accept on the original objective", "A candidate succeeds only when quality and the named deployment metric both pass."),
        ],
        """
flowchart LR
  V["Value state<br/>which entries are zero?"] --> R["Representation state<br/>what is stored?"]
  R --> S["Shape state<br/>which axes changed?"]
  S --> E["Execution state<br/>which operator ran?"]
  E --> P["Product metric<br/>did the target improve?"]
""",
    ),
    2: G(
        [
            ("Hold the zero budget fixed", "Compare layouts at the same global sparsity so granularity is the independent variable."),
            ("Check the local contract", "Block and N:M layouts require local grouping rules that a global percentage cannot express."),
            ("Check physical shape", "Channel removal changes dimensions and can reuse ordinary dense kernels at a smaller size."),
            ("Match the target runtime", "Choose only among formats with an implemented loader, operator, and supported shapes on the deployment stack."),
        ],
        """
flowchart TD
  Z["Same 50% zero budget"] --> U["Unstructured zeros<br/>same shape"]
  Z --> B["Block sparsity<br/>same shape + block metadata"]
  Z --> N["2:4 sparsity<br/>local pattern contract"]
  Z --> C["Channel pruning<br/>smaller physical shape"]
  U --> K["Runtime support decides value"]
  B --> K
  N --> K
  C --> K
""",
    ),
    4: G(
        [
            ("Freeze the dense checkpoint", "All schedules start from identical weights, data order, and optimizer conditions."),
            ("Apply a declared support change", "Record which weights or structures are removed at each event."),
            ("Recover under the constraint", "Reapply masks or structural constraints after updates so pruned values cannot silently regrow."),
            ("Re-evaluate and decide", "Check quality, sparsity, export, and runtime gates before continuing or rolling back."),
        ],
        """
flowchart LR
  D["Frozen dense checkpoint"] --> P["Prune to next target"]
  P --> R["Recovery training<br/>with mask enforced"]
  R --> E["Held-out evaluation"]
  E --> Q{"quality and sparsity pass?"}
  Q -->|"next stage"| P
  Q -->|"final pass"| X["export + runtime test"]
  Q -->|"fail"| B["rollback or revise schedule"]
""",
    ),
    5: G(
        [
            ("Apply the mask", "PyTorch stores the original parameter and a mask, then computes their product through a hook."),
            ("Audit logical sparsity", "Count zeros and verify forward behavior without making a storage claim."),
            ("Remove the reparameterization", "Materialize the masked dense tensor and confirm state_dict keys and load behavior."),
            ("Choose an actual storage format", "Compression, CSR, and backend-specific packing answer different deployment questions."),
        ],
        """
flowchart LR
  W["dense weight"] --> A["apply pruning"]
  A --> P["weight_orig + weight_mask"]
  P --> F["forward uses weight_orig × mask"]
  P --> R["prune.remove()"]
  R --> M["materialized dense tensor<br/>containing zeros"]
  M --> S["optional compression or<br/>explicit sparse encoding"]
""",
    ),
    7: G(
        [
            ("Rank output channels", "A filter score selects complete output channels, not isolated scalar weights."),
            ("Slice the producer", "Remove matching Conv output filters and their bias entries."),
            ("Propagate the index set", "Slice normalization state and every consuming layer's corresponding input channels."),
            ("Rebuild and benchmark", "Verify graph shapes and outputs before comparing the physically narrower dense convolution."),
        ],
        """
flowchart LR
  C1["Conv: remove output channels I"] --> B["BatchNorm: remove state I"]
  B --> A["Activation"]
  A --> C2["Next Conv: remove input channels I"]
  C2 --> O["Smaller dense graph"]
  I["one retained-index ledger"] -.-> C1
  I -.-> B
  I -.-> C2
""",
    ),
    9: G(
        [
            ("Choose a root pruning operation", "Start from one producer and a concrete retained-channel index set."),
            ("Follow merge semantics", "Addition requires aligned branch outputs; concatenation requires offset-aware index mapping."),
            ("Update coupled state", "Propagate through Conv, BatchNorm, residual branches, and downstream consumers as one group."),
            ("Reject invalid groups before mutation", "Check dimensionality, divisibility, and over-pruning constraints, then run a forward shape audit."),
        ],
        """
flowchart LR
  X["input"] --> F["branch f"]
  X --> G["branch g"]
  F --> A["Add: shapes must match"]
  G --> A
  A --> C["consumer"]
  I["remove channel set I"] -. "propagate" .-> F
  I -. "same output indices" .-> G
  I -. "remove input indices" .-> C
""",
    ),
    11: G(
        [
            ("Define the pruning window", "The begin step, end step, update frequency, and final target are part of the experiment."),
            ("Compute each intermediate target", "A polynomial schedule controls how large each support shock is, not only the final sparsity."),
            ("Update and enforce the mask", "Re-rank only at declared events and prevent later optimizer steps from regrowing removed weights."),
            ("Read quality as a trajectory", "Compare accuracy immediately before pruning, immediately after, and after recovery—not only at the end."),
        ],
        """
flowchart LR
  S0["dense start"] --> P1["small pruning event"]
  P1 --> R1["recover"]
  R1 --> P2["larger target"]
  P2 --> R2["recover"]
  R2 --> PF["final sparsity"]
  PF --> RF["final recovery + gate"]
  G["polynomial schedule"] -.-> P1
  G -.-> P2
  G -.-> PF
""",
    ),
    13: G(
        [
            ("Group along the contracted axis", "Reshape the supported weight dimension into consecutive groups of four."),
            ("Keep exactly two values", "Top-2 magnitude selection creates 50% global sparsity and 100% local 2:4 compliance."),
            ("Convert to the backend representation", "A compliant dense tensor is not yet a cuSPARSELt or TensorRT sparse operand."),
            ("Prove the selected tactic", "Validate output, capture the sparse operator or tactic, and compare with a matched dense baseline."),
        ],
        """
flowchart LR
  W["dense group<br/>w0 w1 w2 w3"] --> K["keep top two magnitudes"]
  K --> M["2:4 values<br/>two nonzeros + two zeros"]
  M --> C{"backend conversion<br/>supported?"}
  C -->|"yes"| S["compressed sparse operand"]
  C -->|"no"| D["ordinary dense storage/path"]
  S --> T["sparse tactic + matched benchmark"]
""",
    ),
    14: G(
        [
            ("Inspect the module before pruning", "Record parameter names, buffers, hooks, and the exact checkpoint identity."),
            ("Apply a pruning method", "The API installs weight_orig, weight_mask, and a forward pre-hook."),
            ("Train or evaluate with the mask active", "Audit optimizer behavior and ensure the effective weight preserves the intended zeros."),
            ("Finalize deliberately", "Use prune.remove when a materialized dense zero tensor is desired, then separately test save, load, export, and rollback."),
        ],
        """
stateDiagram-v2
  [*] --> Dense
  Dense --> Reparameterized: apply pruning
  Reparameterized --> Reparameterized: train/evaluate with mask
  Reparameterized --> Materialized: prune.remove()
  Materialized --> Exported: save and export tests pass
  Reparameterized --> Dense: restore dense checkpoint
  Materialized --> Dense: rollback checkpoint
""",
    ),
    18: G(
        [
            ("Validate the source weights", "Prove 2:4 compliance on the correct axis before engine construction."),
            ("Export without destroying the contract", "Retain the supported dtype, shape, and weight layout through ONNX or the builder input."),
            ("Inspect build evidence", "Capture TensorRT and Polygraphy logs, tactic selection, and any dense fallback reason."),
            ("Benchmark the built engine", "Compare numerical outputs and latency against an engine built from the frozen dense baseline."),
        ],
        """
flowchart LR
  W["2:4-compliant weights"] --> X["exported graph"]
  X --> B["TensorRT build with sparsity enabled"]
  B --> Q{"sparse tactic selected?"}
  Q -->|"yes"| E["sparse engine"]
  Q -->|"no"| D["dense tactic or build diagnostic"]
  E --> P["Polygraphy correctness check"]
  P --> M["matched latency benchmark"]
""",
    ),
    19: G(
        [
            ("Build an index ledger", "For every removed channel, record the affected weight, bias, normalization, merge, and consumer dimensions."),
            ("Export the structural candidate", "Use representative inputs and explicit dynamic-axis rules instead of treating export success as validation."),
            ("Run graph checks in order", "Apply ONNX checker, shape inference, and a runtime execution with known inputs."),
            ("Compare semantics", "Match output names, shapes, and numerical values with the framework candidate before accepting the graph."),
        ],
        """
flowchart LR
  P["physically pruned model"] --> E["ONNX export"]
  E --> C["onnx.checker"]
  C --> S["shape inference"]
  S --> R["ONNX Runtime execution"]
  R --> V["shape + numerical comparison"]
  V -->|"fail"| L["repair index, bias,<br/>merge, or postprocess ledger"]
  L --> E
""",
    ),
    22: G(
        [
            ("Choose the structural unit", "Attention heads, hidden channels, FFN neurons, and full layers change different dimensions and interfaces."),
            ("Propagate coupled dimensions", "Head removal affects Q/K/V and output projection slices; FFN removal couples up and down projections."),
            ("Rebuild the executable graph", "Config fields, cache shapes, residual dimensions, and exported metadata must agree with the new structure."),
            ("Measure the remaining bottleneck", "A smaller attention block may not improve end-to-end latency when FFN, memory traffic, or launch overhead dominates."),
        ],
        """
flowchart TD
  T["Transformer block"] --> A["Attention heads"]
  T --> F["FFN neurons"]
  T --> L["whole-layer depth"]
  A --> QA["slice Q/K/V + output projection"]
  F --> QF["slice up/gate + down projection"]
  L --> QL["update layer list + cache/config"]
  QA --> V["rebuild, validate, benchmark"]
  QF --> V
  QL --> V
""",
    ),
    23: G(
        [
            ("Freeze a representative calibration set", "Both methods depend on the activations seen during layer-wise pruning."),
            ("Compute method-specific scores", "Wanda combines weight magnitude and activation norms; SparseGPT uses a second-order reconstruction approximation."),
            ("Prune one layer and propagate activations", "Later layers must receive outputs from the already-pruned prefix."),
            ("Evaluate beyond perplexity", "Compare sparsity, perplexity, zero-shot tasks, runtime representation, and actual inference performance separately."),
        ],
        """
flowchart LR
  C["calibration activations"] --> W["Wanda score<br/>|weight| × activation norm"]
  C --> S["SparseGPT score<br/>second-order reconstruction"]
  M["current layer weights"] --> W
  M --> S
  W --> P1["pruned candidate A"]
  S --> P2["pruned candidate B"]
  P1 --> E["perplexity + task + runtime gates"]
  P2 --> E
""",
    ),
    25: G(
        [
            ("Freeze benchmark identity", "Pin model, runtime, hardware, input shapes, batch/concurrency, threads, warm-up, and sampling window."),
            ("Prove operator identity", "Capture graph or tactic evidence showing that the intended sparse or smaller operator actually ran."),
            ("Measure distributions", "Retain repeated samples and report p50, p95, p99, throughput, memory, and initialization separately."),
            ("Require a margin above noise", "Accept only when the confidence interval or repeated-run spread is smaller than the claimed improvement."),
        ],
        """
flowchart LR
  B["frozen dense baseline"] --> H["same harness"]
  P["pruned candidate"] --> H
  H --> W["warm-up"]
  W --> S["repeated synchronized samples"]
  S --> D["latency distribution + throughput"]
  S --> O["operator/tactic trace"]
  D --> G{"quality, tail latency,<br/>memory, and speed gates pass?"}
  O --> G
""",
    ),
    26: G(
        [
            ("Compare the same evaluation slices", "Overall accuracy can hide regressions in rare classes, long inputs, or safety-critical cohorts."),
            ("Localize the error", "Use confusion matrices, per-slice deltas, and representative failures to identify where pruning changed behavior."),
            ("Apply one recovery intervention", "Fine-tuning, distillation, a lower pruning target, or selective restoration should be tested separately."),
            ("Rollback on predeclared thresholds", "Keep the dense checkpoint and the last accepted sparse checkpoint loadable throughout recovery."),
        ],
        """
flowchart TD
  P["pruned candidate"] --> E["overall + slice evaluation"]
  E --> Q{"all critical slices pass?"}
  Q -->|"yes"| R["runtime and release gates"]
  Q -->|"no"| L["localize failing structures/slices"]
  L --> I["one recovery intervention"]
  I --> E
  I -->|"budget exhausted"| B["rollback to accepted checkpoint"]
""",
    ),
    28: G(
        [
            ("Write one target card per platform", "Edge and server deployments have different workloads, runtimes, memory limits, energy constraints, and cost objectives."),
            ("Select only supported structures", "A format useful to TensorRT on a GPU may have no benefit in TFLite or a mobile CPU runtime."),
            ("Benchmark on each real path", "Measure cold start and energy on edge; measure concurrency, tail latency, and capacity on servers."),
            ("Allow different winners", "Do not force one checkpoint to win both matrices when platform-specific candidates meet their objectives more honestly."),
        ],
        """
flowchart TD
  M["same dense model"] --> E["edge target card"]
  M --> S["server target card"]
  E --> EP["package size, cold start,<br/>RAM, energy, device latency"]
  S --> SP["throughput, p95/p99,<br/>GPU memory, concurrency cost"]
  EP --> EC["edge-specific pruning candidate"]
  SP --> SC["server-specific pruning candidate"]
  EC --> D["platform decision matrix"]
  SC --> D
""",
    ),
    3: G(
        [
            ("Freeze the graph before measuring", "Record every layer shape, dtype, parameter count, and analytical operation count before changing the model."),
            ("Define workload points", "Batch size, input shape, sequence length, and concurrency belong to the baseline identity rather than to a footnote."),
            ("Measure a distribution", "Warm the stack, synchronize device work, retain repeated latency samples, and reset the memory window."),
            ("Keep metric meanings separate", "Parameters and FLOPs describe structure; latency, throughput, and peak memory describe one execution path."),
        ],
        """
flowchart LR
  M["frozen dense model"] --> S["shape + parameter + FLOP ledger"]
  W["frozen workload grid"] --> H["reproducible timing harness"]
  M --> H
  H --> L["latency distribution"]
  H --> T["throughput"]
  H --> P["peak memory"]
  S --> B["baseline report"]
  L --> B
  T --> B
  P --> B
""",
    ),
    6: G(
        [
            ("Choose a global target", "The total zero budget is a constraint on the whole model, not a requirement that every layer reach the same rate."),
            ("Normalize comparable scores", "Collect importance values under one calibration protocol and account for layer scale before ranking globally."),
            ("Protect constrained layers", "Apply minimum width, divisibility, first/last-layer, residual, and hardware-alignment rules before allocating the rest."),
            ("Validate the allocation", "Compare the resulting per-layer budget with uniform pruning on quality, physical structure, and the target runtime."),
        ],
        """
flowchart TD
  G["global sparsity target"] --> S["collect normalized scores per layer"]
  C["layer constraints<br/>minimum width, alignment, topology"] --> A["allocate removable budget"]
  S --> A
  A --> L1["sensitive layer: low sparsity"]
  A --> L2["redundant layer: higher sparsity"]
  A --> L3["protected layer: no pruning"]
  L1 --> V["quality + runtime validation"]
  L2 --> V
  L3 --> V
""",
    ),
    8: G(
        [
            ("Train channel gates", "BatchNorm gamma values receive a sparsity penalty while the network still trains with its original physical shape."),
            ("Rank channels after convergence", "Use the learned gate magnitudes with minimum-width and dependency constraints, not an arbitrary mid-training snapshot."),
            ("Rebuild the narrow network", "Slice convolution weights, BatchNorm state, residual partners, and consumers using one retained-index ledger."),
            ("Recover and compare", "Fine-tune the physical candidate, then test quality and dense-kernel latency at the new shapes."),
        ],
        """
flowchart LR
  C["Conv output channels"] --> B["BatchNorm gamma gates"]
  R["task loss + lambda × |gamma|"] --> B
  B --> K["select retained channels"]
  K --> N["rebuild physically narrow graph"]
  N --> F["recovery fine-tuning"]
  F --> V["quality + latency gates"]
""",
    ),
    10: G(
        [
            ("Capture the relevant activation", "Retain the hidden channel h and its gradient under a representative calibration loss."),
            ("Compute the first-order score", "Aggregate the magnitude of h times dL/dh for each channel, with the sign policy stated explicitly."),
            ("Validate the ranking", "Ablate channels one at a time on held-out data and compare predicted importance with actual loss increase."),
            ("Recheck after joint pruning", "Independent first-order scores can fail when several interacting channels are removed together."),
        ],
        """
flowchart LR
  H["hidden activation h"] --> S["Taylor score |h × dL/dh|"]
  G["loss gradient dL/dh"] --> S
  S --> R["rank channels"]
  R --> A["held-out one-channel ablations"]
  A --> C["ranking correlation"]
  C --> J["joint-pruning validation"]
""",
    ),
    12: G(
        [
            ("Attach a learnable gate", "Place one gate on the structural unit to be selected, such as a channel, head, or block."),
            ("Optimize task and sparsity objectives together", "Track the task loss, regularization pressure, and gate distribution rather than only the final zero count."),
            ("Freeze a discrete structure", "Choose and record a threshold, then convert soft gates into an explicit retained-index set."),
            ("Remove the gated structure physically", "Rebuild and recover the model so the runtime sees smaller tensors instead of multiplying by near-zero gates."),
        ],
        """
flowchart LR
  X["activation"] --> G["learnable structural gate"]
  T["task loss"] --> O["joint optimization"]
  R["sparsity regularizer"] --> O
  O --> G
  G --> H["threshold and freeze indices"]
  H --> P["physical graph surgery"]
  P --> V["recover + validate"]
""",
    ),
    15: G(
        [
            ("Trace with representative inputs", "Dependency discovery must see the operators, merges, and shapes used by the intended execution path."),
            ("Request one root pruning action", "Choose a layer, pruning function, and concrete index set rather than editing tensors directly."),
            ("Inspect the generated group", "Review every coupled operation and reject a group that violates minimum channels, grouping, or model interfaces."),
            ("Execute and validate the mutation", "Run forward, parameter, shape, export, and quality checks before treating the DepGraph result as usable."),
        ],
        """
flowchart LR
  M["model + example inputs"] --> D["DepGraph trace"]
  R["root prune request"] --> G["dependency group"]
  D --> G
  G --> C{"group constraints pass?"}
  C -->|"no"| X["reject or reduce indices"]
  C -->|"yes"| P["execute group pruning"]
  P --> V["forward + shape + quality checks"]
""",
    ),
    16: G(
        [
            ("Wrap the model before training", "The pruning wrapper owns masks and schedule state; it is not equivalent to a permanently smaller Keras layer."),
            ("Advance the pruning step", "Callbacks or explicit updates must keep the schedule synchronized with optimizer steps."),
            ("Strip training-only wrappers", "After training, materialize the masked weights and remove wrapper state before export."),
            ("Verify the deployment artifact", "Load the stripped model, convert to the target format, and measure compressed size and runtime behavior separately."),
        ],
        """
stateDiagram-v2
  [*] --> DenseKeras
  DenseKeras --> Wrapped: prune_low_magnitude
  Wrapped --> Scheduled: training + pruning-step updates
  Scheduled --> Stripped: strip_pruning
  Stripped --> Exported: SavedModel / TFLite conversion
  Exported --> Verified: load, size, quality, runtime checks
""",
    ),
    17: G(
        [
            ("Choose the CPU runtime first", "OpenVINO, NNCF, and Intel Neural Compressor support different models, sparsity patterns, and optimization workflows."),
            ("Optimize with representative data", "Calibration or accuracy-aware tuning must use the same preprocessing and task contract as the baseline."),
            ("Inspect the exported representation", "Verify IR or serialized size, shapes, precision, and whether the runtime preserved a useful sparse pattern."),
            ("Benchmark on the target CPU", "Pin threads, cores, batch, warm-up, and latency mode; a GPU-side zero pattern is not CPU performance evidence."),
        ],
        """
flowchart LR
  M["dense framework model"] --> O["NNCF / INC optimization"]
  C["calibration + accuracy criteria"] --> O
  O --> I["OpenVINO IR or runtime artifact"]
  I --> Q["representation and shape audit"]
  Q --> B["target-CPU benchmark"]
  B --> G{"quality, latency,<br/>size gates pass?"}
""",
    ),
    20: G(
        [
            ("Select channels per residual stage", "A ResNet channel decision must respect main-path and shortcut output compatibility at every addition."),
            ("Propagate indices through the block", "Slice Conv, BatchNorm, projection shortcuts, and downstream input channels as a coupled transformation."),
            ("Rebuild from the retained-index ledger", "Update module dimensions explicitly so parameter and FLOP reductions are physical and inspectable."),
            ("Recover and benchmark end to end", "Fine-tune from the dense checkpoint, evaluate accuracy, and time the target image workload rather than one convolution only."),
        ],
        """
flowchart LR
  X["stage input"] --> M["main Conv-BN path"]
  X --> S["identity or projection shortcut"]
  M --> A["residual add"]
  S --> A
  I["shared retained-channel indices"] -.-> M
  I -.-> S
  A --> N["next physically narrow stage"]
  N --> V["fine-tune + accuracy + latency"]
""",
    ),
    21: G(
        [
            ("Map the whole task graph", "Detection and segmentation couple backbone features to neck scales, heads, anchors, masks, and post-processing dimensions."),
            ("Protect task-sensitive interfaces", "Keep feature pyramid channel agreements, spatial resolutions, class outputs, and mask geometry valid."),
            ("Evaluate task slices", "Measure small, medium, and large objects or class and boundary slices—not only an aggregate score."),
            ("Include pre- and post-processing", "The deployment gate uses end-to-end latency because NMS, resizing, and mask decoding may dominate after pruning."),
        ],
        """
flowchart LR
  I["input image"] --> B["pruned backbone"]
  B --> P["multi-scale neck / FPN"]
  P --> H1["classification + box heads"]
  P --> H2["mask / segmentation head"]
  H1 --> O["post-processing"]
  H2 --> O
  O --> E["slice quality + end-to-end latency"]
""",
    ),
    24: G(
        [
            ("Name the role of each method", "Distillation transfers behavior, pruning removes capacity, and quantization changes numerical representation."),
            ("Choose an order from constraints", "If retraining is available, prune before final quantization; if a teacher is fixed, distillation can accompany recovery."),
            ("Freeze an intermediate checkpoint", "Evaluate quality after every transformation so the source of a regression remains localizable."),
            ("Compare complete pipelines", "Hold total training budget, final format, runtime, and evaluation suite fixed when testing alternative orders."),
        ],
        """
flowchart TD
  D["dense teacher / baseline"] --> P["prune student structure"]
  D -. "teacher targets" .-> R["recovery + distillation"]
  P --> R
  R --> Q["final quantization"]
  Q --> V["quality + runtime gates"]
  D --> A["alternative order"]
  A --> V2["same budget and final-format gates"]
  V --> C["compare complete pipelines"]
  V2 --> C
""",
    ),
    27: G(
        [
            ("Create an immutable run identity", "Bind code commit, model revision, data split, environment, seed, and configuration before execution."),
            ("Record the pruning trajectory", "Store per-stage sparsity, masks or retained indices, recovery checkpoints, and evaluation slices."),
            ("Attach deployment evidence", "Keep export logs, runtime versions, operator traces, raw timing samples, and memory measurements with the same run."),
            ("Reproduce before promotion", "A second run should rebuild the same candidate and reach tolerances defined in the manifest, not merely produce a similar headline metric."),
        ],
        """
flowchart LR
  I["commit + model + data + env + seed"] --> R["immutable run manifest"]
  R --> P["pruning and recovery stages"]
  P --> A["checkpoints + masks + metrics"]
  A --> E["export + runtime evidence"]
  E --> C["content hashes and final decision"]
  C --> X["independent reproduction run"]
  X --> G{"manifest tolerances pass?"}
""",
    ),
}


def render_guide(guide: dict[str, Any] | None) -> str:
    """Render one optional guide as Markdown."""

    if not guide:
        return ""
    steps = "\n".join(
        f"{index}. **{title}.** {explanation}"
        for index, (title, explanation) in enumerate(guide["steps"], 1)
    )
    return f"""### Mechanism at a glance

```mermaid
{guide['mermaid']}
```

### Walk it step by step

{steps}
"""
