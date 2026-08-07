"""Hand-written delivery content for the Chapter 01 mechanism labs.

The experiments and artifacts remain the source of measured values.  This
module supplies the lesson-specific narrative needed to turn each compact lab
into a tutorial: motivation, derivation, controlled comparison, interpretation,
failure analysis, and follow-up exercises.
"""

from __future__ import annotations

from typing import Any


DELIVERY: dict[int, dict[str, Any]] = {
    2: {
        "hook": "A peak-TFLOPS table describes a capability of the chip, not the path selected for every matrix multiplication. In an LLM, the same nominal BF16 operation can arrive with different M, N, and K dimensions, strides, transpositions, and batch sizes. Those details determine whether useful work fills the matrix-multiply tiles or whether edge handling, memory traffic, and launch overhead dominate.",
        "checks": [
            "Predict whether BF16 will beat FP32 for both shapes, then predict which shape will lose more efficiency.",
            "State what timing can prove and what extra trace would be required before naming a Tensor Core instruction.",
            "Choose the shape information that must be preserved for another reader to reproduce the result.",
        ],
        "derivation": "For `C[M,N] = A[M,K] @ B[K,N]`, the leading operation count is `2MKN`. That number is only the numerator of the performance story. A first roofline estimate divides it by bytes moved; a dispatch estimate also asks whether M, N, K, layout, alignment, and dtype fit a library kernel's tiling rules. When `N=2055`, the mathematical work increases by only about 0.34% relative to `N=2048`, yet the physical implementation may need a tail tile or a different kernel. A large timing discontinuity is therefore evidence about shape sensitivity, not proof of one particular instruction.\n\nThis distinction matters in attention and MLP layers because their matrices are not interchangeable. Prefill creates large M dimensions, while Decode often presents GEMV-like or very small-M work. A kernel that is excellent for one phase can leave Tensor Cores under-filled in another. The useful unit of reasoning is consequently a shape family plus an operator trace, not the model's advertised precision.",
        "baseline": "FP32 GEMM for the exact aligned and awkward shapes",
        "candidate": "BF16 GEMM for the same tensors and timing protocol",
        "controlled": "GPU, M and K, random distribution, warm-up, repetitions, CUDA-event timing",
        "metrics": "median and p90 latency for each dtype/shape pair",
        "code_walk": "The notebook allocates each shape once, warms the operation four times, and records twelve CUDA-event samples. Synchronization happens inside the timing helper so host launch latency is not mistaken for completed GPU work. The aligned and awkward cases differ only in N; this keeps the comparison narrow enough to attribute a timing change to shape and dispatch behavior.\n\nThe code deliberately does not parse native kernel names. PyTorch-level timing tells us what the application observed, while Nsight Systems or Nsight Compute would be the next evidence layer for `mma`/Tensor Core utilization, tile occupancy, memory throughput, and tail effects.",
        "result_reading": "On the checked-in RTX 5090 run, aligned BF16 took 0.087632 ms versus 0.262096 ms for FP32, a 2.99x ratio. Changing only N from 2048 to 2055 raised BF16 latency to 0.176048 ms—about 2.01x the aligned BF16 time—even though the arithmetic count barely changed. FP32 also slowed, but by a smaller 1.22x ratio.\n\nThe correct conclusion is not that `2055` is universally bad or that one named Tensor Core kernel was missed. It is that dtype speedups are conditional on shape, and that an awkward boundary can erase a large fraction of the expected benefit. Native dispatch identity remains an explicit follow-up measurement.",
        "failure": "A misleading benchmark would compare different shapes, include first-call initialization, report one sample, or infer Tensor Core use from a fast BF16 result. Padding is also not automatically a fix: it may improve tile utilization while adding FLOPs and temporary storage. Accept padding only after measuring the complete padded operator and its downstream layout costs.",
        "next": "Profile both shapes with Nsight Compute and record the selected kernel, achieved occupancy, tensor-pipe utilization, DRAM throughput, and wasted edge work. Then repeat with Decode-like M values such as 1, 8, and 32. The exercise is successful when you can explain a reversal using both the trace and the timing distribution rather than the dtype label alone.",
        "extra_refs": [
            ("PyTorch numerical accuracy notes", "https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html"),
            ("Nsight Compute profiling guide", "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"),
        ],
    },
    3: {
        "hook": "Mixed-precision training is a feedback system. Autocast chooses operation dtypes during the forward pass, gradient scaling changes the numerical range seen by backward, and the optimizer must only step after gradients have been checked and unscaled. Demonstrating one BF16 activation therefore proves much less than demonstrating a complete, finite parameter update.",
        "checks": [
            "Predict the dtype of model parameters and forward outputs inside BF16 autocast.",
            "Predict whether GradScaler's scale should change when every gradient stays finite.",
            "Name the observation that proves an optimizer update occurred rather than only a forward pass.",
        ],
        "derivation": "Let the unscaled loss be `L` and the current scale be `S`. Backward differentiates `S·L`, producing scaled gradients `S·g`. Before the optimizer step, GradScaler divides by S and checks for Inf/NaN. If the check passes, the optimizer consumes g; if it fails, the step is skipped and the scale policy reacts. The ordering is semantic: clipping or inspecting gradients before unscale changes their meaning.\n\nAutocast is a dispatch policy, not a recursive call to `.to(bfloat16)` on the entire model. Eligible compute-heavy operations may emit BF16 while parameters and optimizer state remain FP32. BF16 does not usually need scaling for range in the way FP16 does, but exercising the full scaler API is still useful because the lesson is about the control loop and its evidence, not a single recommended dtype recipe.",
        "baseline": "FP32 parameters and optimizer state outside autocast",
        "candidate": "BF16 autocast forward wrapped in a complete scale/backward/step/update loop",
        "controlled": "same MLP, batch, targets, optimizer, seed, and six training steps",
        "metrics": "loss history, output dtype, parameter dtype, gradient finiteness, scaler value",
        "code_walk": "The environment cell verifies CUDA and fixes the random seed. The experiment constructs one small MLP, keeps its parameters in FP32, enters the autocast context only for forward and loss computation, and then executes the scaler sequence. Every step records five pieces of state so the notebook can distinguish dispatch, numerical health, and optimization progress.\n\nA decreasing toy loss is not a model-quality claim; it is a control-flow check. The stronger invariants are that every recorded gradient is finite, the output is BF16 under autocast, parameters remain FP32, and the loop reaches optimizer updates without an error output.",
        "result_reading": "The six saved steps reduced loss from 1.0376294 to 0.5965154. Every output was `torch.bfloat16`, every gradient check returned true, and parameters remained `torch.float32`. The scale stayed at 65536 because no non-finite event forced the policy to back off during this short run.\n\nTaken together, those fields establish a functioning mixed-precision loop on this PyTorch/CUDA stack. They do not establish faster training, convergence parity on a real dataset, or the best scale-growth policy. Those require longer runs with repeated timing and a frozen quality target.",
        "failure": "Common failures include calling `optimizer.step()` directly on scaled gradients, clipping before `unscale_`, moving master parameters to FP16, or judging success only from the forward dtype. A finite loss can coexist with zeroed small gradients, and a skipped optimizer step can be invisible unless the scale and parameter update are inspected.",
        "next": "Add a deliberately overflowing step and verify that GradScaler skips the update and changes its scale. Then time FP32, FP16+scaler, and BF16 autocast over a longer MLP while comparing the same validation loss trajectory. Preserve the exact optimizer, seed, and batch order so numerical and throughput decisions are not confounded.",
        "extra_refs": [
            ("PyTorch AMP examples", "https://docs.pytorch.org/docs/stable/notes/amp_examples.html"),
            ("PyTorch numerical accuracy notes", "https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html"),
        ],
    },
    4: {
        "hook": "FP16 and BF16 consume the same two bytes, but they spend those bits differently. BF16 inherits FP32's eight-bit exponent and sacrifices fraction precision; FP16 keeps a longer fraction but only five exponent bits. That trade changes where overflow occurs and how much rounding error accumulates, so a format decision cannot be made from byte count alone.",
        "checks": [
            "Predict which 16-bit format represents `1e5` without Inf and which produces the lower GEMM error.",
            "Predict whether equal storage implies equal GEMM latency on this GPU.",
            "Decide which metric would make you choose FP16 despite BF16's wider range.",
        ],
        "derivation": "A normalized binary floating-point value has the form `(-1)^s × 2^e × (1.f)`. Exponent bits determine dynamic range; fraction bits determine spacing between adjacent representable numbers at a fixed exponent. BF16's range is close to FP32, but its seven stored fraction bits make unit-roundoff much larger than FP16's ten. In a dot product, inputs are rounded before multiplication and partial sums may use a wider accumulator, so input format and accumulation format must be named separately.\n\nThis predicts a three-way trade: BF16 should survive large magnitudes, FP16 should often reconstruct ordinary-range values more accurately, and either 16-bit format may use a faster matrix path than FP32. The benchmark tests each axis independently instead of collapsing them into one winner.",
        "baseline": "FP32 GEMM and FP32 reference output",
        "candidate": "FP16 and BF16 GEMMs on the same 1536×1536 matrices",
        "controlled": "shape, random source values, GPU, warm-up, repetitions, comparison reference",
        "metrics": "finite-range probe, RMSE/cosine error, median and p90 latency",
        "code_walk": "The range probe casts `1e5` into both 16-bit formats and records finiteness. The GEMM probe uses the same logical matrices, evaluates output error against FP32, and times each path with twelve post-warm-up CUDA-event samples. Keeping the error and timing records side by side prevents a fast but numerically invalid path from looking successful.\n\nBecause the tensors are random and the shape is one square GEMM, the result is a format demonstration rather than a universal training recommendation. Real networks can amplify rounding through normalization, softmax, optimizer state, and long reductions.",
        "result_reading": "BF16 represented `1e5` while FP16 overflowed; the recorded maximum finite values were approximately `3.39e38` and `65504`. On the ordinary-range GEMM, FP16 had lower RMSE (0.014106) than BF16 (0.112772), exactly the fraction-bit trade predicted by the format layouts. Median latency was nearly tied—0.044608 ms for FP16 and 0.044160 ms for BF16—while FP32 took 0.133376 ms.\n\nThe evidence supports BF16 as a stability-first default for wide-range workloads, not as an accuracy or speed winner in every column. FP16 remained more precise for this input distribution and equally fast within the measured spread.",
        "failure": "Selecting BF16 solely because it did not overflow can hide unacceptable rounding error; selecting FP16 solely for lower RMSE can fail as soon as activations exceed its range. Another failure is to assume the accumulator shares the input dtype. Record autocast policy and operator behavior when reduction accuracy matters.",
        "next": "Repeat the experiment after scaling inputs across several orders of magnitude and add long reductions, softmax, and layer normalization. For training, compare loss curves and gradient-finiteness rates rather than one GEMM. A useful decision chart marks the magnitude range where FP16 first fails and the error tolerance where BF16 becomes unacceptable.",
        "extra_refs": [
            ("PyTorch tensor attributes and dtypes", "https://docs.pytorch.org/docs/stable/tensor_attributes.html"),
            ("PyTorch numerical accuracy notes", "https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html"),
        ],
    },
    5: {
        "hook": "A final NaN is the last symptom in a chain, not the diagnosis. FP16 can overflow during forward, overflow after loss scaling during backward, or silently round tiny gradients to zero. Each failure calls for a different response, so the first bad tensor must be located before changing the scaler.",
        "checks": [
            "Predict which combinations of gradient magnitude and loss scale become zero, finite, or infinite in FP16.",
            "Explain why loss scaling can rescue underflow but cannot repair a forward activation that is already Inf.",
            "Choose probe locations that distinguish forward, scaled-backward, unscaled-gradient, and parameter corruption.",
        ],
        "derivation": "With loss scale S, an exact gradient g is represented during backward as `Sg`. If g is smaller than the FP16 subnormal range, choosing a moderate S can move it onto the representable grid; unscale later restores its mathematical magnitude in a wider type. If `Sg > 65504`, the scaled gradient becomes Inf. And if a forward value already exceeded 65504, multiplying the loss later cannot reconstruct the discarded information.\n\nThis creates a feasible interval for S: large enough that important small gradients survive, but small enough that the largest scaled gradient remains finite. Dynamic scaling searches that interval through observed overflow. It does not guarantee that every tiny gradient is preserved or that the forward pass is stable.",
        "baseline": "FP16 casting of four gradient magnitudes with scale 1",
        "candidate": "the same magnitudes multiplied by scales 256 and 65536",
        "controlled": "tensor size, dtype, GPU, values within each magnitude group",
        "metrics": "zero fraction, finite fraction, Inf fraction, plus a separate forward-overflow probe",
        "code_walk": "The notebook sweeps a Cartesian product rather than waiting for a random training failure. For every magnitude/scale pair it casts the scaled value to FP16 and counts zero, finite, and infinite entries. A separate `1e5` forward probe establishes that some damage can occur before backward begins.\n\nBecause all elements in a row share one magnitude, fractions jump cleanly between zero, finite, and Inf. A real model would produce a distribution, but the synthetic grid makes the representability boundaries easy to see and debug.",
        "result_reading": "At magnitude `1e-8`, scale 1 rounded every value to zero, while scales 256 and 65536 made all entries finite and non-zero. At magnitude 1, scale 65536 overflowed every value. At magnitude 1000, scale 256 was already too large. The independent forward test confirmed that FP16 `1e5` was non-finite.\n\nThe same tool—larger scale—therefore fixes one row and breaks another. That is the central reason GradScaler adapts and skips unsafe optimizer steps. It is also why a scaler change is the wrong fix for forward overflow.",
        "failure": "Looking only at `torch.isfinite(loss)` misses underflow because zeros are finite. Looking only after unscale can hide where overflow began. Logging every tensor is too expensive, so production diagnosis usually places targeted hooks at loss, selected activations, scaled gradients, unscaled gradients, and parameters, then narrows the search.",
        "next": "Instrument a small FP16 network with hooks that report min/max, zero fraction, and finiteness at the four stages. Inject an activation spike and a tiny-gradient layer separately. Verify that lowering the scale helps the first backward-overflow case, raising it helps the underflow case, and neither repairs the injected forward Inf.",
        "extra_refs": [
            ("PyTorch AMP examples", "https://docs.pytorch.org/docs/stable/notes/amp_examples.html"),
            ("PyTorch numerical accuracy notes", "https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html"),
        ],
    },
    6: {
        "hook": "Timing and dispatch are different claims. A faster autocast region shows an application-level effect; a PyTorch profiler event identifies framework operators; only a lower-level trace can justify a native kernel or Tensor Core utilization claim. Good profiling keeps those evidence levels separate instead of using one as a shortcut for another.",
        "checks": [
            "Predict which PyTorch operator events should surround a BF16 matrix multiplication under autocast.",
            "Explain why warm-up and synchronization are required before comparing CUDA timings.",
            "Name the additional evidence needed to claim a particular native Tensor Core kernel.",
        ],
        "derivation": "GPU launches are asynchronous: host elapsed time can measure queue submission rather than device completion. CUDA events timestamp work in the device stream, but initialization, allocator growth, lazy library loading, and compilation can still contaminate early samples. A defensible steady-state number therefore specifies warm-up, synchronization, sample count, and a distribution statistic.\n\nA trace adds causality. At the framework layer, events such as `aten::matmul`, `aten::mm`, and casts reveal the operation graph and unexpected conversions. At the native layer, kernel names and hardware counters reveal tile implementation, tensor-pipe activity, occupancy, and bandwidth. The layers answer complementary questions; neither makes the other redundant.",
        "baseline": "theoretical expectation that autocast selects BF16 for an eligible GEMM",
        "candidate": "actual timed autocast region plus captured PyTorch operator events",
        "controlled": "2048×2048 shape, seed, GPU, five warm-ups, fifteen CUDA-event samples",
        "metrics": "median/p90 latency and selected framework operator names",
        "code_walk": "The notebook profiles one BF16 autocast matrix multiplication and records selected events from the PyTorch profiler. It separately times the same region with CUDA events. Keeping trace collection outside the timed samples avoids conflating profiler overhead with normal latency.\n\nThe result schema calls the events `pytorch_operator_events`, not `native_kernels`. That naming is deliberate: a framework trace is sufficient to audit the Python-level path but not to quantify Tensor Core occupancy.",
        "result_reading": "The saved run measured a 0.104416 ms median and 0.106240 ms p90 over fifteen samples. Five relevant PyTorch operator events were retained. The tight median-to-p90 spread suggests a stable microbenchmark after warm-up, while the event list confirms that an autocast/matmul path was captured.\n\nNothing in these two fields identifies one SASS kernel or reports hardware utilization. The bounded conclusion is therefore that the application path and timing were observed; a native dispatch claim remains open.",
        "failure": "Profiler traces can perturb timing, so reporting a profiled duration as production latency is risky. Conversely, timing without a trace can reward an unintended fallback or cached result. Other traps include missing synchronization, timing tensor allocation, and selecting only the fastest sample.",
        "next": "Capture the same operation in Nsight Systems to connect CPU launch, CUDA API, and kernel timeline, then use Nsight Compute for the selected kernel's tensor-pipe and memory metrics. Repeat with autocast disabled and with an awkward shape. Build one table that keeps wall-clock effect, framework dispatch, native kernel, and hardware counters in separate columns.",
        "extra_refs": [
            ("PyTorch profiler documentation", "https://docs.pytorch.org/docs/stable/profiler.html"),
            ("Nsight Systems user guide", "https://docs.nvidia.com/nsight-systems/UserGuide/index.html"),
            ("Nsight Compute profiling guide", "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"),
        ],
    },
    7: {
        "hook": "Calling a model INT4 usually describes only part of its state. Weight-only layers may store four-bit codes while activations and accumulators use BF16, the KV cache grows with context, and temporary workspaces appear only at runtime. Capacity planning fails when those objects are collapsed into one advertised precision.",
        "checks": [
            "Write the KV-cache byte formula before looking at the projected values.",
            "Predict which memory account grows with sequence length and which stays fixed for a loaded model.",
            "Explain why checkpoint size cannot predict peak CUDA allocation by itself.",
        ],
        "derivation": "For a decoder cache with batch B, layers L, sequence S, KV heads H, head dimension D, two tensors K and V, and b bytes per element, the leading storage is `2·B·L·S·H·D·b`. Weight storage is roughly `parameters × effective bits/8` plus scales and unquantized tensors. Activations depend on execution phase and liveness, while workspaces and allocator reserve depend on backend behavior.\n\nThese terms have different lifetimes. Weights persist after load, KV cache persists per active request, and many activations are temporary. That makes concurrency a multiplication on the cache term, not on the model weights. The ledger must keep bytes, lifecycle, and ownership together.",
        "baseline": "BF16 KV-cache projection and a real BF16 K/V allocation",
        "candidate": "INT8 cache projection for the same model geometry",
        "controlled": "batch 1, 32 layers, 8 KV heads, head dimension 128, identical context lengths",
        "metrics": "projected cache GiB by context and byte count of an allocated representative tensor pair",
        "code_walk": "The notebook first calculates the formula for three sequence lengths, then allocates representative K and V tensors on CUDA and checks their exact element-count bytes. This joins arithmetic with a live tensor object without pretending to load a full model.\n\nScales, paging fragmentation, prefix-cache blocks, and temporary attention workspaces are intentionally outside the simple projection. They belong in the next ledger revision when a named serving backend is tested.",
        "result_reading": "For the fixed 32-layer geometry, projected BF16 KV storage was 0.25 GiB at 2,048 tokens, 1.0 GiB at 8,192, and 4.0 GiB at 32,768. The INT8 arithmetic projection was exactly half each value. The live probe allocated two BF16 tensors of shape `[2, 4096, 8, 128]` totaling 16,777,216 bytes.\n\nThe linear fourfold growth from 8K to 32K is the important systems result. Weight quantization does not change it. Cache quantization may increase feasible context or concurrency, but only after scale overhead, attention compatibility, error, and latency are measured.",
        "failure": "A common error is multiplying weight memory by request count or forgetting to multiply cache by layers and by both K and V. Another is treating free memory reported before model load as deployable capacity. Allocator reserve, CUDA graphs, kernels, and safety margin must be added before setting concurrency.",
        "next": "Extend the ledger with grouped-query attention variants, tensor parallel sharding, cache block size, scale metadata, and allocator fragmentation. Then run a vLLM or TensorRT-LLM server and compare predicted versus observed cache capacity at 2K, 8K, and 32K contexts.",
        "extra_refs": [
            ("vLLM cache configuration", "https://docs.vllm.ai/en/stable/api/vllm/config/cache/"),
            ("vLLM quantized KV cache", "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/"),
        ],
    },
    8: {
        "hook": "The label INT4 hides the parameters that determine what four bits mean. Scale chooses the real interval covered by the codes, zero point chooses where real zero lands, and group size chooses how many values share one range estimate. Those choices change both reconstruction error and metadata, even before a deployment kernel enters the picture.",
        "checks": [
            "Derive symmetric INT4 quantize and dequantize equations for code range [-8, 7].",
            "Predict how RMSE, scale count, and effective bits per weight change as group size shrinks.",
            "Explain why saturation fraction alone does not rank quantizers.",
        ],
        "derivation": "For symmetric signed b-bit quantization, let `qmax = 2^(b-1)-1`, `s = max(|x|)/qmax`, `q = clamp(round(x/s), -qmax-1, qmax)`, and `x̂ = s·q`. With asymmetric quantization a zero point z shifts the code grid: `q = clamp(round(x/s)+z, qmin, qmax)` and `x̂=s(q-z)`. Grouping repeats this calculation over local slices rather than the whole tensor.\n\nIf each group stores one FP16 scale, its metadata cost is `16/group_size` bits per weight. Nominal INT4 therefore becomes 5.0 effective bits at group size 16, 4.25 at 64, and 4.125 at 128 before padding or zero-point metadata. Smaller groups can isolate outliers but may be incompatible with the fastest backend kernels.",
        "baseline": "one fixed outlier-containing 1024×1024 weight matrix",
        "candidate": "symmetric INT4 with group sizes 16, 64, and 128",
        "controlled": "same codes, scale dtype assumption, grouping axis, seed, and error reference",
        "metrics": "RMSE/cosine error, saturation fraction, scale count, effective bits per weight",
        "code_walk": "The notebook holds the matrix and quantization formula fixed and changes only group size. Each candidate is dequantized back to floating point before error is measured. Metadata is computed from the number of scales, making the storage comparison honest instead of repeating the nominal four-bit label.\n\nThis is a numerical model. It does not pack nibbles, instantiate a production quantized linear layer, or time an INT4 kernel. That separation lets the lab answer the math question without overstating backend performance.",
        "result_reading": "Group size 16 produced the lowest RMSE, 0.200316, and cosine 0.992188, but required 65,536 scales and 5.0 effective bits per weight. At group size 128, scale count fell to 8,192 and effective storage to 4.125 bits, while RMSE rose to 0.508112 and cosine fell to 0.950873. Group size 64 sat between them.\n\nThe saturation fraction decreased with larger groups because the shared maximum widened each step size; fewer values landed on the extreme code, but reconstruction became coarser. This is why a lower saturation count is not automatically a better quantizer.",
        "failure": "Comparing only weight RMSE ignores how inputs weight different columns. Comparing only effective bits ignores alignment, padding, and scale loads. Finally, a group size with good numerical behavior can lose in production if the backend does not provide a fused kernel for that layout.",
        "next": "Add asymmetric zero points for shifted distributions, compare per-row and per-column grouping, and weight the error by held-out activations. Then pack two INT4 codes per byte and time a compatible native kernel so numerical, storage, and operator gates are all represented.",
        "extra_refs": [
            ("TensorRT quantization workflows", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html"),
            ("PyTorch quantization fundamentals", "https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html"),
        ],
    },
    9: {
        "hook": "Post-training quantization freezes scales from examples. If those examples omit long prompts, a rare domain, or activation outliers, the quantizer can look excellent on calibration data and clip production traffic. Calibration quality is therefore a coverage problem before it is a sample-count problem.",
        "checks": [
            "Predict which calibration set minimizes clipping and which minimizes average rounding error on the mixed held-out set.",
            "Explain why evaluation data must remain separate after scale selection.",
            "List deployment strata that random sampling might under-represent.",
        ],
        "derivation": "A max-range calibrator chooses `s=max(|x_cal|)/qmax`; a percentile calibrator deliberately clips a tail to shrink the step. Both estimate a property of the calibration distribution. Generalization fails when the deployment distribution has larger or differently located tails. More copies of the same narrow prompts reduce estimator noise but do not reduce distribution bias.\n\nThe held-out clipping fraction measures values outside the frozen representable range. RMSE measures the combined cost of clipped tails and quantization steps. Those objectives can disagree: an outlier-aware scale can avoid clipping yet waste resolution on most ordinary values.",
        "baseline": "scales frozen from a narrow synthetic calibration distribution",
        "candidate": "balanced and explicitly outlier-aware calibration sets",
        "controlled": "INT8 formula, held-out mixed tensor, evaluation metrics, seed",
        "metrics": "frozen scale, held-out clipping fraction, RMSE, MAE, cosine, max error",
        "code_walk": "The notebook creates three calibration populations, freezes one scale from each, and evaluates all of them on the same mixed held-out tensor. It never recomputes a scale on evaluation data. That makes the comparison a small distribution-shift test rather than a reconstruction demo.\n\nThe examples are synthetic so domain labels are controllable. A model study would replace them with stratified prompts and layer activation captures while preserving the same calibration/evaluation separation.",
        "result_reading": "The narrow scale clipped 2.647752% of held-out values and produced RMSE 0.317395 with a max error of 27.9039. Balanced calibration reduced clipping to 0.025001% and RMSE to 0.085046. Outlier-aware calibration eliminated clipping, but its larger scale raised MAE to 0.067231; its RMSE, 0.077629, remained slightly better because it avoided catastrophic tail errors.\n\nThere is no universally best row without a deployment objective. If tail failures are unacceptable, the outlier-aware scale wins this probe. If average small-value resolution dominates, a clipped or mixed policy may be preferable.",
        "failure": "Tuning percentiles on the final regression set leaks evaluation into calibration. Reporting only mean error can hide rare catastrophic clipping, while reporting only max error can let one outlier consume the entire code range. Coverage metadata—domain, length, language, tool use, and frequency—is part of the quantization artifact.",
        "next": "Build a stratified calibration manifest for real prompts and compare random, balanced, and tail-oversampled selections at fixed sample count. Evaluate per-layer clipping and task slices on a disjoint set, then test whether the selected scale policy remains stable across model revisions.",
        "extra_refs": [
            ("NVIDIA Model Optimizer PTQ documentation", "https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html"),
            ("TensorRT quantization workflows", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html"),
        ],
    },
    10: {
        "hook": "LLM activations often contain persistent channel outliers that make one tensor-wide INT8 scale waste most of its codes. SmoothQuant does not delete those outliers; it moves part of their range into corresponding weight channels through an exactly equivalent floating-point reparameterization, then quantizes the easier pair.",
        "checks": [
            "Prove that reciprocal channel scaling leaves `XWᵀ` unchanged before quantization.",
            "Predict why alpha values near either endpoint can hurt combined W8A8 error.",
            "Choose the validation metric that should select alpha after calibration.",
        ],
        "derivation": "For positive channel scales s, define `X' = X / s` and `W' = W · s` along matching input channels. Then `X'W'ᵀ = (X/s)(W·s)ᵀ = XWᵀ`. A common SmoothQuant form constructs s from activation and weight maxima with an exponent alpha, so alpha controls how much range is assigned to each side.\n\nThe equality holds before quantization. After W8A8 rounding, shrinking activation outliers reduces activation step size while enlarged weight channels increase weight step size. The objective is the error of the composed quantized linear output, not activation amax in isolation.",
        "baseline": "W8A8 quantization without activation-to-weight migration (`alpha=0`)",
        "candidate": "reciprocal channel scaling for alpha 0.25, 0.5, 0.75, and 1.0",
        "controlled": "same outlier-heavy X and W, per-tensor INT8 reference quantizer, held shapes",
        "metrics": "floating-point equivalence max error and quantized output RMSE/cosine by alpha",
        "code_walk": "The notebook first evaluates the invariant in floating point for every alpha. Only after that check does it quantize both transformed tensors and compare the output with the original FP32 linear layer. This ordering prevents an algebra or broadcasting bug from being mistaken for quantization error.\n\nThe sweep uses one calibration-like tensor and reports a numerical model, not a TensorRT-LLM SmoothQuant kernel. A production experiment would freeze scales on calibration data, evaluate held-out tasks, and measure a named W8A8 backend.",
        "result_reading": "Floating-point equivalence stayed within roughly `6.1e-5` for every alpha. Quantized RMSE followed a U-shape: 3.298184 at alpha 0, 1.663379 at 0.25, a minimum of 1.151840 at 0.5, then 1.634807 at 0.75 and 3.224155 at 1.0. Cosine similarity peaked at 0.999785 for alpha 0.5.\n\nThe middle value balanced activation and weight difficulty for this synthetic distribution. The endpoints moved too much error to one side. This supports the migration mechanism while leaving the best alpha model- and layer-dependent.",
        "failure": "Choosing alpha from the same held-out set used for final quality reporting leaks the test. Reducing activation range without quantizing weights can give a false victory. Another failure is folding scales into weights but forgetting the corresponding activation transform or its runtime/fusion cost.",
        "next": "Freeze channel statistics on one tensor set and select alpha on a separate validation set, then report task quality on a third. Compare per-layer versus global alpha and inspect which layers retain outliers. Finally run a native W8A8 backend and verify that the scale transforms are folded or fused as intended.",
        "extra_refs": [
            ("SmoothQuant paper implementation", "https://github.com/mit-han-lab/smoothquant"),
            ("TensorRT quantization schemes", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html"),
        ],
    },
    11: {
        "hook": "Nearest-weight quantization assumes every weight error matters equally. A linear layer disproves that assumption: inputs can excite some columns strongly and barely touch others, so the same weight perturbation can create very different output error. GPTQ uses approximate second-order information to organize that sensitivity during one-shot quantization.",
        "checks": [
            "Predict whether raw weight RMSE or held-out layer-output RMSE better matches the GPTQ objective.",
            "Explain how input covariance makes two equal-magnitude weights differ in importance.",
            "State why the notebook's sensitivity fallback is an intuition model rather than a GPTQ implementation.",
        ],
        "derivation": "For a layer `Y=XWᵀ`, a weight perturbation ΔW produces `ΔY=XΔWᵀ`. The squared reconstruction loss is proportional to `||XΔWᵀ||²`, which can be written using the input Gram matrix `XᵀX`. This matrix is the local curvature signal: errors along frequently excited directions cost more than errors along quiet directions. GPTQ quantizes while using an approximate inverse Hessian to compensate remaining weights.\n\nThe notebook does not reproduce that sequential update. It uses an input-weighted error score to identify sensitive columns and preserves a fixed fraction in higher precision. That smaller construction isolates the central idea—optimize layer behavior, not the visual closeness of W—without claiming production GPTQ equivalence.",
        "baseline": "naive group-wise INT4 applied uniformly to the layer weights",
        "candidate": "INT4 plus a 12.5% input-sensitive column fallback",
        "controlled": "same layer, calibration inputs, held-out inputs, quantizer, and fallback budget",
        "metrics": "held-out output RMSE, MAE, cosine, max error, preserved fraction",
        "code_walk": "The experiment forms representative inputs, computes a naive quantized layer, estimates which columns create the largest input-weighted reconstruction cost, and restores only the highest-ranked columns. Both candidates are then evaluated on held-out inputs rather than on the calibration tensor used for ranking.\n\nThis makes the causal variable the allocation of a fixed precision budget. It still omits blockwise Hessian inversion, error propagation, act-order variants, packing, and a GPTQ runtime kernel, all of which are required for an end-to-end backend claim.",
        "result_reading": "Naive INT4 produced output RMSE 2.324805 and cosine 0.994488. Preserving 12.5% of columns selected by input-weighted sensitivity reduced RMSE to 1.597145 and raised cosine to 0.997389; MAE fell from 1.842271 to 1.269206.\n\nThe reduction demonstrates that equal storage bits can be allocated more intelligently when activation evidence is available. It does not show that this heuristic matches GPTQ quality, quantization time, or inference speed. Its value is to make the second-order objective observable in a small lab.",
        "failure": "Ranking on the final test inputs leaks evaluation and exaggerates robustness. Preserving columns also changes average bit width, so a fair comparison must report the precision budget. A low layer RMSE can still fail after nonlinearities or across a full model, and a good checkpoint can still be slow without a compatible packed kernel.",
        "next": "Replace the heuristic with a small sequential Hessian-aware quantizer and compare quantization order, damping, and block size. Then evaluate error layer by layer and after stacking several layers. Finally load a GPTQModel-compatible checkpoint in a serving backend and keep quantization quality separate from operator throughput.",
        "extra_refs": [
            ("GPTQ reference implementation", "https://github.com/IST-DASLab/gptq"),
            ("GPTQModel documentation", "https://github.com/ModelCloud/GPTQModel"),
        ],
    },
    12: {
        "hook": "AWQ begins from the observation that a small subset of weights can dominate model behavior when paired with large activation channels. Rather than minimizing average weight error, it uses activation statistics to search a per-channel scaling that protects salient weights while retaining a hardware-friendly weight-only layout.",
        "checks": [
            "Predict whether the largest weight magnitudes alone identify the best channels to protect.",
            "Explain why AWQ evaluates layer outputs on held-out activations instead of only weight reconstruction.",
            "Predict the shape of error as scaling strength increases from zero to one.",
        ],
        "derivation": "For a linear layer, equivalent channel scaling can transform weights and inverse-transform activations without changing the floating-point result. AWQ searches a scaling strength informed by activation magnitudes so quantization gives more effective resolution to salient channels. The W4A16 label means four-bit weight storage with floating-point activations; accumulation and other layers still need explicit dtypes.\n\nScaling too little leaves salient weights exposed. Scaling too aggressively expands other channels and makes their shared quantization ranges coarse. The optimum is therefore empirical and depends on calibration coverage, group size, layer distribution, and the held-out objective.",
        "baseline": "uniform W4A16 reference quantization at alpha 0",
        "candidate": "activation-aware channel scaling across alpha 0.25–1.0",
        "controlled": "same weights, calibration/held-out split, group quantizer, activation distribution",
        "metrics": "held-out layer-output RMSE, MAE, cosine and selected alpha",
        "code_walk": "The notebook freezes a calibration activation tensor, derives channel importance, searches five scaling strengths, and evaluates every candidate on held-out activations. The best alpha is chosen from output error, not weight error.\n\nThe code is an AWQ-inspired numerical model. It does not implement the paper's complete search, protect exactly the same salient set, reorder or pack weights, or dispatch an AWQ CUDA kernel. Those omissions are stated so the mechanism lesson is not confused with backend reproduction.",
        "result_reading": "Held-out RMSE improved from 2.771756 at alpha 0 to 2.273520 at alpha 0.25, then worsened to 2.562305, 3.748475, and 5.898383 as alpha increased. Cosine similarity followed the same pattern and peaked at 0.996096 for alpha 0.25.\n\nThe non-monotonic curve is the lesson: activation-aware protection can help, but more scaling is not more protection once it transfers too much range pressure elsewhere. The selected value is valid only for this frozen toy distribution.",
        "failure": "Using one activation batch for both search and final evaluation can overfit the scale. Reporting W4 storage without the higher-precision activation path misstates memory and compute. And a numerical improvement does not imply latency improvement; the online dequantization and packed GEMM path must exist for the chosen shape.",
        "next": "Repeat the search across several calibration domains and report how stable the selected alpha is. Compare magnitude-only, activation-only, and joint rankings at equal average bit width. Then test an official AWQ checkpoint with operator evidence and batch/sequence sweeps in a serving runtime.",
        "extra_refs": [
            ("AWQ paper", "https://arxiv.org/abs/2306.00978"),
            ("AWQ reference implementation", "https://github.com/mit-han-lab/llm-awq"),
        ],
    },
    13: {
        "hook": "QLoRA makes the base model cheap enough to keep frozen, but it does not make fine-tuning free. Activations, adapter parameters, gradients, optimizer states, temporary dequantization, and sequence length remain on the memory ledger. A useful feasibility calculation names each object instead of multiplying parameter count by four bits and stopping.",
        "checks": [
            "Estimate BF16 and ideal INT4 storage for seven billion parameters before opening the result.",
            "Identify which tensors require gradients in a LoRA update and which remain frozen.",
            "Explain why activation checkpointing can matter even when the base weights are four-bit.",
        ],
        "derivation": "A LoRA update writes `ΔW = BA`, where A and B have rank r much smaller than the full matrix dimensions. QLoRA keeps W frozen in a quantized representation, dequantizes as needed for compute, and backpropagates only into A and B. NF4 uses a non-uniform codebook designed for roughly normal weight distributions; double quantization compresses scale metadata, while paged optimizers address memory spikes.\n\nThe ledger separates persistent storage from training-time liveness. Ideal base bytes are `P·4/8`, but adapter weights, adapter gradients, two Adam moments, activations, and workspaces have their own dtype and multiplicity. Sequence length can dominate because saved activations scale with tokens even though base storage does not.",
        "baseline": "7B BF16 base-weight arithmetic plus a frozen CUDA reference layer",
        "candidate": "ideal INT4 base ledger with trainable low-rank adapters",
        "controlled": "parameter count, adapter rank assumption, optimizer-state rule, toy layer shape",
        "metrics": "base GiB, LoRA/Adam MiB, gradient finiteness, frozen-base flag, toy loss",
        "code_walk": "The notebook first computes a transparent 7B ledger. It then runs a small forward/backward pass in which the fake-quantized base matrix has `requires_grad=False` and only low-rank adapter matrices receive gradients. The finite-gradient check proves the intended training path exists on CUDA.\n\nThe fake quantizer explains memory ownership but is not bitsandbytes NF4. The ledger also excludes full-model activations because they depend on architecture, microbatch, sequence length, checkpointing, and attention implementation.",
        "result_reading": "The arithmetic ledger placed a 7B BF16 base at 13.039 GiB and ideal four-bit storage at 3.260 GiB. Under the toy adapter assumptions, trainable LoRA weights occupied 8 MiB and two Adam moments 32 MiB. The base stayed frozen and adapter gradients were finite.\n\nThose small adapter lines explain QLoRA's appeal, but the missing activation line can still be larger than the trainable state for long contexts. The result proves the ownership pattern and a toy CUDA backward pass, not a 7B end-to-end fine-tuning capacity number.",
        "failure": "Calling the base 'four-bit' while materializing a full BF16 copy defeats the ledger. Counting optimizer state for frozen weights overestimates memory, while omitting adapter moments underestimates it. A memory fit based on parameters alone can OOM during backward when saved activations and temporary buffers peak.",
        "next": "Run a real QLoRA step with bitsandbytes or another supported backend and measure `max_memory_allocated` by sequence length, microbatch, rank, and checkpointing policy. Compare predicted persistent bytes with observed peak, and explain the residual using allocator snapshots and activation liveness.",
        "extra_refs": [
            ("QLoRA paper", "https://arxiv.org/abs/2305.14314"),
            ("QLoRA reference implementation", "https://github.com/artidoro/qlora"),
        ],
    },
    14: {
        "hook": "`load_in_4bit=True` is not a complete numerical specification. A bitsandbytes configuration also selects a codebook such as NF4, a compute dtype for dequantized matrix operations, and optionally nested quantization for metadata. The loaded module class and backend availability determine whether those settings became a real operator or stayed configuration text.",
        "checks": [
            "Distinguish quantization codebook, packed storage dtype, compute dtype, and nested quantization.",
            "Predict whether NF4 or uniform INT4 gives lower RMSE for normally distributed weights.",
            "State what evidence would be required to label the result a native bitsandbytes run.",
        ],
        "derivation": "Uniform INT4 places evenly spaced reconstruction levels over a selected range. NF4 instead uses a non-uniform codebook whose levels allocate more resolution where a normal distribution has more probability mass. A stored code selects one level; matrix multiplication still needs dequantization/scaling and a floating-point compute path. Double or nested quantization reduces the cost of quantization constants, not the activation arithmetic to two bits.\n\nCodebook quality depends on the weight distribution and normalization rule. NF4 can lower average error for bell-shaped weights while producing a larger worst-case error near tails than a range-fitted uniform grid. The deployment decision also includes kernel support and compute dtype stability.",
        "baseline": "uniform symmetric INT4 reconstruction of normally distributed weights",
        "candidate": "reference NF4 codebook reconstruction of the same weights",
        "controlled": "weight tensor, normalization, number of codes, error reference, seed",
        "metrics": "RMSE/MAE/cosine/max error and bitsandbytes installation probe",
        "code_walk": "The notebook maps the same random weights through a reference NF4 codebook and a uniform INT4 quantizer, then compares both with the original tensor. It separately checks whether bitsandbytes is importable. Keeping these branches separate prevents a numerical codebook experiment from masquerading as a library benchmark.\n\nNo transformers model is loaded, no `Linear4bit` module is instantiated, and no bitsandbytes kernel is timed in the checked-in environment. The evidence label therefore remains `numerical-model`.",
        "result_reading": "NF4 achieved RMSE 0.127836 and MAE 0.109566, lower than uniform INT4 at RMSE 0.142396 and MAE 0.122676 for this normal tensor. Uniform INT4 had a smaller max error, 0.339059 versus NF4's 0.719360, showing that average and tail objectives can disagree. The environment probe reported `bitsandbytes_installed=false`.\n\nThe result supports the distribution-aware codebook intuition only. It says nothing about native layer memory, throughput, nested-quant overhead, or task quality on this RTX stack.",
        "failure": "A reference codebook can differ from library normalization, block size, packing, and scale dtype. Claiming bitsandbytes speed from it would be false. Another trap is choosing NF4 from average RMSE while a downstream layer is sensitive to rare tail errors.",
        "next": "Install a release compatible with the current PyTorch/CUDA stack, load one `Linear4bit` layer, and record its actual module, storage tensors, compute dtype, output error, and operator trace. Repeat with and without nested quantization and then with a small model-quality suite.",
        "extra_refs": [
            ("Transformers bitsandbytes guide", "https://huggingface.co/docs/transformers/main/quantization/bitsandbytes"),
            ("bitsandbytes documentation", "https://huggingface.co/docs/bitsandbytes/main/en/index"),
        ],
    },
    15: {
        "hook": "A PyTorch-native quantization API still depends on a precise package, ABI, hardware, and kernel combination. Conversion success, packed storage, numerical agreement, and latency are separate gates. Preserving a failed compatibility attempt is more useful than silently substituting a fake quantizer and calling it TorchAO.",
        "checks": [
            "Predict the evidence sequence required before comparing TorchAO INT4 latency with BF16.",
            "Decide whether an installed `torchao` package is enough to claim native execution.",
            "Explain how an ABI or auxiliary-kernel dependency can block an otherwise supported GPU.",
        ],
        "derivation": "A weight-only conversion replaces eligible linear modules with a representation that stores packed low-bit weights and dispatches a compatible operator for floating-point inputs. The theoretical bandwidth reduction appears only if conversion succeeds, packing is retained, and the runtime avoids materializing full dequantized weights. Package metadata alone proves none of those conditions.\n\nThe compatibility chain is `Python package → PyTorch ABI → auxiliary kernel package → GPU architecture → quantization config → converted module → executed operator`. A break near the beginning prevents meaningful memory, error, or latency comparison farther down the chain.",
        "baseline": "BF16 linear module, reserved as the fallback path",
        "candidate": "TorchAO `Int4WeightOnlyConfig` conversion on the same CUDA stack",
        "controlled": "PyTorch 2.12/CUDA 13 environment, RTX 5090, layer/config intent",
        "metrics": "package presence, conversion status, exact exception class/message; downstream metrics only on success",
        "code_walk": "The notebook imports TorchAO, constructs the intended conversion, and catches the exact failure rather than replacing the candidate. The JSON result records both `torchao_installed=true` and `conversion=failed`, which distinguishes package discovery from backend readiness.\n\nBecause conversion stopped before a quantized module existed, the notebook correctly omits storage, output-error, operator, and latency numbers for the candidate. Fabricating those from a reference quantizer would answer a different lesson.",
        "result_reading": "TorchAO was present, but conversion raised `ImportError: Requires mslk >= 1.0.0`. The native INT4 operator did not execute, so the evidence label is `compatibility-probe`, not `native-backend`. This negative result establishes the exact boundary of the saved environment and a reproducible next action.\n\nLesson 01 used a full-model TorchAO path that did execute under its tested configuration. The contrast is valuable: backend support can depend on API/configuration and dependency versions even on the same GPU, so results must stay attached to their exact path.",
        "failure": "The worst response would be to catch the error, run a hand-written fake quantizer, and leave the heading 'TorchAO benchmark'. Another failure is installing arbitrary nightly wheels until import succeeds without checking ABI compatibility or whether the environment was altered for other lessons.",
        "next": "Create an isolated environment using the TorchAO compatibility matrix, install a matching MSLK/PyTorch build, and rerun conversion. Only after success should the lab add module type, packed storage, output error, operator trace, warm-up, repeated latency, and a comparison with the BF16 baseline.",
        "extra_refs": [
            ("TorchAO quantization API", "https://docs.pytorch.org/ao/stable/api_reference/index.html"),
            ("TorchAO repository", "https://github.com/pytorch/ao"),
        ],
    },
    16: {
        "hook": "TensorRT INT4 is not merely a tensor cast. The graph must express quantize/dequantize semantics, weights must use supported per-block scales, and signed four-bit codes must be packed two per byte in the expected order. A correct reference packer is a prerequisite, not evidence that an engine was built.",
        "checks": [
            "Write the signed INT4 code range and calculate packed bytes for a 512×1024 matrix.",
            "Predict the metadata and error implications of block size 64.",
            "Separate Q/DQ correctness, packing correctness, engine build, operator trace, and timing into distinct gates.",
        ],
        "derivation": "For TensorRT-style symmetric INT4, codes lie in `[-8,7]` and dequantization multiplies by a per-block scale. Two four-bit two's-complement nibbles fit in one byte; unpacking must restore sign correctly. With 524,288 weights, ideal packed code storage is 262,144 bytes before scales and alignment.\n\nGraph Q/DQ nodes preserve the scale decision across export and allow the compiler to place quantized boundaries. TensorRT currently treats INT4 as weight-only and constrains block sizes/axes. A Python Q/DQ tensor can test the math, but only a serialized engine and inspected layer implementation establish TensorRT execution.",
        "baseline": "floating-point 512×1024 weight tensor",
        "candidate": "block-64 INT4 Q/DQ plus explicit nibble pack/unpack",
        "controlled": "weight tensor, grouping axis, scale rule, code order, CUDA numerical reference",
        "metrics": "packed bytes, exact code round-trip, RMSE/cosine, TensorRT package probe",
        "code_walk": "The notebook quantizes blocks, packs adjacent signed codes into low/high nibbles, unpacks them, restores sign, and asserts exact equality with the original codes. It then dequantizes for error measurement. A separate import probe records whether TensorRT is available.\n\nThis ordering distinguishes serialization bugs from numerical loss. Exact code round-trip is necessary even when dequantized RMSE looks plausible, because a nibble-order or sign bug can be masked by aggregate statistics.",
        "result_reading": "The 512×1024 matrix produced exactly 262,144 packed bytes, and every code survived pack/unpack. Block-64 Q/DQ yielded RMSE 0.107706 and cosine 0.994257. TensorRT was not installed, so no engine, TensorRT layer, or latency result exists.\n\nThe outcome validates a semantic reference and serialized code layout. It does not validate TensorRT's supported axis rules for a concrete ONNX graph or the performance of an INT4 WoQ kernel.",
        "failure": "Mistakes include treating unsigned nibbles as signed values, reversing low/high order, dropping scale layout, or claiming 0.5 byte per weight without metadata and padding. A successful engine build can still insert dequantize work that defeats the expected benefit, so engine inspection is required.",
        "next": "Export a minimal Q/DQ ONNX graph with block size 64, build it under a pinned TensorRT version, inspect the engine layers, and compare outputs with the reference packer. Then profile latency and memory for several M dimensions to find where WoQ becomes beneficial.",
        "extra_refs": [
            ("TensorRT capabilities", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html"),
            ("TensorRT quantization workflows", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html"),
        ],
    },
    17: {
        "hook": "Quantization pipelines cross tool boundaries: calibration may happen in ModelOpt, checkpoint export in one schema, and engine build in TensorRT-LLM. If model revision, recipe, scales, build flags, and rollback identity are not carried together, a fast engine cannot be reproduced or safely compared with its baseline.",
        "checks": [
            "List the fields required to reproduce a quantized checkpoint-to-engine handoff.",
            "Explain why a scale checksum is useful but insufficient for engine identity.",
            "Predict the decision when neither ModelOpt nor TensorRT-LLM is installed.",
        ],
        "derivation": "A pipeline artifact is a directed chain: base model revision → calibration sample manifest → quantization recipe and scales → exported checkpoint → builder version/flags → engine → quality and performance report. Hashes establish byte identity at a boundary; semantic fields establish how those bytes should be interpreted.\n\nFP8, INT4, and FP4 are different graph and scaling recipes, not points on one interchangeable slider. The manifest should therefore make format, group/block size, calibration, handoff status, and rollback target explicit. Missing stages remain false rather than being inferred from a numerical probe.",
        "baseline": "versioned BF16 rollback revision",
        "candidate": "INT4 handoff manifest with scale fingerprint",
        "controlled": "fixed synthetic scale tensor, schema requirements, base/rollback identifiers",
        "metrics": "manifest completeness, SHA-256 fingerprint, package availability, numerical Q/DQ error",
        "code_walk": "The notebook generates a small CUDA quantization fingerprint, hashes the scale bytes, and builds a manifest with required fields. It independently probes ModelOpt and TensorRT-LLM and records both handoff flags. Validation checks schema completeness, not engine success.\n\nThis is intentionally a pipeline-contract lab. The synthetic Q/DQ error catches accidental recipe changes, while the hash catches byte changes; neither substitutes for loading the exported checkpoint or building an engine.",
        "result_reading": "The manifest passed its required-field check and recorded scale SHA-256 `4fc993…d117e`. The numerical probe had RMSE 0.107446 and cosine 0.994265. Both ModelOpt and TensorRT-LLM handoff flags were false because the packages were unavailable.\n\nThat combination is a valid reproducibility artifact and an explicit stop. It supports preparing the handoff schema, not claims about FP8/INT4/FP4 engine quality or throughput.",
        "failure": "Using `latest` model or container tags makes a manifest non-reproducible. Hashing scales but omitting the grouping axis can preserve bytes while changing meaning. Another failure is comparing engines built with different scheduler, tensor-parallel, or plugin settings and attributing the difference to quantization alone.",
        "next": "Run ModelOpt calibration in an isolated pinned container, export a checkpoint plus manifest, build a TensorRT-LLM engine, and add engine hash, builder flags, layer inspection, quality suite, and SLO report. Test that the rollback artifact loads under the same serving interface.",
        "extra_refs": [
            ("NVIDIA Model Optimizer documentation", "https://nvidia.github.io/Model-Optimizer/"),
            ("TensorRT-LLM documentation", "https://nvidia.github.io/TensorRT-LLM/"),
        ],
    },
    18: {
        "hook": "Serving performance belongs to a runtime, not to a checkpoint label. vLLM combines quantized linear kernels with scheduling, continuous batching, paged KV cache, prefix caching, and a request distribution. A PyTorch microbenchmark can warn about shape sensitivity, but it cannot stand in for requests-per-second or latency percentiles from a vLLM server.",
        "checks": [
            "Separate checkpoint-format support, hardware support, kernel dispatch, and service-load performance.",
            "Predict whether the reference W4 dequantized matrix path wins at every tested batch.",
            "Design a serving workload that reports TTFT and inter-token latency separately.",
        ],
        "derivation": "Prefill and Decode produce different matrix shapes and interact differently with batching. Service throughput also depends on arrival rate, prompt/output lengths, scheduler policy, cache capacity, and queueing. A weight-only checkpoint that loads successfully can still fall back to a slow kernel for some layers or lose its memory benefit to KV cache at long context.\n\nThe acceptance chain is format metadata → model load → quantized module/operator trace → output quality → controlled request workload → latency/throughput/capacity. An import probe only reaches the first compatibility edge.",
        "baseline": "BF16 PyTorch matrix path for batches 1, 8, and 32",
        "candidate": "reference dequantized W4 matrix path at the same shapes",
        "controlled": "weight/input shapes, GPU, warm-up, repetitions; no server or scheduler",
        "metrics": "operator median/p90 by batch plus vLLM installation and service-benchmark status",
        "code_walk": "The notebook probes vLLM availability, then runs a backend-independent PyTorch shape experiment. The W4 candidate is a dequantized reference tensor, so it tests how the resulting matrix shape behaves—not vLLM's AWQ/GPTQ kernel. Results are stored under `pytorch_shape_warning` to make that boundary visible.\n\nA true service cell would start a server, wait for readiness, issue a frozen request trace, collect TTFT/ITL/latency percentiles and throughput, then terminate cleanly. None of that is synthesized here.",
        "result_reading": "The tiny matrix probe produced nearly tied medians: at batch 1, BF16 was 0.019520 ms and the reference W4-dequant tensor 0.019424 ms; at batch 8 they were 0.019168 and 0.018912 ms; at batch 32 the candidate reversed slightly to 0.019072 versus 0.018976 ms. vLLM was not installed and service performance is explicitly `not_measured`.\n\nSub-microsecond differences of this kind are not a serving result. They show that shape can reverse a small operator comparison and reinforce why a full request workload is needed.",
        "failure": "Reporting this table as vLLM speed would mislabel the backend and ignore scheduling. Other traps are benchmarking one warm cache prompt, mixing different model revisions, omitting output length, and comparing throughput at unequal latency or quality. Quantization compatibility matrices also change across versions, so the exact release must be pinned.",
        "next": "Install a supported vLLM release in a separate environment, load one documented AWQ or GPTQ model, confirm module/operator selection, and run `vllm bench serve` with fixed prompt/output distributions and concurrency. Report TTFT p50/p95, ITL, end-to-end latency, tokens/s, GPU memory, and rejected requests.",
        "extra_refs": [
            ("vLLM quantization documentation", "https://docs.vllm.ai/en/latest/features/quantization/"),
            ("vLLM benchmark CLI", "https://docs.vllm.ai/en/latest/cli/bench/serve.html"),
        ],
    },
    19: {
        "hook": "Once weights are compressed, KV cache can become the dominant memory term for long contexts and concurrent requests. Quantizing it changes more than capacity: scales must be stored or computed, keys and values are reconstructed inside attention, and small perturbations can change softmax-weighted outputs.",
        "checks": [
            "Compute BF16 bytes for K and V with shape `[1,4096,8,128]` before reading the artifact.",
            "Predict the ideal INT8 reduction and identify why the measured reduction is smaller than 50%.",
            "Choose an output-level metric that is more informative than K/V tensor RMSE alone.",
        ],
        "derivation": "Cache storage is `2·B·S·Hkv·D·bytes`, multiplied by layers in a full model. Quantization adds scale metadata whose granularity may be per tensor, head, token, or block. Attention consumes `softmax(QKᵀ/√D)V`; errors in K affect logits and softmax weights, while errors in V affect the weighted sum. Their consequences are therefore not captured by one raw cache-error number.\n\nCapacity improves only if the backend stores the quantized form persistently rather than dequantizing a full copy. Latency may improve, stay flat, or worsen depending on fused attention support and scale handling.",
        "baseline": "BF16 K and V tensors for one representative long-context attention slice",
        "candidate": "INT8 K/V plus explicit scale storage",
        "controlled": "batch, sequence 4096, 8 KV heads, head dimension 128, queries, attention computation",
        "metrics": "total bytes including scales and attention-output RMSE/cosine",
        "code_walk": "The notebook creates real CUDA K/V tensors, quantizes them, counts code and scale bytes, and evaluates attention outputs against the BF16 reference using the same query tensor. Measuring after the softmax/value path ties numerical error to the consumer of the cache.\n\nThis remains a reference implementation. It does not exercise vLLM's FP8 cache format, paged block allocator, per-head scaling, or a fused quantized attention kernel, so service latency is outside the claim.",
        "result_reading": "BF16 cache storage was 16,777,216 bytes. INT8 codes plus scales used 8,650,752 bytes, a 48.4375% reduction rather than an ideal 50% because metadata remained. Attention-output RMSE was 0.00023131 with cosine 0.999958 and max absolute error 0.00070267.\n\nThe error is small for this random slice, but it is not a language-model quality result. The useful conclusion is that metadata-aware capacity and consumer-level numerical error were both measured; end-to-end quality and fused-kernel cost remain open.",
        "failure": "Ignoring scale bytes overstates capacity, while comparing cache tensors without attention can understate behavioral impact. A single random context misses layer-dependent and long-range sensitivity. Another failure is to count extra capacity as throughput without testing whether scheduler concurrency and attention latency actually improve.",
        "next": "Repeat by layer/head and context length, compare per-tensor versus per-head scales, and evaluate logit/sequence quality in a small model. Then run a supported vLLM FP8 KV-cache configuration and measure maximum tokens, concurrent requests, TTFT, ITL, and accuracy under the same request set.",
        "extra_refs": [
            ("vLLM quantized KV cache", "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/"),
            ("LLM Compressor KV-cache example", "https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/"),
        ],
    },
    20: {
        "hook": "A dtype name can exist at four levels: a mathematical format, a hardware instruction, a library recipe, and a framework operator. Blackwell support does not guarantee that the installed PyTorch, Transformer Engine, TensorRT, or ModelOpt build exposes the same FP8 or NVFP4 path. Each layer must be probed independently.",
        "checks": [
            "Distinguish E4M3 FP8 from E5M2 and ordinary INT4 from block-scaled NVFP4.",
            "Predict whether the installed PyTorch build can execute a scaled FP8 matrix multiply.",
            "State what additional evidence is needed before claiming NVFP4 performance.",
        ],
        "derivation": "FP8 E4M3 allocates four exponent and three fraction bits after sign, trading range for precision; E5M2 spends another bit on range. NVFP4 uses FP4 E2M1 values with block scaling, so its real representation includes both four-bit data and scale hierarchy. TensorRT's current scheme uses block size 16 for NVFP4, while framework APIs and supported axes remain version-specific.\n\nScaled matrix multiplication also requires choosing input and output scales. A successful `torch._scaled_mm` call proves one framework-level path for one shape and format; it does not prove Transformer Engine recipes or TensorRT NVFP4 kernels.",
        "baseline": "higher-precision reference matrix multiplication for error comparison",
        "candidate": "PyTorch scaled FP8 E4M3 GEMM on RTX 5090",
        "controlled": "1024-class matrix shape, scaling procedure, warm-up, fifteen timing samples",
        "metrics": "API success, RMSE/cosine, median/p90, library availability, NVFP4 status",
        "code_walk": "The notebook checks for float8 dtype support and calls `torch._scaled_mm` with explicit scales. It compares the output with a higher-precision reference and times repeated CUDA execution. Separate probes record Transformer Engine availability and leave NVFP4 `not_measured` when its recipe/operator is unavailable.\n\nThis design prevents the real FP8 result from being generalized to a different format. The JSON names the exact API so a future software change can be detected.",
        "result_reading": "The scaled FP8 GEMM succeeded through `torch._scaled_mm`, with median 0.017568 ms and p90 0.018560 ms over fifteen samples. Output cosine was 0.999285 and RMSE 1.208455 for the tested scale and shape. Transformer Engine was not installed, and NVFP4 remained `not_measured`.\n\nThe measured path is therefore real PyTorch GPU evidence for FP8, not proof of a Transformer Engine or NVFP4 backend. The absolute error also shows why format support must be paired with scaling and quality policy.",
        "failure": "Casting tensors to a float8 dtype without a successful matrix operator proves storage only. Comparing raw FP8 latency against a different shape or excluding scale computation can misstate speed. Treating NVFP4 as signed uniform INT4 loses its block-scale semantics entirely.",
        "next": "Install a matching Transformer Engine or TensorRT stack in isolation, run documented FP8 and NVFP4 recipes, and capture operator identity, scale granularity, end-to-end scale overhead, error, and latency. Build a matrix with rows for format and columns for hardware, library, API, operator, and tested status.",
        "extra_refs": [
            ("Transformer Engine documentation", "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html"),
            ("TensorRT quantization schemes", "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html"),
            ("TensorRT DynamicQuantize operator", "https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/operators/DynamicQuantize.html"),
        ],
    },
    21: {
        "hook": "A multimodal model does not have one activation distribution. Patch projection sees pixels and local contrast, the vision encoder sees image tokens, the connector maps modalities, and the language decoder sees text-conditioned states. Calibrating only text can leave the vision path with unobserved ranges and brittle low-bit behavior.",
        "checks": [
            "Predict how high-contrast image patches change the output error of one quantized patch projection.",
            "Identify the calibration strata needed for a vision-language model rather than a text-only LLM.",
            "Explain why a patch-projection result cannot establish full VLM quality.",
        ],
        "derivation": "A ViT patch projection is a convolution with kernel and stride equal to patch size. For 16×16 RGB patches, each output token combines 768 input values. Weight quantization error is filtered by the image distribution: high-contrast or sparse extreme pixels can amplify particular columns that ordinary Gaussian-like calibration does not emphasize.\n\nFarther downstream, cross-attention and modality connectors introduce their own outliers and quality objectives. A sound plan therefore calibrates and evaluates per component and per modality slice, then rejoins them with end-to-end captioning, VQA, OCR, or grounding tasks.",
        "baseline": "floating-point 64-channel, 16×16 patch projection",
        "candidate": "group-192 INT4-dequantized projection weights",
        "controlled": "same weights, image shape, projection stride, normal/high-contrast paired inputs",
        "metrics": "projection-output RMSE/MAE/cosine/max error for each image distribution",
        "code_walk": "The notebook isolates the first vision operation so tensor axes remain readable: weights have shape `[64,3,16,16]`, then flatten to groups for quantization and return to convolution layout. It evaluates the same candidate on ordinary random images and images with periodic contrast spikes.\n\nThis component test answers whether input distribution changes local error. It deliberately excludes transformer blocks, the language decoder, preprocessing, and task metrics, so its conclusion stops before full-model quality.",
        "result_reading": "For normal images, projection RMSE was 0.040851 with cosine 0.997531. High-contrast patches increased RMSE to 0.063011 and max absolute error from 0.202358 to 0.358466, while cosine remained 0.997666.\n\nThe higher absolute error under contrast shift shows why one calibration distribution is insufficient even when cosine looks stable. Whether that change affects a VLM answer depends on downstream normalization and attention, which this lab does not model.",
        "failure": "A text-only calibration set never exercises the patch projection. Average image embeddings can also hide OCR, diagrams, dark images, or saturated regions. Another mistake is to use image reconstruction metrics for a model whose deployment objective is answer correctness or grounding.",
        "next": "Capture activation ranges from photographs, documents, charts, OCR-heavy images, and high-contrast synthetic cases. Quantize vision encoder, connector, and decoder separately, then run end-to-end task slices. Use mixed precision when one modality component is consistently more sensitive.",
        "extra_refs": [
            ("AWQ paper", "https://arxiv.org/abs/2306.00978"),
            ("PyTorch Conv2d documentation", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"),
        ],
    },
    22: {
        "hook": "A quantized model is deployable only when its bytes and interpretation travel together. Packed weights without scales are meaningless; correct weights with the wrong tokenizer or base revision are unsafe; and an artifact without a checksum cannot be distinguished from a partial copy. Packaging is therefore part of inference correctness, not administrative cleanup.",
        "checks": [
            "List the minimum fields needed to load, validate, and roll back a quantized artifact.",
            "Predict the packed payload size for the notebook's weight and scale tensors.",
            "Explain what a SHA-256 digest proves and what semantic errors it cannot detect.",
        ],
        "derivation": "A package contract binds schema version, base revision, quantization format, group size/axis, tensor shapes, scale dtype, runtime/backend requirement, checksums, and rollback target. The checksum establishes byte identity; the schema establishes how those bytes should be decoded. Both are needed.\n\nProduction packages also include tokenizer/config files, special-token policy, architecture code revision, licenses, model card, and quality/performance reports. Keeping a minimal manifest in the lab makes the invariant testable without publishing checkpoint data.",
        "baseline": "unversioned in-memory reference weights with no handoff contract",
        "candidate": "temporary packed INT4 payload plus validated manifest and checksum",
        "controlled": "fixed tensor shape, group size, serializer, required-field set, rollback ID",
        "metrics": "payload bytes, SHA-256, required-field completeness, cleanup status",
        "code_walk": "The notebook quantizes a 256×512 matrix, serializes codes and scales into a temporary file, computes SHA-256, records size and interpretation fields, validates required keys, and lets the temporary payload disappear after the check. Only the small manifest evidence remains public.\n\nThe exercise proves packaging logic without committing weights. It does not claim compatibility with SafeTensors, Hugging Face quantization configs, or a named production runtime.",
        "result_reading": "The generated reference payload was 139,264 bytes and received digest `bd46d8…714de`. Every required manifest field was present, including base revision, format, group size, shape, runtime, and BF16 rollback target. The temporary payload was deleted after validation.\n\nThis is a reproducibility and safety result: another process can verify identity and interpretation metadata. It is not a model export, engine load, or distribution license decision.",
        "failure": "A digest cannot detect that the wrong scale axis was declared if both producer and consumer share the same bad schema. Mutable model tags and missing tokenizer revisions also break reproducibility. Never place secrets, local paths, proprietary weights, or unlicensed datasets in a public package to make a tutorial appear complete.",
        "next": "Define a JSON Schema for the manifest, add per-file hashes and total-size checks, and write a loader that rejects an incompatible runtime or base revision before allocating GPU memory. Test truncation, swapped scale files, wrong group size, and rollback loading as deliberate failure cases.",
        "extra_refs": [
            ("SafeTensors documentation", "https://huggingface.co/docs/safetensors/index"),
            ("Hugging Face model cards", "https://huggingface.co/docs/hub/model-cards"),
        ],
    },
    23: {
        "hook": "Quantization quality is not one cosine score. A release can preserve average logits while changing top-1 decisions, rare domains, long-context behavior, calibration-sensitive layers, or safety-critical outputs. Regression testing turns those failure modes into frozen gates that can block a numerically small but behaviorally important change.",
        "checks": [
            "Predict whether cross-entropy, perplexity, and top-1 agreement will all move in the same relative direction.",
            "Explain why the synthetic perplexity magnitude is not meaningful as a language-model score.",
            "Design at least three deployment slices that an aggregate metric could hide.",
        ],
        "derivation": "For targets y and logits z, cross-entropy measures probability assigned to y; perplexity is `exp(loss)` and can magnify small loss changes. Top-1 agreement instead asks whether the candidate preserves the baseline decision, regardless of whether either decision is correct. Logit distance, task accuracy, exact-match, calibration, and human/safety checks answer still different questions.\n\nA release gate should define baselines, datasets, seeds, tolerances, and slice policies before the candidate is evaluated. Otherwise thresholds drift to accommodate the observed regression.",
        "baseline": "floating-point synthetic classifier logits over 4,096 tokens",
        "candidate": "INT4-dequantized weight logits for the same hidden states and targets",
        "controlled": "tokens, vocabulary, targets, hidden states, weight matrix, seed",
        "metrics": "loss, derived perplexity, overall and half-slice top-1 agreement",
        "code_walk": "The notebook generates one fixed synthetic classification problem, computes baseline and quantized logits, and evaluates the same targets. It reports the complete set and two halves so a slice disagreement cannot be hidden by one aggregate.\n\nBecause random logits yield enormous losses and perplexities, the absolute values are intentionally labeled synthetic. The exercise demonstrates metric relationships and gate structure, not language modeling ability.",
        "result_reading": "Candidate loss increased from 32.049492 to 32.212620. Exponentiation turned that modest difference into synthetic perplexities of about `8.30e13` and `9.77e13`. Overall top-1 agreement was 0.836914; the two halves were 0.838379 and 0.835449.\n\nThe near-equal slices do not reveal a concentrated failure in this constructed set, but roughly 16% decision disagreement is clearly visible. A real release would need task correctness, not only agreement with the baseline.",
        "failure": "Perplexity can overflow or become hard to interpret at extreme synthetic losses. Baseline agreement can preserve a baseline mistake, and average accuracy can hide a critical slice. Reusing calibration prompts for regression also lets quantizer selection overfit the gate.",
        "next": "Replace synthetic logits with a small named model and a frozen, redistribution-safe suite: perplexity on held-out text, task accuracy, long-context slices, multilingual/code/tool-use samples, and answer/logit agreement. Publish thresholds and reversal criteria before running the candidate.",
        "extra_refs": [
            ("lm-evaluation-harness", "https://github.com/EleutherAI/lm-evaluation-harness"),
            ("PyTorch reproducibility notes", "https://docs.pytorch.org/docs/stable/notes/randomness.html"),
        ],
    },
    24: {
        "hook": "Throughput, latency, concurrency, and memory are coupled but not interchangeable. A larger batch can raise examples per second while increasing per-request waiting time and peak memory. A useful benchmark starts from an SLO and a request distribution, then reports enough axes to explain why one configuration wins.",
        "checks": [
            "Predict how operator latency, examples per second, and peak allocation change from batch 1 to 128.",
            "Explain why this operator benchmark cannot report time to first token or queueing latency.",
            "Choose percentile and load information required for a serving comparison.",
        ],
        "derivation": "For a synchronous batch B with measured operator time T, idealized throughput is `B/T`. That calculation excludes arrivals, batching delay, scheduler overhead, token-by-token Decode, and response streaming. In a service, increasing concurrency can improve GPU utilization until queueing and memory pressure drive tail latency or rejection.\n\nLatency distributions also matter: median describes a typical warm request, while p95/p99 expose interference and queueing. Peak CUDA allocation is not total process memory and should be paired with reserved memory and cache capacity when deployment fit is evaluated.",
        "baseline": "batch-1 BF16 two-layer MLP operator workload",
        "candidate": "the same operator at batches 8, 32, and 128",
        "controlled": "model, hidden sizes, dtype, GPU, warm-up five, repeats twenty",
        "metrics": "median/p90 operator latency, derived examples/s, peak allocated MiB",
        "code_walk": "The notebook constructs one fixed BF16 MLP, allocates each batch input, resets peak-memory statistics, and records twenty CUDA-event samples after warm-up. Throughput is derived from batch divided by median device time, making its simplified assumptions explicit.\n\nNo request scheduler, tokenizer, KV cache, network, or output loop is present. The evidence is an operator batching curve that teaches metric relationships, not an online service benchmark.",
        "result_reading": "Median operator latency stayed near 0.046 ms from batch 1 through 32, so derived throughput rose from 21,656 to 696,136 examples/s. Batch 128 increased median to 0.049024 ms but still reached 2.61 million examples/s. Peak allocated memory grew from 64.023 to 67.000 MiB.\n\nThe table shows why throughput can improve dramatically while latency barely changes and memory rises. It does not include the time a request waits to join that batch, which may dominate an interactive SLO.",
        "failure": "Comparing tokens/s at different output lengths or latency percentiles is unfair. Deriving service throughput from a single operator omits non-quantized layers and scheduling. Another trap is reporting only the best concurrency before OOM or rejection, without a safety margin and sustained-load duration.",
        "next": "Run a real serving workload with a frozen prompt/output-length distribution and arrival process. Record TTFT, inter-token latency, end-to-end p50/p95/p99, tokens/s, requests/s, queue depth, rejection, power, and peak/reserved memory for each concurrency level.",
        "extra_refs": [
            ("vLLM benchmark CLI", "https://docs.vllm.ai/en/latest/cli/bench/serve.html"),
            ("CUDA event timing API", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html"),
        ],
    },
    25: {
        "hook": "Quantization often fails by regime rather than on average. Activation outliers amplify weight error, a shifted domain changes channel importance, tiny batches expose launch/dequant overhead, long context expands cache pressure, and MoE routing concentrates work unevenly. A failure matrix makes those reversals visible before production does.",
        "checks": [
            "Predict which synthetic case produces the largest W4 output RMSE.",
            "Predict whether the reference W4 path is faster in every batch/distribution case.",
            "Design separate tests for long-context cache and MoE routing, which this linear probe does not contain.",
        ],
        "derivation": "For fixed weight error ΔW, output error is `XΔWᵀ`; scaling or shifting X directly changes its magnitude and direction. This explains why a quantizer calibrated on ordinary activations can degrade under outliers or domain shift without any weight bytes changing. Small batches add a systems failure mode because fixed launch, unpack, or scale overhead is amortized over less work.\n\nLong context and MoE require additional objects: cache bytes/attention error and expert-routing load balance. They belong in the matrix but cannot be inferred from one dense linear layer.",
        "baseline": "BF16 matrix multiplication in four controlled input regimes",
        "candidate": "the same multiplication with group-128 INT4-dequantized weights",
        "controlled": "weight matrix and quantizer; only batch/distribution regime changes",
        "metrics": "output RMSE/cosine/max error and median/p90 timing per regime",
        "code_walk": "The notebook quantizes one weight matrix once, then evaluates ordinary, batch-1, activation-outlier, and shifted-domain inputs. Each row carries both numerical error and timing for baseline/candidate. That paired design prevents a quality failure from being hidden by a small speed result.\n\nThe candidate is a dequantized PyTorch reference tensor, not a packed production W4 kernel. Timing differences therefore illustrate regime sensitivity of the composed path, not an INT4 hardware speed claim.",
        "result_reading": "Ordinary and batch-1 RMSE were about 3.73 and 3.69. Activation outliers raised RMSE to 14.0754 and max error to 59.1648; the shifted domain produced RMSE 13.7526 and max error 67.1768. Timing changes stayed tiny and varied by row.\n\nAn aggregate over all four cases could hide the roughly 3.7x error jump in the shifted regimes. The correct response is a targeted calibration, fallback, or rejection rule—not a global statement that W4 is acceptable.",
        "failure": "One stress tensor cannot represent production tail frequency, and synthetic timing with dequantized weights is not a native backend result. A matrix that lists long context or MoE without actually constructing cache or routing evidence would also be misleading; unexecuted axes must remain marked as future gates.",
        "next": "Add a real KV-cache length sweep, rare-language/code/tool-use activation captures, and a toy MoE with expert-load imbalance. Define acceptance by slice, not only aggregate. Use the failing rows to design mixed-bit fallbacks and then rerun the full matrix.",
        "extra_refs": [
            ("AWQ paper", "https://arxiv.org/abs/2306.00978"),
            ("vLLM quantized KV cache", "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/"),
        ],
    },
    26: {
        "hook": "Uniform four-bit quantization spends the same precision on layers with different sensitivity. A mixed-bit policy measures how much each layer perturbs the end-to-end output, then allocates a fixed higher-precision budget to the worst offenders. The budget and the reassembled model result are as important as the ranking.",
        "checks": [
            "Predict which layers receive INT8 when only two fallbacks are allowed.",
            "Compute the expected average bit width for two INT8 and four INT4 equal-size layers.",
            "Explain why isolated layer sensitivity must be followed by an assembled-model evaluation.",
        ],
        "derivation": "Let candidate bit assignment b_l minimize model error subject to `Σ n_l b_l / Σ n_l ≤ B`, where n_l is layer size and B is the average-bit budget. A simple greedy policy measures the output RMSE caused by quantizing one layer at a time and assigns extra precision to the largest scores. Interactions make this only a heuristic: two individually safe layers can amplify each other when quantized together.\n\nTherefore the procedure has two stages—rank under a fixed probe, then assemble and retest the complete assignment. Storage, kernel compatibility, and latency must also be recalculated because mixed formats can add dispatch boundaries.",
        "baseline": "six-layer floating-point MLP and an all-INT4 candidate",
        "candidate": "INT8 for the two most sensitive layers, INT4 for the remaining four",
        "controlled": "equal layer sizes, calibration input, quantizers, two-layer fallback budget",
        "metrics": "per-layer isolated RMSE, selected layers, average weight bits, assembled output error",
        "code_walk": "The notebook quantizes each of six equal-size matrices to INT4 and INT8. It replaces one layer at a time to measure sensitivity against the full-precision network, selects the top two, constructs the mixed model, and re-evaluates end to end.\n\nBecause layers are equal size, the budget is transparent: `(2×8 + 4×4)/6 = 5.333` bits per weight. A real transformer would weight layers by parameter count and backend-compatible groupings.",
        "result_reading": "Layers 0 and 1 had the highest isolated RMSE, 0.0013786 and 0.00130424, so they received INT8. The mixed assignment used 5.333 average bits and produced assembled RMSE 0.00248394 with cosine 0.976198.\n\nThe assembled error is larger than any isolated score, demonstrating interaction across layers. The ranking still gives a reproducible budgeted candidate, but whether it beats all-INT4 or another allocation must be judged with a frozen quality target and actual storage/runtime measurements.",
        "failure": "Selecting fallback layers on the final task set overfits deployment evaluation. Comparing mixed-bit quality without reporting average bits is unfair. Backend fragmentation can also erase theoretical benefit if INT4 and INT8 layers use incompatible packing or force synchronization/materialization.",
        "next": "Add all-INT4 and all-INT8 assembled baselines, search several budgets, and plot quality versus effective bytes. Repeat sensitivity on multiple domains and sequence lengths. Then run a backend that supports the mixed formats and measure operator boundaries, memory, and latency.",
        "extra_refs": [
            ("GPTQ paper", "https://arxiv.org/abs/2210.17323"),
            ("AWQ paper", "https://arxiv.org/abs/2306.00978"),
        ],
    },
    27: {
        "hook": "A quantized artifact is not ready when conversion finishes; it is ready when a versioned candidate passes frozen gates and a tested rollback path exists. Release decisions should be deterministic from evidence, so the same manifest produces the same promote-or-rollback result rather than depending on operator optimism.",
        "checks": [
            "Predict whether the candidate passes a 10% latency gate and an RMSE≤0.5 gate.",
            "Explain why both gates are required even when latency improves.",
            "List the additional evidence needed before changing `rollback` to a live canary.",
        ],
        "derivation": "A release manifest binds candidate and baseline revisions, environment, quantization recipe, quality thresholds, performance SLOs, owners, observability, canary fraction, and rollback target. Each gate evaluates a named artifact; the decision is the conjunction for critical gates, not an average score.\n\nRollback must restore a loadable, compatible baseline and be rehearsed before promotion. A local synthetic decision can validate the gate machinery while remaining explicit that no container, traffic, or service health signal was exercised.",
        "baseline": "versioned BF16 matrix path `bf16-v1`",
        "candidate": "reference INT4-dequantized path `reference-int4-v1`",
        "controlled": "same tensors, fifteen timing samples, fixed RMSE/latency thresholds",
        "metrics": "baseline/candidate median and p90, output error, individual gate booleans, release decision",
        "code_walk": "The notebook measures both paths, computes error, evaluates two predeclared booleans, and writes a manifest whose decision is `promote_to_canary` only if all gates pass. The rollback target is stored even when the candidate fails.\n\nThis is a deterministic release-policy test. It is not a container build, model-card audit, shadow deployment, or canary against live traffic.",
        "result_reading": "Candidate median latency was 0.018848 ms versus 0.019360 ms for the baseline, so the ≤10% regression gate passed. But output RMSE was 5.317538, far above the 0.5 threshold, so the quality gate failed and the manifest selected `rollback`.\n\nA small speed improvement cannot compensate for a failed critical quality gate. The result illustrates why release criteria must be conjunctive and frozen before the candidate is observed.",
        "failure": "Changing thresholds after seeing the result converts a gate into a justification. A rollback identifier without a verified artifact is not a rollback plan. Production promotion also needs sustained load, error rates, GPU health, output monitoring, and a human/operator decision path.",
        "next": "Package baseline and candidate into pinned containers, validate cold load and warm restart, run an offline quality suite and shadow traffic, then perform a small canary with automated rollback triggers. Rehearse the rollback and record recovery time before expanding traffic.",
        "extra_refs": [
            ("Hugging Face model cards", "https://huggingface.co/docs/hub/model-cards"),
            ("NVIDIA Triton model management", "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html"),
        ],
    },
    28: {
        "hook": "Cloud cost begins with a memory feasibility ledger, but it cannot end there. Ideal weight bits, unquantized layers, scale metadata, KV cache per request, workspace, fragmentation, tensor parallelism, throughput, utilization, and hourly price all determine whether one GPU is usable and economical.",
        "checks": [
            "Estimate ideal BF16 and INT4 weight GiB for 70B parameters.",
            "Compute one-request KV cache for 80 layers, 8 KV heads, dimension 128, and 8K context.",
            "Predict whether ideal INT4 weights fit a 32,607 MiB RTX 5090 after a 10% reserve.",
        ],
        "derivation": "Weight bytes start at `P·bits/8`. KV bytes per request are `2·L·S·Hkv·D·cache_bytes`, then concurrency multiplies that term. A safety reserve should cover kernels, graph capture, allocator behavior, and unexpected peaks before dividing remaining bytes by per-request cache.\n\nEven a memory fit does not produce a cost result. Cost per million tokens depends on achieved tokens/s, utilization, batching, power/cloud price, failure rate, and replica count. The notebook intentionally stops at arithmetic capacity when no engine throughput exists.",
        "baseline": "70B BF16 weights with BF16 KV cache",
        "candidate": "ideal INT4 weights with BF16 or INT8 KV cache",
        "controlled": "70B parameters, 80 layers, 8 KV heads, head dimension 128, context 8192, 10% reserve",
        "metrics": "live total/free GiB, weight GiB, KV GiB/request, fit boolean, projected request count",
        "code_walk": "The notebook reads live RTX 5090 memory, calculates three plans, reserves 10%, and only then computes request capacity. It records zero rather than a negative or optimistic concurrency when weights already exceed usable memory.\n\nThe INT4 term is explicitly ideal: it excludes scales, padding, embeddings/norms retained in higher precision, engine, and workspace. That label prevents the arithmetic from being mistaken for a successful model load.",
        "result_reading": "Live total memory was 31.358 GiB. BF16 weights projected to 130.385 GiB; ideal INT4 still required 32.596 GiB, already larger than total memory and larger still relative to the 10% reserve. BF16 KV cache was 2.5 GiB/request and INT8 KV 1.25 GiB/request, but every single-GPU plan correctly returned zero requests because weights did not fit.\n\nKV compression cannot rescue a base model that fails the weight-fit gate. A real 70B deployment therefore needs further compression/overhead reduction, multi-GPU sharding, CPU offload, or a different GPU class before concurrency is discussed.",
        "failure": "Using decimal GB instead of binary GiB can create misleading margin near capacity. Ideal four-bit arithmetic omits metadata and high-precision tensors, and free memory on an otherwise empty process is not engine capacity. Cost comparisons without throughput and quality at equal SLO are also meaningless.",
        "next": "Add measured overhead from a real engine, tensor-parallel sharding/communication, fragmentation, and batch-dependent workspaces. Once the model loads, benchmark sustained tokens/s and compute cost per million tokens at equal quality and p95 latency across candidate GPU plans.",
        "extra_refs": [
            ("vLLM cache configuration", "https://docs.vllm.ai/en/stable/api/vllm/config/cache/"),
            ("CUDA Programming Guide", "https://docs.nvidia.com/cuda/cuda-programming-guide/index.html"),
        ],
    },
    29: {
        "hook": "A custom INT4 kernel earns its complexity only if it removes work from the end-to-end path. Packing weights is helpful, but materializing a full dequantized matrix before calling BF16 GEMM adds reads, writes, conversions, and launches. The target is a fused load–unpack–scale–MMA–epilogue path with a supported tile layout.",
        "checks": [
            "Calculate packed bytes for a 4096×4096 INT4 weight matrix.",
            "Predict the latency of a composed unpack/dequant/GEMM path relative to direct BF16 GEMM at M=32.",
            "Name the evidence needed before calling an implementation a CUTLASS or custom-kernel result.",
        ],
        "derivation": "The logical pipeline is packed global load → nibble extraction/sign extension → scale load → dequantized fragments → matrix multiply/accumulate → epilogue. If dequantization writes a full BF16 matrix to global memory, the path pays both packed reads and a large materialized write/read before GEMM. Fusion keeps reconstructed values in registers/fragments and amortizes scale work across a tile.\n\nKernel profitability depends on M, N, K, group size, memory coalescing, register pressure, occupancy, and epilogue fusion. A semantic PyTorch composition is a correctness baseline and an upper-bound warning, not a custom kernel.",
        "baseline": "direct BF16 GEMM for shape M=32, K=N=4096",
        "candidate": "PyTorch-composed unpack, sign restore, dequantization, and GEMM",
        "controlled": "same X/W values, group size 128, packed layout, GPU timing helper",
        "metrics": "packed bytes, BF16 median/p90, composed median/p90, implementation identity",
        "code_walk": "The notebook packs two codes per byte, reconstructs signed codes, applies block scales, materializes BF16 weights, and multiplies. It times the complete composed function rather than timing only the final GEMM. The result field explicitly says `not fused CUTLASS`.\n\nThis gives a readable semantic reference for testing a future CUDA/Triton/CUTLASS implementation. The future kernel must match its outputs while eliminating materialization and reducing launches.",
        "result_reading": "Packed storage was 8,388,608 bytes for 16,777,216 weights, exactly 0.5 byte per code before scales. Direct BF16 GEMM took 0.027136 ms median. The composed unpack/dequant/matmul path took 0.328720 ms—about 12.1x slower.\n\nThe slowdown is not evidence that INT4 hardware is slow. It is evidence that the unfused reference performs too much integration work and memory traffic. It establishes the optimization target and a correctness oracle.",
        "failure": "Timing only GEMM after pre-dequantizing outside the measurement hides the dominant cost. Calling a Python composition a custom kernel is false. A fused kernel can also regress if register pressure lowers occupancy or if unsupported shapes fall back, so shape coverage and dispatch must be audited.",
        "next": "Implement a minimal fused kernel in CUTLASS, CUDA, or Triton for one frozen shape. Verify packed-layout compatibility and numerical parity, then profile instruction mix, global bytes, occupancy, tensor-pipe utilization, and end-to-end latency across M values. Add a safe fallback for unsupported shapes.",
        "extra_refs": [
            ("CUTLASS documentation", "https://docs.nvidia.com/cutlass/latest/overview.html"),
            ("CUTLASS repository", "https://github.com/NVIDIA/cutlass"),
            ("CUDA Programming Guide", "https://docs.nvidia.com/cuda/cuda-programming-guide/index.html"),
        ],
    },
    30: {
        "hook": "A serviceable 70B INT4 project is a sequence of gates, not a conversion command. Weight fit enables engine work; engine identity enables quality and load testing; passing quality, SLO, capacity, observability, canary, and rollback gates enables production. Any unexecuted critical gate keeps the decision at `not ready`.",
        "checks": [
            "Predict whether ideal 70B INT4 weights fit after reserve on the recorded RTX 5090.",
            "Evaluate which deployment gates can be answered by a toy mixed-bit matrix and which require a real engine.",
            "Write the minimum reversal conditions that would move the final decision toward canary.",
        ],
        "derivation": "The gate graph begins with immutable model/recipe identity and capacity arithmetic. It then requires a supported backend build and operator trace, frozen quality suite, representative service load, cost/capacity margin, observability, owner, canary plan, and tested rollback. Dependencies matter: service SLO is undefined before a loadable engine exists.\n\nA toy mixed-bit probe can validate the idea of fallback and a numeric threshold, but it cannot answer 70B task quality. Likewise, ideal `P/2` bytes ignores scale metadata and unquantized layers. Marking those distinctions in the final decision is part of the deliverable.",
        "baseline": "BF16 rollback concept and unexecuted production gates",
        "candidate": "ideal 70B INT4 capacity plus a toy mixed-bit numerical probe",
        "controlled": "live GPU memory, 70B parameter count, 10% reserve, fixed toy threshold",
        "metrics": "ideal weight GiB, fit boolean, toy RMSE/cosine, six gate booleans, final decision",
        "code_walk": "The notebook reads live capacity, computes ideal INT4 bytes, runs a small CUDA mixed-bit matrix probe, and builds a gate dictionary. It sets engine, quality-suite, and service-SLO gates false because those experiments were not run. The final decision is derived from all gates rather than written optimistically.\n\nThis makes the notebook an executable deployment-plan skeleton. It is not a 70B load test, quantized checkpoint, or cost benchmark.",
        "result_reading": "Ideal INT4 weights were 32.596 GiB versus 31.358 GiB total GPU memory, so single-GPU weight fit failed before metadata or reserve. The toy mixed-bit probe produced RMSE 3.720175, above its threshold of 2, although cosine was 0.993368. Rollback identity was defined, but engine build, quality suite, and service SLO were all false. The derived decision was `not_ready_for_service`.\n\nThis is the correct outcome: arithmetic compression and one toy probe cannot fill missing production evidence. The gate matrix tells the next engineer exactly what remains rather than converting absence into a success claim.",
        "failure": "Calling ideal weight fit a successful load ignores the largest uncertainty. Allowing one high cosine score to override task failures also weakens the gate graph. A plan without owners, artifacts, deadlines, observability, and rollback rehearsal may be complete on paper but unusable during an incident.",
        "next": "Select a feasible multi-GPU or larger-memory target, build a pinned native engine, and capture layer/operator evidence. Run the frozen quality suite and representative load, fill capacity/cost margins, define monitoring and owners, then rehearse rollback. Only all-passing critical gates should change the decision to canary-ready.",
        "extra_refs": [
            ("NVIDIA Model Optimizer documentation", "https://nvidia.github.io/Model-Optimizer/"),
            ("TensorRT-LLM documentation", "https://nvidia.github.io/TensorRT-LLM/"),
            ("vLLM benchmark CLI", "https://docs.vllm.ai/en/latest/cli/bench/serve.html"),
        ],
    },
}


RESULT_SPECS: dict[int, list[tuple[str, tuple[Any, ...], str]]] = {
    2: [
        ("Aligned BF16 median", ("timings", "aligned", "bfloat16", "median_ms"), "ms"),
        ("Aligned FP32 median", ("timings", "aligned", "float32", "median_ms"), "ms"),
        ("Awkward BF16 median", ("timings", "awkward", "bfloat16", "median_ms"), "ms"),
        ("Awkward FP32 median", ("timings", "awkward", "float32", "median_ms"), "ms"),
        ("Recorded samples per case", ("timings", "aligned", "bfloat16", "repeats"), "int"),
    ],
    3: [
        ("Initial loss", ("history", 0, "loss"), "float"),
        ("Final loss", ("history", -1, "loss"), "float"),
        ("Autocast output dtype", ("history", 0, "output_dtype"), "text"),
        ("All recorded gradients finite", ("history", -1, "grads_finite"), "bool"),
        ("Parameter dtype", ("parameter_dtype",), "text"),
        ("Final scaler value", ("history", -1, "scale"), "float"),
    ],
    4: [
        ("FP16 represents 1e5", ("range_probe", "fp16_1e5_finite"), "bool"),
        ("BF16 represents 1e5", ("range_probe", "bf16_1e5_finite"), "bool"),
        ("FP16 RMSE", ("formats", "float16", "error", "rmse"), "float"),
        ("BF16 RMSE", ("formats", "bfloat16", "error", "rmse"), "float"),
        ("FP16 median", ("formats", "float16", "timing", "median_ms"), "ms"),
        ("BF16 median", ("formats", "bfloat16", "timing", "median_ms"), "ms"),
        ("FP32 median", ("formats", "float32", "timing", "median_ms"), "ms"),
    ],
    5: [
        ("1e-8, scale 1: zero fraction", ("gradient_sweep", 0, "zero_fraction"), "pct"),
        ("1e-8, scale 256: zero fraction", ("gradient_sweep", 1, "zero_fraction"), "pct"),
        ("1, scale 65536: Inf fraction", ("gradient_sweep", 8, "inf_fraction"), "pct"),
        ("1000, scale 256: Inf fraction", ("gradient_sweep", 10, "inf_fraction"), "pct"),
        ("Forward 1e5 overflowed", ("forward_overflow_at_1e5",), "bool"),
    ],
    6: [
        ("GEMM shape", ("shape",), "shape"),
        ("Median", ("timing", "median_ms"), "ms"),
        ("p90", ("timing", "p90_ms"), "ms"),
        ("Samples", ("timing", "repeats"), "int"),
        ("PyTorch operator events", ("pytorch_operator_events",), "list"),
    ],
    7: [
        ("BF16 KV at 2,048 tokens", ("projected_kv", 0, "bf16_gib"), "gib"),
        ("BF16 KV at 8,192 tokens", ("projected_kv", 1, "bf16_gib"), "gib"),
        ("BF16 KV at 32,768 tokens", ("projected_kv", 2, "bf16_gib"), "gib"),
        ("INT8 KV at 32,768 tokens", ("projected_kv", 2, "int8_gib"), "gib"),
        ("Live allocation probe", ("allocation_probe", "bytes"), "bytes"),
    ],
    8: [
        ("Group 16 RMSE", ("group_results", 0, "error", "rmse"), "float"),
        ("Group 16 effective bits", ("group_results", 0, "effective_bits_per_weight"), "bits"),
        ("Group 64 RMSE", ("group_results", 1, "error", "rmse"), "float"),
        ("Group 64 effective bits", ("group_results", 1, "effective_bits_per_weight"), "bits"),
        ("Group 128 RMSE", ("group_results", 2, "error", "rmse"), "float"),
        ("Group 128 effective bits", ("group_results", 2, "effective_bits_per_weight"), "bits"),
    ],
    9: [
        ("Narrow clipping", ("calibration_results", "narrow", "clipping_fraction"), "pct"),
        ("Narrow RMSE", ("calibration_results", "narrow", "error", "rmse"), "float"),
        ("Balanced clipping", ("calibration_results", "balanced", "clipping_fraction"), "pct"),
        ("Balanced RMSE", ("calibration_results", "balanced", "error", "rmse"), "float"),
        ("Outlier-aware clipping", ("calibration_results", "outlier_aware", "clipping_fraction"), "pct"),
        ("Outlier-aware RMSE", ("calibration_results", "outlier_aware", "error", "rmse"), "float"),
    ],
    10: [
        ("Alpha 0 RMSE", ("alpha_sweep", 0, "output_error", "rmse"), "float"),
        ("Alpha 0.25 RMSE", ("alpha_sweep", 1, "output_error", "rmse"), "float"),
        ("Alpha 0.5 RMSE", ("alpha_sweep", 2, "output_error", "rmse"), "float"),
        ("Alpha 0.75 RMSE", ("alpha_sweep", 3, "output_error", "rmse"), "float"),
        ("Alpha 1 RMSE", ("alpha_sweep", 4, "output_error", "rmse"), "float"),
        ("Worst floating equivalence error", ("alpha_sweep", 2, "float_equivalence_max_abs"), "float"),
    ],
    11: [
        ("Naive INT4 RMSE", ("naive_output_error", "rmse"), "float"),
        ("Sensitivity fallback RMSE", ("sensitivity_fallback_error", "rmse"), "float"),
        ("Naive cosine", ("naive_output_error", "cosine"), "float"),
        ("Fallback cosine", ("sensitivity_fallback_error", "cosine"), "float"),
        ("Preserved columns", ("preserved_column_fraction",), "pct"),
    ],
    12: [
        ("Selected alpha", ("best_alpha",), "float"),
        ("Alpha 0 RMSE", ("alpha_sweep", 0, "heldout_error", "rmse"), "float"),
        ("Alpha 0.25 RMSE", ("alpha_sweep", 1, "heldout_error", "rmse"), "float"),
        ("Alpha 0.5 RMSE", ("alpha_sweep", 2, "heldout_error", "rmse"), "float"),
        ("Alpha 1 RMSE", ("alpha_sweep", 4, "heldout_error", "rmse"), "float"),
    ],
    13: [
        ("7B BF16 base", ("seven_b_ledger", "bf16_base_gib"), "gib"),
        ("7B ideal INT4 base", ("seven_b_ledger", "int4_base_ideal_gib"), "gib"),
        ("LoRA trainable state", ("seven_b_ledger", "lora_trainable_mib"), "mib"),
        ("Adam states", ("seven_b_ledger", "adam_states_mib"), "mib"),
        ("Base frozen", ("base_requires_grad",), "inverse_bool"),
        ("Adapter gradients finite", ("adapter_grad_finite",), "bool"),
    ],
    14: [
        ("NF4 RMSE", ("nf4_error", "rmse"), "float"),
        ("Uniform INT4 RMSE", ("uniform_int4_error", "rmse"), "float"),
        ("NF4 max error", ("nf4_error", "max_abs"), "float"),
        ("Uniform INT4 max error", ("uniform_int4_error", "max_abs"), "float"),
        ("bitsandbytes installed", ("bitsandbytes_installed",), "bool"),
    ],
    15: [
        ("TorchAO installed", ("torchao", "torchao_installed"), "bool"),
        ("Conversion status", ("torchao", "conversion"), "text"),
        ("Failure type", ("torchao", "error_type"), "text"),
        ("Failure message", ("torchao", "error_message"), "text"),
    ],
    16: [
        ("Weight shape", ("shape",), "shape"),
        ("Group size", ("group_size",), "int"),
        ("Packed code bytes", ("packed_bytes",), "bytes"),
        ("Exact pack/unpack", ("codes_exact_after_unpack",), "bool"),
        ("Q/DQ RMSE", ("qdq_error", "rmse"), "float"),
        ("TensorRT installed", ("tensorrt_installed",), "bool"),
    ],
    17: [
        ("Manifest complete", ("manifest_complete",), "bool"),
        ("Format / group", ("manifest", "recipe", "format"), "text"),
        ("Group size", ("manifest", "recipe", "group_size"), "int"),
        ("ModelOpt handoff", ("manifest", "handoff", "modelopt"), "bool"),
        ("TensorRT-LLM handoff", ("manifest", "handoff", "tensorrt_llm"), "bool"),
        ("Numerical RMSE", ("numerical_probe", "rmse"), "float"),
    ],
    18: [
        ("vLLM installed", ("vllm_installed",), "bool"),
        ("Service benchmark", ("vllm_service_benchmark",), "text"),
        ("Batch 1 BF16 median", ("pytorch_shape_warning", 0, "bf16", "median_ms"), "ms"),
        ("Batch 1 reference W4 median", ("pytorch_shape_warning", 0, "reference_w4_dequant", "median_ms"), "ms"),
        ("Batch 32 BF16 median", ("pytorch_shape_warning", 2, "bf16", "median_ms"), "ms"),
        ("Batch 32 reference W4 median", ("pytorch_shape_warning", 2, "reference_w4_dequant", "median_ms"), "ms"),
    ],
    19: [
        ("BF16 cache", ("bf16_bytes",), "bytes"),
        ("INT8 cache plus scales", ("int8_plus_scale_bytes",), "bytes"),
        ("Memory reduction", ("memory_reduction_pct",), "raw_pct"),
        ("Attention-output RMSE", ("attention_output_error", "rmse"), "float"),
        ("Attention-output cosine", ("attention_output_error", "cosine"), "float"),
    ],
    20: [
        ("PyTorch API", ("probe", "api"), "text"),
        ("FP8 GEMM", ("probe", "fp8_gemm"), "text"),
        ("Median", ("probe", "fp8_timing", "median_ms"), "ms"),
        ("FP8 RMSE", ("probe", "fp8_error", "rmse"), "float"),
        ("Transformer Engine installed", ("probe", "transformer_engine_installed"), "bool"),
        ("NVFP4 backend", ("probe", "nvfp4_backend"), "text"),
    ],
    21: [
        ("Patch weight shape", ("patch_projection", "weight_shape"), "shape"),
        ("Normal-image RMSE", ("domain_errors", "normal", "rmse"), "float"),
        ("High-contrast RMSE", ("domain_errors", "high_contrast", "rmse"), "float"),
        ("Normal max error", ("domain_errors", "normal", "max_abs"), "float"),
        ("High-contrast max error", ("domain_errors", "high_contrast", "max_abs"), "float"),
    ],
    22: [
        ("Manifest complete", ("manifest_complete",), "bool"),
        ("Payload bytes", ("manifest", "bytes"), "bytes"),
        ("Format", ("manifest", "format"), "text"),
        ("Group size", ("manifest", "group_size"), "int"),
        ("SHA-256", ("manifest", "sha256"), "short_hash"),
        ("Temporary payload removed", ("temporary_payload_deleted_after_check",), "bool"),
    ],
    23: [
        ("Baseline loss", ("synthetic_probe", "baseline_loss"), "float"),
        ("Candidate loss", ("synthetic_probe", "candidate_loss"), "float"),
        ("Baseline synthetic perplexity", ("synthetic_probe", "baseline_perplexity"), "sci"),
        ("Candidate synthetic perplexity", ("synthetic_probe", "candidate_perplexity"), "sci"),
        ("Top-1 agreement", ("synthetic_probe", "top1_agreement"), "pct"),
    ],
    24: [
        ("Batch 1 median", ("operator_workload", 0, "timing", "median_ms"), "ms"),
        ("Batch 1 throughput", ("operator_workload", 0, "examples_per_second"), "rate"),
        ("Batch 32 median", ("operator_workload", 2, "timing", "median_ms"), "ms"),
        ("Batch 32 throughput", ("operator_workload", 2, "examples_per_second"), "rate"),
        ("Batch 128 median", ("operator_workload", 3, "timing", "median_ms"), "ms"),
        ("Batch 128 throughput", ("operator_workload", 3, "examples_per_second"), "rate"),
        ("Batch 128 peak allocation", ("operator_workload", 3, "peak_allocated_mib"), "mib"),
    ],
    25: [
        ("Ordinary RMSE", ("failure_matrix", 0, "output_error", "rmse"), "float"),
        ("Small-batch RMSE", ("failure_matrix", 1, "output_error", "rmse"), "float"),
        ("Activation-outlier RMSE", ("failure_matrix", 2, "output_error", "rmse"), "float"),
        ("Shifted-domain RMSE", ("failure_matrix", 3, "output_error", "rmse"), "float"),
        ("Largest shifted max error", ("failure_matrix", 3, "output_error", "max_abs"), "float"),
    ],
    26: [
        ("INT8 fallback layers", ("int8_fallback_layers",), "list"),
        ("Average weight bits", ("average_weight_bits",), "bits"),
        ("Layer 0 isolated RMSE", ("layer_sensitivity", 0, "rmse"), "float"),
        ("Layer 1 isolated RMSE", ("layer_sensitivity", 1, "rmse"), "float"),
        ("Assembled RMSE", ("assembled_output_error", "rmse"), "float"),
        ("Assembled cosine", ("assembled_output_error", "cosine"), "float"),
    ],
    27: [
        ("Baseline median", ("baseline_timing", "median_ms"), "ms"),
        ("Candidate median", ("candidate_timing", "median_ms"), "ms"),
        ("Candidate RMSE", ("output_error", "rmse"), "float"),
        ("Latency gate", ("release_manifest", "gates", "latency_regression_lte_10pct"), "bool"),
        ("Quality gate", ("release_manifest", "gates", "rmse_lte_0_5"), "bool"),
        ("Decision", ("release_manifest", "decision"), "text"),
    ],
    28: [
        ("Live total memory", ("live_memory", "total_gib"), "gib"),
        ("BF16 weight projection", ("plans", "bf16_weights_bf16_kv", "weight_gib"), "gib"),
        ("Ideal INT4 weight projection", ("plans", "int4_ideal_weights_bf16_kv", "weight_gib"), "gib"),
        ("BF16 KV per request", ("plans", "int4_ideal_weights_bf16_kv", "kv_per_request_gib"), "gib"),
        ("INT8 KV per request", ("plans", "int4_ideal_weights_int8_kv", "kv_per_request_gib"), "gib"),
        ("Ideal INT4 single-GPU fit", ("plans", "int4_ideal_weights_int8_kv", "single_gpu_weight_fit"), "bool"),
    ],
    29: [
        ("Shape M×K×N", ("shape_mkn",), "shape"),
        ("Packed code bytes", ("packed_bytes",), "bytes"),
        ("BF16 median", ("bf16_timing", "median_ms"), "ms"),
        ("Composed path median", ("composed_unpack_dequant_matmul", "median_ms"), "ms"),
        ("Implementation", ("implementation",), "text"),
    ],
    30: [
        ("Live GPU total", ("live_gpu_total_gib",), "gib"),
        ("Ideal INT4 weights", ("ideal_int4_weight_gib",), "gib"),
        ("Single-GPU ideal fit", ("deployment_gates", "single_gpu_ideal_weight_fit"), "bool"),
        ("Toy mixed-bit RMSE", ("toy_mixed_bit_error", "rmse"), "float"),
        ("Quality suite passed", ("deployment_gates", "quality_suite_passed"), "bool"),
        ("Service SLO passed", ("deployment_gates", "service_slo_passed"), "bool"),
        ("Decision", ("decision",), "text"),
    ],
}


def nested_get(data: Any, path: tuple[Any, ...]) -> Any:
    value = data
    for key in path:
        value = value[key]
    return value


def display_value(value: Any, kind: str) -> str:
    if kind == "ms":
        return f"{float(value):.6f} ms"
    if kind == "float":
        return f"{float(value):.6f}"
    if kind == "pct":
        return f"{float(value) * 100:.4f}%"
    if kind == "raw_pct":
        return f"{float(value):.4f}%"
    if kind == "gib":
        return f"{float(value):.3f} GiB"
    if kind == "mib":
        return f"{float(value):.3f} MiB"
    if kind == "bytes":
        return f"{int(value):,} bytes"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "rate":
        return f"{float(value):,.2f} examples/s"
    if kind == "bits":
        return f"{float(value):.3f} bits/weight"
    if kind == "sci":
        return f"{float(value):.3e}"
    if kind == "bool":
        return "yes" if bool(value) else "no"
    if kind == "inverse_bool":
        return "yes" if not bool(value) else "no"
    if kind == "shape":
        return " × ".join(str(x) for x in value)
    if kind == "list":
        return ", ".join(str(x) for x in value)
    if kind == "short_hash":
        text = str(value)
        return f"`{text[:8]}…{text[-6:]}`"
    return str(value)


def result_table(no: int, artifact: dict[str, Any]) -> str:
    rows = ["| Measured field | Checked-in value |", "|---|---:|"]
    for label, path, kind in RESULT_SPECS[no]:
        rows.append(f"| {label} | {display_value(nested_get(artifact, path), kind)} |")
    return "\n".join(rows)


def environment_line(artifact: dict[str, Any]) -> str:
    env = artifact["environment"]
    return (
        f"{env['gpu']}; compute capability {env['compute_capability']}; "
        f"PyTorch {env['torch']}; CUDA runtime {env['cuda_runtime']}"
    )
