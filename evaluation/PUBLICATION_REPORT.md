# Delphi ONNX fidelity and performance evaluation

## Executive result

The browser port reproduced the PyTorch reference to substantially better than the declared probability-level tolerance. Across all 7,143 complete held-out trajectories, ONNX Runtime Web Wasm had logit mean absolute error (MAE) 2.880e-06 and maximum absolute error 3.052e-05; WebGPU had MAE 5.897e-06 and maximum 1.640e-04. Every one of 9,071,610 logits per backend satisfied `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. The largest conditional event-probability errors were 1.143e-06 (Wasm) and 6.794e-06 (WebGPU), maximum total-variation distances were 1.272e-06 and 8.397e-06, and top-1 agreement was 100% for both.

Performance was usable but backend- and context-dependent, so the manuscript should replace an unqualified “near-native” claim with measurements. A fixed 16-event prefix plus 32 generated events took a median 112.242 ms in PyTorch CPU, 149.660 ms in browser Wasm (1.33× PyTorch), and 176.450 ms in browser WebGPU (1.57×). These results apply to one desktop only; they do not establish mobile or general cross-device performance.

## Methods

### Model and reference implementation

We evaluated the locally supplied synthetic-data checkpoint `OriginalModel.pt`, not the inaccessible checkpoint used in the published Delphi-2M study. The checkpoint is intentionally not committed to this repository; its SHA-256 was `cbab7a70a811f7d3383162e3fc1ff38fe994103a98447a6eb874e131d78a6a87`, which identifies the exact local input needed to reproduce this run. It contained 2,243,400 parameters (12 layers, 12 attention heads, embedding dimension 120, vocabulary size 1,270, and maximum context 48). Evaluation mode disabled dropout. The tested ONNX artifact SHA-256 was `94c64a0b0ccee973c7b9a127b404f5dd202cf48e6eaa9d2c6bba9171585b37e4` and its output was the raw `[batch, position, event]` logit tensor immediately before stochastic event/time sampling.

The source implementation and validation data were pinned to [gerstung-lab/Delphi commit `fb72166b`](https://github.com/gerstung-lab/Delphi/tree/fb72166be6b29d8db819227a59487e51c1df1454). This comparison therefore establishes equivalence for the synthetic checkpoint in this repository; it cannot establish equivalence to the unavailable published model.

### Fixed evaluation cohort

The upstream synthetic validation file `val.bin` was downloaded from the pinned commit and verified against SHA-256 `f57f6a63e339f0c3643709f80443a75ecb05850986a24b91fb2d1910c1d11484`. Patient 427985 was excluded because the upstream byte-level split bisected that trajectory, leaving only an 11-row validation fragment. The resulting primary cohort contained 7,143 complete held-out trajectories and 181,282 raw records: 3,661 female and 3,482 male trajectories, 245 with tied event ages, and 1,859 ending in death. Raw trajectory length ranged from 15 to 69 events (median 24, IQR 19–30). Processed input length ranged from 25 to 48 (median 39).

The primary cohort test compared the final-position logits for all 7,143 trajectories (9,071,610 logits per runtime). To exercise intermediate positions and the hand-written browser pipeline, a second fixed set used NumPy seed 1337 to select 32 trajectories without replacement from each sex-by-raw-length-quartile stratum (2 sexes × 4 quartiles × 32 = 256), plus one authored case containing lifestyle tokens, repeated no-event tokens, and tied ages. This yielded 257 cases, 10,178 evaluated positions, and 12,926,060 logits per runtime. The resolved patient IDs are stored in the generated manifest.

### Preprocessing and probability comparison

Reference contexts followed the pinned upstream fixed-width inference path: the rightmost 49 raw records were placed into a left-masked window; regular no-event records were inserted at day 1 and every 1,826.25 days through the last observed age; the combined records were ordered by age with `torch.argsort`; raw token IDs were increased by one; and the rightmost 49 combined records were retained. The final record was reserved as the prediction target, leaving at most 48 input records. Reproducing the fixed-width intermediate array also reproduced upstream ordering for exact-age ties. Tokens were `int64` and ages were IEEE-754 `float32` days.

The browser check then exercised the public SDK input boundary on the 257 all-position cases: it supplied event names and ages in years to `prepareTrajectoryInputs`, then compared the resulting token IDs and float32 day values with the reference tensors. JavaScript and Python had zero mismatches in 10,178 token cells, zero bit mismatches in 10,178 float32 age cells, and zero event-name mismatches after mapping the tokens back to names, on both browser backends. The SDK does not consume raw `val.bin`; raw cohort-window construction therefore remained a reference-data preparation step, while the SDK-specific tokenization, age conversion, tensor construction, masking, inference, output readback, and event-name mapping were exercised in JavaScript.

Sampling was intentionally excluded from fidelity testing. At each evaluated position, the harness invoked the SDK pre-sampling mask and verified its masked cells against checkpoint `ignore_tokens=[0,2,...,12]` and the previously observed event tokens. It then independently derived and compared (i) raw logits, (ii) the conditional event distribution obtained by normalizing `exp(logit)` over unmasked events, and (iii) `logsumexp(logit)` over unmasked events, which is the log total event rate implied at the exponential-race sampling boundary. Ten repeated fixed-input browser inferences were bitwise identical on each backend. Benchmark seed 20260813 controlled stochastic trajectory timing, but sampled trajectories were not treated as a numerical-fidelity endpoint.

### Acceptance criteria

The executable protocol recorded the following criteria before aggregating results:

- every logit must satisfy `abs(error) <= atol + rtol * abs(reference)`, with `atol=0.0001` and `rtol=0.0001`;
- maximum per-context total-variation distance, maximum absolute event-probability error, and maximum absolute log-total-rate error must each be no greater than `0.0001`; and
- top-1 event agreement must equal 100%.

The combined absolute-plus-relative logit rule matters for interpreting the WebGPU maximum absolute error (1.640e-04): it exceeds `1e-4` in isolation but occurred on a logit of magnitude about 20 and passed the declared combined tolerance. Probability- and rate-level maximum errors remained below `1e-4` without a relative term.

### Performance protocol

Measurements were made on 13 August 2026 on a MacBook Pro Mac15,6 running macOS 26.6.1, with Apple M3 Pro (11-core CPU, 14-core GPU) and 18 GiB unified memory, while connected to AC power. PyTorch 2.8.0 used CPU because MPS was unavailable, with 5 intra-op and 11 inter-op threads. Browser measurements used headless Google Chrome 151.0.7922.137 and ONNX Runtime Web 1.27.0. Wasm was explicitly limited to one thread. WebGPU was the only configured execution provider and used the browser's default (not explicitly set) power preference; ONNX Runtime nevertheless assigned shape/control nodes internally to CPU, as reported by its runtime diagnostics, rather than falling back to a second configured provider. The Python ONNX Runtime 1.19.2 CPU results are reported only as a conversion diagnostic, not as the deployed endpoint.

For each context length (12, 24, and 48), each runtime cycled through the same 30 fixed validation contexts. Each measured endpoint had 10 warm-up executions followed by 200 timed executions. “Model-only” included prebuilt tensors, inference, and output materialization. Browser “SDK end-to-end step” additionally included JavaScript array-to-tensor construction and final-position extraction; model-only and SDK calls were paired with alternating order to limit order and thermal bias. Full-trajectory latency used the same fixed 16-event prefix in Python and JavaScript and comprised exactly 32 sampled events with termination disabled, plus a final all-position inference: 33 graph executions total, with 2 warm-ups and 30 measured runs. Cold initialization and first-inference latency used 10 fresh Chrome processes per backend; first inference used a fixed context of length 34. Runtime, model, and label artifacts were served without caching over loopback. The initialization timer began after the page and SDK module loaded, and therefore captures runtime/model/label fetching, parsing, graph compilation, and session creation—but not HTML navigation, SDK-module loading, or real Internet transfer time.

Memory was measured in a separate browser trial. In-page memory used `performance.measureUserAgentSpecificMemory()`. A second sampler excluded every pre-existing Chrome PID, then summed resident-set size (RSS) every 100 ms across newly appearing processes whose command contained “Google Chrome”; it reports the increment above the dedicated trial's idle page. This is not a parent/child process-tree traversal: unrelated Chrome processes that started during the trial could contaminate the total, and summed RSS can double-count shared pages. Unified-memory macOS also does not expose an isolated GPU-memory peak. These values are feasibility proxies rather than precise model-only allocations.

## Results

### Fidelity: primary browser endpoint

| Backend and scope | Contexts | Logits | Logit MAE | Max abs logit error | Within combined tolerance | Max abs probability error | Max TV distance | Max abs log-total-rate error | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wasm, all final positions | 7,143 | 9,071,610 | 2.880e-06 | 3.052e-05 | 100.00% | 1.143e-06 | 1.272e-06 | 4.857e-06 | 100.00% |
| WebGPU, all final positions | 7,143 | 9,071,610 | 5.897e-06 | 1.640e-04 | 100.00% | 6.794e-06 | 8.397e-06 | 3.517e-05 | 100.00% |
| Wasm, stratified all positions | 10,178 | 12,926,060 | 2.839e-06 | 3.052e-05 | 100.00% | 1.052e-06 | 1.199e-06 | 4.887e-06 | 100.00% |
| WebGPU, stratified all positions | 10,178 | 12,926,060 | 2.878e-06 | 1.469e-04 | 100.00% | 3.811e-06 | 6.716e-06 | 2.155e-05 | 100.00% |

No browser context exceeded any probability- or rate-level acceptance threshold, and there were zero masking-cell mismatches. Mean top-5 overlap was 99.9972% for Wasm (one boundary-rank difference across the cohort) and 100% for WebGPU; mean top-10 overlap was 100% for both. The pre-evaluation SDK default omitted the checkpoint-level ignored-token mask. That legacy setting changed the PyTorch conditional distribution by mean TV 4.158e-05 and maximum TV 6.382e-03; the SDK was corrected to use `[0,2,...,12]`. This demonstrates why pipeline-level validation was necessary even though the ONNX graph itself was already close.

### Fidelity: intermediate Python ONNX Runtime diagnostic

| Scope | Logit MAE | Max abs logit error | Within combined tolerance | Max abs probability error | Max TV distance | Max abs log-total-rate error | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All final positions | 2.858e-06 | 3.052e-05 | 100.00% | 9.755e-07 | 1.237e-06 | 4.923e-06 | 100.00% |
| Stratified all positions | 2.856e-06 | 3.052e-05 | 100.00% | 9.453e-07 | 1.189e-06 | 4.861e-06 | 100.00% |

This diagnostic is consistent with the remaining browser differences arising from backend floating-point variation rather than an export failure. It is not substituted for the browser endpoint in the primary result.

### Steady-state single-step latency

Values are median ms (IQR); p95 ms. Each cell summarizes 200 runs after 10 warm-ups.

| Runtime | Context 12 | Context 24 | Context 48 |
|---|---:|---:|---:|
| PyTorch CPU (reference) | 2.522 (2.501–2.563); p95 2.706 | 2.906 (2.882–2.949); p95 3.043 | 3.943 (3.912–4.002); p95 4.075 |
| ORT Python CPU (diagnostic) | 0.546 (0.542–0.551); p95 0.623 | 0.900 (0.896–0.905); p95 1.031 | 1.619 (1.612–1.628); p95 1.874 |
| ORT Web Wasm, 1 thread | 1.770 (1.745–1.815); p95 1.905 | 3.240 (3.220–3.275); p95 3.440 | 6.570 (6.535–6.611); p95 6.721 |
| ORT Web WebGPU | 4.850 (4.740–5.156); p95 6.362 | 4.910 (4.735–5.139); p95 5.612 | 4.775 (4.600–4.968); p95 8.189 |

Browser SDK end-to-end step latency, including tensor construction and final-position extraction, was:

| Runtime | Context 12 | Context 24 | Context 48 |
|---|---:|---:|---:|
| ORT Web Wasm, 1 thread | 1.785 (1.750–1.820); p95 1.945 | 3.250 (3.225–3.280); p95 3.471 | 6.582 (6.550–6.630); p95 6.751 |
| ORT Web WebGPU | 4.895 (4.775–5.180); p95 6.331 | 4.895 (4.735–5.111); p95 5.756 | 4.765 (4.619–4.986); p95 8.263 |

At context 48, browser model-only latency was 1.67× PyTorch for Wasm and 1.21× for WebGPU. WebGPU was 27.3% faster than Wasm at this length, but its fixed dispatch/readback overhead made it slower at lengths 12 and 24. The Python ONNX Runtime values are implementation diagnostics and should not be used to characterize browser performance.

### Full-trajectory latency

| Runtime | Runs | Median ms (IQR) | p95 ms | Ratio to PyTorch |
|---|---:|---:|---:|---:|
| PyTorch CPU | 30 | 112.242 (111.711–113.075) | 120.895 | 1.00× |
| ORT Web Wasm, 1 thread | 30 | 149.660 (147.350–152.236) | 154.141 | 1.33× |
| ORT Web WebGPU | 30 | 176.450 (168.076–195.814) | 221.484 | 1.57× |

### Cold initialization, artifact size, and memory

| Backend | Session initialization, median (IQR); p95 ms | First inference, median (IQR); p95 ms | Model MB | Raw first-use payload MB |
|---|---:|---:|---:|---:|
| Wasm | 263.875 (257.005–268.010); p95 288.516 | 46.857 (35.756–48.724); p95 49.462 | 9.758 | 23.669 |
| WebGPU | 462.947 (457.253–483.067); p95 520.198 | 81.312 (71.562–86.935); p95 89.349 | 9.758 | 37.033 |

The ONNX model itself was 9,758,087 bytes (9.758 decimal MB). The raw first-use totals include the model, 17,970-byte SDK, 362,615-byte labels, and the backend-specific runtime JavaScript and Wasm binary; they are uncompressed file sizes, not content-encoded network transfer sizes. Across the 10 WebGPU cold trials, the maximum first-inference time was 89.515 ms, compared with median 81.312 ms and p95 89.349 ms.

The 60 browser steady-state warm-up executions (10 model-only and 10 SDK end-to-end calls at each of the three context lengths) took 249.905 ms in Wasm and 346.320 ms in WebGPU in the full benchmark sessions. These totals are reported for procedural completeness; the fresh-browser length-34 first-inference distribution better characterizes one-time warm-up cost because the 60-execution totals mix two endpoints and three context lengths.

| Backend | In-page after session MB | In-page after rollout MB | Idle Chrome RSS MB | Peak Chrome RSS MB | Incremental peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Wasm | 44.916 | 45.155 | 924.778 | 1269.072 | 344.293 |
| WebGPU | 38.179 | 38.576 | 928.465 | 1581.449 | 652.984 |

## Publication-ready manuscript text

### Methods insertion

> We assessed numerical fidelity against the PyTorch implementation using a locally supplied synthetic-data checkpoint (`OriginalModel.pt`, SHA-256 `cbab7a70a811f7d3383162e3fc1ff38fe994103a98447a6eb874e131d78a6a87`); the original Delphi-2M paper checkpoint was unavailable, and the local checkpoint is not committed to the conversion repository. We pinned the upstream synthetic validation data and preprocessing code to Delphi commit `fb72166b` and verified the validation file by SHA-256. After excluding one patient whose trajectory was bisected by the upstream byte-level train/validation split, the test cohort comprised 7,143 complete held-out trajectories. Preprocessing reproduced the upstream fixed-width path: it placed the rightmost 49 raw records in a left-masked window, inserted a no-event token every 5 years, ordered the combined records by age using `torch.argsort` (including its exact-age tie behavior), shifted token IDs by one, retained the rightmost 49 combined records, reserved the final record as the target, and supplied at most 48 input records as int64 tokens and float32 ages in days. We compared final-position logits for every trajectory and all-position logits for a fixed seed-1337 stratified subset of 256 trajectories (32 per sex-by-length-quartile stratum) plus one authored lifestyle/no-event edge case. Because sampling is stochastic, fidelity was evaluated immediately before sampling. After applying the checkpoint’s ignored-token and previously-seen-event masks, we compared raw logits, conditional event probabilities (masked softmax), and log total event rate (masked log-sum-exp). Acceptance required all logits to satisfy `|Δ| <= 10^-4 + 10^-4|reference|`, maximum event-probability error, total-variation distance, and log-total-rate error no greater than `10^-4`, and 100% top-1 agreement. The actual ONNX Runtime Web Wasm and WebGPU paths were the primary endpoints; ONNX Runtime Python was an intermediate conversion diagnostic.
>
> Performance was measured on one MacBook Pro Mac15,6 with Apple M3 Pro (11-core CPU, 14-core GPU) and 18 GiB unified memory using PyTorch 2.8.0 CPU, ONNX Runtime Web 1.27.0 Wasm (one thread), and WebGPU in headless Chrome 151. At each context length (12, 24, and 48), all runtimes cycled through the same 30 fixed validation contexts; each endpoint had 10 warm-up executions followed by 200 timed executions. We recorded model-only latency (prebuilt tensors through materialized output), browser SDK end-to-end step latency (including tensor construction and final-position extraction), and 30 complete 32-event rollouts from the same 16-event prefix (33 graph executions, after two warm-ups). Ten fresh-browser trials measured session initialization and length-34 first-inference latency. Artifact sizes are raw uncompressed bytes. Memory was estimated both in-page and as the increment above an idle page in sampled summed RSS of newly launched Chrome processes from a separate trial.

### Results insertion

> ONNX Runtime Web reproduced the PyTorch reference within the declared tolerance. Across 9,071,610 final-position logits, Wasm and WebGPU logit MAE values were 2.880e-06 and 5.897e-06, with maximum absolute errors 3.052e-05 and 1.640e-04, respectively; 100% satisfied the combined `atol=rtol=10^-4` criterion. Maximum conditional event-probability errors were 1.143e-06 and 6.794e-06, maximum total-variation distances were 1.272e-06 and 8.397e-06, maximum log-total-rate errors were 4.857e-06 and 3.517e-05, and top-1 agreement was 100% for both. The 10,178-position stratified test likewise passed all criteria. JavaScript and Python preprocessing agreed exactly in all 10,178 token and age cells, with no event-name postprocessing mismatches. This evaluation identified and corrected an SDK default mask that had omitted checkpoint-ignored tokens.
>
> On the tested desktop, median model-only latency at context lengths 12/24/48 was 2.522/2.906/3.943 ms for PyTorch CPU, 1.770/3.240/6.570 ms for browser Wasm, and 4.850/4.910/4.775 ms for browser WebGPU. A 32-event rollout took a median 112.242 ms in PyTorch, 149.660 ms in Wasm (1.33×), and 176.450 ms in WebGPU (1.57×). The ONNX artifact was 9.758 MB; raw first-use payloads were 23.669 MB (Wasm) and 37.033 MB (WebGPU). Median cold session initialization/length-34 first-inference times were 263.875/46.857 ms for Wasm and 462.947/81.312 ms for WebGPU; 60 browser steady-state warm-up executions took 249.905 and 346.320 ms. Incremental peak newly launched Chrome-process RSS was 344.293 MB and 652.984 MB, respectively. These measurements demonstrate sub-200-ms median 32-event rollouts on the tested desktop, but not an unqualified “near-native” or mobile-performance claim.

## Draft response to reviewers

### Comment 1: fidelity

Thank you; we agree that conversion alone was insufficient evidence of correctness. We added a deterministic, end-to-end comparison against the PyTorch implementation using the synthetic-data checkpoint, since the checkpoint from the original Delphi-2M publication is not accessible. The primary endpoint is the actual JavaScript/ONNX Runtime Web path, not Python ONNX Runtime. We compared 9,071,610 final-position logits from all 7,143 complete held-out validation trajectories and 12,926,060 all-position logits from a fixed stratified set of 256 trajectories plus one authored lifestyle/no-event edge case. Sampling was excluded so that both implementations could be compared immediately before the stochastic boundary; instead, we compared the resulting masked event-probability distributions and log total event rates.

For Wasm, logit MAE/max absolute error were 2.880e-06/3.052e-05; for WebGPU they were 5.897e-06/1.640e-04. All logits satisfied `|Δ| <= 10^-4 + 10^-4|reference|`. Maximum probability error, total-variation distance, and log-total-rate error were 1.143e-06, 1.272e-06, and 4.857e-06 for Wasm, and 6.794e-06, 8.397e-06, and 3.517e-05 for WebGPU. Top-1 agreement was 100% for both. JavaScript preprocessing matched Python exactly in every tested token and float32 age cell, and event-name postprocessing had zero mismatches. Importantly, the exercise exposed an SDK default-mask discrepancy, which we corrected; we now describe and test the checkpoint’s ignored-token mask explicitly. We added these methods, quantitative results, tolerances, fixed seeds, and reproducibility artifacts to the manuscript and repository.

### Comment 2: performance

Thank you; we replaced qualitative performance assertions with measurements and limited our inference to the tested desktop. On an Apple M3 Pro, browser model-only latency at context lengths 12/24/48 was 1.770/3.240/6.570 ms for one-thread Wasm and 4.850/4.910/4.775 ms for WebGPU, compared with 2.522/2.906/3.943 ms for PyTorch CPU. A fixed 32-event rollout required median 112.242 ms in PyTorch, 149.660 ms in Wasm, and 176.450 ms in WebGPU. We also report 10-trial cold initialization and length-34 first-inference times, 60-execution browser warm-up totals, the 9.758-MB model, backend-specific raw first-use payloads of 23.669 and 37.033 MB, and memory proxies.

Mobile measurements were outside the scope of this revision. We therefore removed/qualified “near-native speeds” and any wording that implies measured performance across mobile and desktop devices. The revised text states that portability is an architectural property of browser deployment, while the quantitative performance evidence is from one desktop configuration only.

## Limitations and claims boundary

- The evaluated checkpoint was newly trained on synthetic data and is not the inaccessible model from the original Delphi-2M paper.
- `OriginalModel.pt` is locally supplied and not committed here. Its SHA-256 makes this evaluation input identifiable, but independent reproduction requires depositing that checkpoint in an accessible archive.
- Fidelity is established for raw logits and their deterministic probability/rate transformation before sampling. It does not assert identity of stochastic sampled paths across different random-number-generator implementations.
- Only one desktop, browser version, operating system, and power state were measured. No mobile result or broad cross-device speed conclusion is supported.
- Chrome was headless and artifacts were loopback-served without caching. Real download time depends on hosting, compression, cache state, and network conditions.
- WebGPU timing includes output readback and can be dominated by dispatch overhead for this small model and short contexts.
- Summed RSS over newly launched Chrome processes is a conservative proxy that may double-count shared pages or include an unrelated process launched during the trial; the WebGPU value does not isolate GPU allocation on unified-memory hardware.
- Timing results are snapshots, not universal constants; thermal state, browser updates, background work, and hardware can change them.

## Reproducibility artifacts

The exact underlying measurements are in `evaluation/results/*.json`; compact machine-readable tables are `fidelity_table.csv`, `latency_table.csv`, and `startup_size_memory_table.csv`. The evaluation manifest records data/model hashes, fixed case IDs, preprocessing, seeds, reference tensor locations, and acceptance criteria. See `evaluation/README.md` for commands. Re-run `python3 evaluation/make_publication_report.py` after new measurements to regenerate this document and all three CSV files.
