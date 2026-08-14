## Benchmarking: Fidelity, latency

### Table 1. Evaluation host, software environment, and provenance

| Property | Value |
|---|---|
| Machine | MacBook Pro (Mac15,6) |
| Processor | Apple M3 Pro (11-core CPU, 14-core GPU) |
| Total memory | 18.0 GB (19,327,352,832 bytes) |
| OS | Darwin 25.6.0, arm64 |
| Browser | Headless Chrome 151.0.7922.137 |
| ONNX Runtime Web version | 1.27.0 |
| ONNX Runtime (Python) version | 1.19.2 |
| PyTorch version | 2.8.0 |
| Python version | 3.9.6 |
| Wasm threads (browser) | 1 (single-threaded; page was cross-origin-isolated and thus eligible for multi-threaded Wasm, but this was not enabled for the present benchmark) |
| Cross-origin isolated / secure context | Yes / Yes |

**Provenance (SHA-256 checksums, shared across all benchmark runs reported here):**

| Artifact | SHA-256 (abbreviated) |
|---|---|
| Evaluation manifest | `85848e5f…` |
| Upstream Delphi-2M commit | `fb72166b…` |
| Model checkpoint | `cbab7a70…` |
| ONNX model file | `94c64a0b…` |
| JavaScript SDK | `f3d6af7b…` |
| Label file | `e797c1eb…` |

*All benchmark files (Tables 2–9) share these identical provenance values, confirming that browser (Wasm/WebGPU), native PyTorch, and native ONNX Runtime results all refer to the same model version and code state.*

---

### Table 2. Client-side artifact payload size (download footprint), by backend

| Component | Wasm backend | WebGPU backend |
|---|---|---|
| ONNX model file | 9.76 MB | 9.76 MB |
| SDK (JavaScript) | 17.4 KB | 17.4 KB |
| Label file | 362 KB | 362 KB |
| ONNX Runtime Web (JS glue) | 50.1 KB | 67.2 KB |
| ONNX Runtime Web (Wasm binary) | 12.9 MB | 25.6 MB |
| **Total first-use payload** | **23.7 MB** | **37.0 MB** |

*The WebGPU backend requires a larger ONNX Runtime Web binary, roughly doubling first-use download size relative to the Wasm-only backend.*

---

### Table 3. Startup latency, by backend

| Metric | Wasm | WebGPU |
|---|---|---|
| Session initialization, warm (single run) | 250.1 ms | 451.0 ms |
| Session initialization, cold (median of 10 trials) | 263.9 ms [Q1 257.0, Q3 268.0, p95 288.5] | 462.9 ms [Q1 457.3, Q3 483.1, p95 520.2] |
| First inference after cold init (median of 10 trials) | 46.9 ms [Q1 35.8, Q3 48.7, p95 49.5] | 81.3 ms [Q1 71.6, Q3 86.9, p95 89.3] |

---

### Table 4. Steady-state per-step inference latency, by context length and runtime

Median latency in milliseconds (n = 200 runs per cell), by context length in tokens.

| Context length | Wasm (browser) | WebGPU (browser) | PyTorch CPU (native, eager) | ONNX Runtime CPU (native, Python) |
|---|---|---|---|---|
| 12 | 1.77 | 4.85 | 2.52 | 0.55 |
| 24 | 3.24 | 4.91 | 2.91 | 0.90 |
| 48 | 6.57 | 4.78 | 3.94 | 1.62 |

*Note the WebGPU backend's per-step latency is approximately flat across context length in this range (dispatch/readback overhead dominates at this model scale), whereas both Wasm and native CPU paths scale with context length as expected for attention cost. The native ONNX Runtime (Python) path is fastest at every context length, reflecting optimizations not available through the WebAssembly compilation target.*

---

### Table 5. Full-trajectory generation latency

16-event input prefix extended by 32 sampled events (33 total model graph executions per trajectory), n = 30 trajectories.

| Runtime | Median (ms) | Q1 (ms) | Q3 (ms) | p95 (ms) | Ratio to native PyTorch CPU (median) |
|---|---|---|---|---|---|
| Wasm (browser) | 149.7 | 147.4 | 152.2 | 154.1 | 1.33× |
| WebGPU (browser) | 176.5 | 168.1 | 195.8 | 221.5 | 1.57× |
| PyTorch CPU (native, eager) | 112.2 | 111.7 | 113.1 | 120.9 | 1.00× (reference) |

---

### Table 6. Memory usage, by backend

| Metric | Wasm | WebGPU |
|---|---|---|
| In-page memory after session initialization | 44.9 MB | 38.2 MB |
| In-page memory after 10 warm inferences | 45.1 MB | 38.5 MB |
| In-page memory after a full 32-event trajectory rollout | 45.2 MB | 38.6 MB |
| Chrome process idle RSS | 924.8 MB | 926.9 MB |
| Chrome process peak RSS | 1,269.1 MB | 1,557.3 MB |
| Chrome process incremental peak RSS (attributable to the trial) | 344.3 MB | 630.4 MB |

*Process-level RSS is a coarse proxy: peak summed RSS of Chrome processes launched for the dedicated trial, sampled every 100 ms. This may double-count shared pages, assumes no unrelated Chrome process activity during the trial, and does not isolate GPU memory on unified-memory macOS systems. In-page, JS-heap-attributable memory (rows 1–3) is the more directly interpretable figure and shows only modest (~0.1–0.3 MB) growth across a 32-event generation rollout, i.e., no evidence of substantial per-step memory growth at this trajectory length.*

---

### Table 7. Fidelity — logit-level agreement, browser (ONNX Runtime Web) vs. reference PyTorch

| Comparison | Backend | Contexts / logits compared | Mean abs. error | Max abs. error | Tolerance (atol, rtol) | Fraction within tolerance |
|---|---|---|---|---|---|---|
| Final-position logits, all held-out trajectories | Wasm | 7,143 contexts / 9,071,610 logits | 2.9 × 10⁻⁶ | 3.1 × 10⁻⁵ | 1×10⁻⁴, 1×10⁻⁴ | 100% |
| Final-position logits, all held-out trajectories | WebGPU | 7,143 contexts / 9,071,610 logits | 5.9 × 10⁻⁶ | 1.6 × 10⁻⁴ | 1×10⁻⁴, 1×10⁻⁴ | 100% |
| All-position logits, stratified subset | Wasm | 10,178 contexts / 12,926,060 logits | 2.8 × 10⁻⁶ | 3.1 × 10⁻⁵ | 1×10⁻⁴, 1×10⁻⁴ | 100% |
| All-position logits, stratified subset | WebGPU | 10,178 contexts / 12,926,060 logits | 2.9 × 10⁻⁶ | 1.5 × 10⁻⁴ | 1×10⁻⁴, 1×10⁻⁴ | 100% |

*WebGPU shows a slightly larger maximum absolute error than Wasm, consistent with expected floating-point reduction-order differences on GPU; all values remain within the combined absolute+relative tolerance.*

---

### Table 8. Fidelity — output-distribution agreement, browser vs. reference

Event-probability distributions from which the SDK's time-to-event sampler draws.

| Comparison | Backend | Contexts | Mean total variation distance | Max total variation distance | Top-1 agreement | Mean top-5 overlap | Mean top-10 overlap |
|---|---|---|---|---|---|---|---|
| Final-position distributions | Wasm | 7,143 | 6.5 × 10⁻⁷ | 1.3 × 10⁻⁶ | 100% | 99.997% | 100% |
| Final-position distributions | WebGPU | 7,143 | 1.4 × 10⁻⁶ | 8.4 × 10⁻⁶ | 100% | 100% | 100% |
| All-position distributions, stratified | Wasm | 10,178 | 4.9 × 10⁻⁷ | 1.2 × 10⁻⁶ | 100% | 100% | 100% |
| All-position distributions, stratified | WebGPU | 10,178 | 6.1 × 10⁻⁷ | 6.7 × 10⁻⁶ | 100% | 100% | 100% |

*Top-1 agreement of 100% means the browser and reference implementations select the identical next event under greedy decoding in every case tested. Zero mask-cell mismatches were observed in any comparison (i.e., the set of valid next-event tokens matched exactly between implementations).*

---

### Table 9. Preprocessing exactness and generation determinism

| Check | Result |
|---|---|
| Token cells compared (preprocessing) | 10,178 |
| Token mismatches | 0 |
| Age-encoding cells compared | 10,178 |
| Float32-bit-level age-encoding mismatches | 0 |
| Postprocessed event-name mismatches | 0 |
| Repeated end-to-end generation runs (fixed seed) | 10 |
| Bitwise-identical runs | 10 / 10 |
| Maximum absolute difference across repeated runs | 0 |

**Diagnostic note.** During development of this evaluation harness, an earlier SDK version was found to use a default token-masking configuration that differed from the checkpoint's configured ignore-token list. This produced a mean total variation distance of 4.2 × 10⁻⁵ (maximum 6.4 × 10⁻³) relative to the reference distributions — a real, silent divergence between the hand-reimplemented JavaScript pipeline and the reference implementation. This was identified and corrected in the SDK prior to the results reported in Tables 7–9 above.


*Corresponding raw benchmark files (JSON) and the full evaluation manifest are available in the SDK repository, https://github.com/episphere/delphi-onnx, alongside the checksums listed in Table 1.*
