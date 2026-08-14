# Reproducing the Delphi ONNX evaluation

This evaluation answers two separate questions:

1. Does the actual JavaScript/ONNX Runtime Web pipeline reproduce the PyTorch checkpoint before stochastic sampling?
2. What are model size, startup time, steady-state step latency, fixed-length trajectory latency, and memory use on the tested desktop?

The primary fidelity endpoints are browser Wasm and browser WebGPU. ONNX Runtime Python is retained as an intermediate export diagnostic. Sampling is excluded from fidelity because the Python and JavaScript random-number generators are different; the deterministic logits, masked event distributions, and total event rate immediately before sampling are compared instead. Sampling remains enabled for full-trajectory latency.

## Fixed inputs and integrity checks

The evaluator uses the locally supplied synthetic-data model in `OriginalModel.pt`, not the inaccessible checkpoint from the original Delphi-2M paper. The checkpoint is untracked and must be supplied separately; its recorded hash below identifies the exact file used. The evaluator downloads the upstream synthetic `val.bin` from Delphi commit `fb72166be6b29d8db819227a59487e51c1df1454` when no `--val-bin` is supplied, and refuses to continue if its SHA-256 is not:

```text
f57f6a63e339f0c3643709f80443a75ecb05850986a24b91fb2d1910c1d11484
```

The measured artifacts were:

```text
OriginalModel.pt  cbab7a70a811f7d3383162e3fc1ff38fe994103a98447a6eb874e131d78a6a87
delphi.onnx       94c64a0b0ccee973c7b9a127b404f5dd202cf48e6eaa9d2c6bba9171585b37e4
```

The fixed evaluation design is:

- final-position logits for every complete held-out trajectory (7,143 cases; 9,071,610 logits per runtime);
- every position in 256 cases selected with seed 1337, using 32 cases from each sex-by-raw-length-quartile stratum;
- every position in one authored edge case containing lifestyle events, no-event records, and tied ages; and
- 30 fixed real validation contexts at each length 12, 24, and 48, cycled identically by all runtimes for step latency, with benchmark seed 20260813.

One upstream patient (427985) is excluded because the byte-level train/validation boundary splits that trajectory. The compact selected-ID list is tracked in `fixtures/stratified_patient_ids.json`; the generated manifest additionally records the exact inputs, hashes, seeds, output-reference offsets, and acceptance criteria.

## Environment setup

The recorded run used macOS 26.6.1 on a MacBook Pro Mac15,6 with an Apple M3 Pro (11-core CPU, 14-core GPU) and 18 GiB unified memory, Python 3.9.6, PyTorch 2.8.0, ONNX Runtime Python 1.19.2, ONNX Runtime Web 1.27.0, and Google Chrome 151.0.7922.137. Wasm was explicitly restricted to one thread. WebGPU used the default browser power preference and was the only configured execution provider, although ONNX Runtime internally assigned shape/control nodes to CPU; no fallback execution provider was configured. PyTorch used CPU because MPS was unavailable.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r evaluation/requirements.txt
npm ci
```

The browser runner currently points to the standard macOS Chrome installation at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`. Change the `CHROME` constant in `run_browser_evaluation.mjs` when reproducing on a machine with another executable location. A WebGPU-capable Chrome session is required for the WebGPU arm.

## Run the evaluation

Generate the fixed cohort, PyTorch reference tensors, Python ONNX Runtime diagnostic, and native latency measurements:

```bash
.venv/bin/python evaluation/run_python_evaluation.py
```

To use an already-downloaded validation file:

```bash
.venv/bin/python evaluation/run_python_evaluation.py --val-bin /absolute/path/to/val.bin
```

Run both browser backends, including 10 fresh-browser cold trials, memory, fidelity, preprocessing, determinism, 200 timed executions per context length, and 30 full trajectories:

```bash
npm run evaluate:browser
```

Useful narrower reruns are:

```bash
.venv/bin/python evaluation/run_python_evaluation.py --latency-only
node evaluation/run_browser_evaluation.mjs --backend wasm
node evaluation/run_browser_evaluation.mjs --backend webgpu
node evaluation/run_browser_evaluation.mjs --memory-only
```

The full fidelity run evaluates more than 21 million logits per browser backend and can take several minutes. Do not compare timings collected while another intensive job is using the machine.

Regenerate the publication report and compact CSV tables from the raw JSON and manifest:

```bash
npm run evaluate:report
```

## Outputs

- `generated/manifest.json`: exact cohort, selected IDs, inputs, hashes, seeds, and acceptance criteria.
- `generated/references/*.f32`: little-endian float32 PyTorch reference logits.
- `results/python_fidelity.json`: Python ONNX Runtime conversion diagnostic and legacy-mask diagnostic.
- `results/python_latency.json`: PyTorch CPU and Python ONNX Runtime CPU timings.
- `results/browser_wasm.json` and `results/browser_webgpu.json`: primary browser fidelity, preprocessing, determinism, startup, artifact-size, latency, and an isolated memory trial. The report prefers this embedded, provenance-matched memory record after a full run.
- `results/browser_memory_*.json`: results from narrower `--memory-only` runs. The report uses these only when an embedded current trial is unavailable.
- `results/fidelity_table.csv`, `results/latency_table.csv`, and `results/startup_size_memory_table.csv`: compact exact-value tables.
- `PUBLICATION_REPORT.md`: publication-ready Methods, Results, reviewer responses, limitations, and readable result tables.

`generated/` and the cached upstream data are intentionally ignored by Git because the reference tensors are large and deterministic to regenerate. The compact result JSON, CSV tables, and report are suitable for version control.

## Interpretation boundary

The evaluation establishes numerical equivalence for this repository’s synthetic checkpoint and actual browser inference paths at the pre-sampling boundary. It does not validate the unavailable original Delphi-2M checkpoint. The latency result applies to one desktop configuration only; no mobile device was measured. Accordingly, manuscript claims should report the measured values and avoid generalized “near-native,” mobile-performance, or cross-device-speed language.
