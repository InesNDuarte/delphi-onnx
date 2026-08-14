#!/usr/bin/env python3
"""Regenerate compact tables and publication text from the raw evaluation JSON."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
RESULTS = EVALUATION / "results"
GENERATED = EVALUATION / "generated"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def ms(value: float) -> str:
    return f"{value:.3f}"


def interval(summary: Dict[str, Any]) -> str:
    return (
        f"{ms(summary['median_ms'])} "
        f"({ms(summary['q1_ms'])}–{ms(summary['q3_ms'])}); "
        f"p95 {ms(summary['p95_ms'])}"
    )


def mb(value: int) -> str:
    return f"{value / 1_000_000:.3f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def fidelity_rows(
    python_fidelity: Dict[str, Any],
    wasm: Dict[str, Any],
    webgpu: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    comparisons = [
        (
            "primary_browser_endpoint",
            "ONNX Runtime Web Wasm",
            wasm["fidelity"],
            True,
        ),
        (
            "primary_browser_endpoint",
            "ONNX Runtime Web WebGPU",
            webgpu["fidelity"],
            True,
        ),
        (
            "conversion_diagnostic",
            "ONNX Runtime Python CPU",
            python_fidelity["pytorch_vs_python_onnxruntime"],
            False,
        ),
    ]
    scopes = [
        ("all_cohort_final", "all_cohort_final_logits", "all_cohort_final_distributions"),
        (
            "stratified_all_positions",
            "stratified_all_position_logits",
            "stratified_all_position_distributions",
        ),
    ]
    for role, backend, comparison, is_browser in comparisons:
        for scope, logits_key, distributions_key in scopes:
            logits = comparison[logits_key]
            distributions = comparison[distributions_key]
            tv = distributions["total_variation_distance"]
            js = distributions["jensen_shannon_divergence"]
            rate = distributions["absolute_log_total_rate_error"]
            failures = distributions.get("acceptance_failures", {})
            rows.append(
                {
                    "comparison_role": role,
                    "backend": backend,
                    "scope": scope,
                    "contexts": distributions["contexts"],
                    "logit_count": logits["count"],
                    "logit_mean_absolute_error": logits["mean_absolute_error"],
                    "logit_root_mean_square_error": logits["root_mean_square_error"],
                    "logit_median_absolute_error": logits.get(
                        "median_absolute_error_approx", logits.get("median_absolute_error")
                    ),
                    "logit_p95_absolute_error": logits.get(
                        "p95_absolute_error_approx", logits.get("p95_absolute_error")
                    ),
                    "logit_p99_absolute_error": logits.get(
                        "p99_absolute_error_approx", logits.get("p99_absolute_error")
                    ),
                    "logit_maximum_absolute_error": logits["maximum_absolute_error"],
                    "logit_atol": logits["atol"],
                    "logit_rtol": logits["rtol"],
                    "logit_within_tolerance_fraction": logits["within_tolerance_fraction"],
                    "probability_mean_absolute_error": distributions[
                        "mean_absolute_probability_error"
                    ],
                    "probability_maximum_absolute_error": distributions[
                        "maximum_absolute_probability_error"
                    ],
                    "total_variation_mean": tv["mean"],
                    "total_variation_median": tv["median"],
                    "total_variation_p95": tv["p95"],
                    "total_variation_maximum": tv["maximum"],
                    "jensen_shannon_mean": js["mean"],
                    "jensen_shannon_maximum": js["maximum"],
                    "absolute_log_total_rate_error_mean": rate["mean"],
                    "absolute_log_total_rate_error_maximum": rate["maximum"],
                    "top1_agreement": distributions["top1_agreement"],
                    "mean_top5_overlap": distributions["mean_top5_overlap"],
                    "mean_top10_overlap": distributions["mean_top10_overlap"],
                    "mask_cell_mismatches": distributions.get("mask_cell_mismatches", ""),
                    "total_variation_acceptance_failures": failures.get(
                        "total_variation_over_1e_4", "" if not is_browser else 0
                    ),
                    "probability_acceptance_failures": failures.get(
                        "event_probability_error_over_1e_4", "" if not is_browser else 0
                    ),
                    "log_total_rate_acceptance_failures": failures.get(
                        "log_total_rate_error_over_1e_4", "" if not is_browser else 0
                    ),
                }
            )
    return rows


def latency_rows(
    python_latency: Dict[str, Any], wasm: Dict[str, Any], webgpu: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pytorch = python_latency["results"]["pytorch_cpu"]
    native_ort = python_latency["results"]["onnxruntime_python_cpu"]
    baselines = {
        str(length): pytorch["steady_state"][str(length)]["median_ms"]
        for length in (12, 24, 48)
    }

    def add(
        role: str,
        runtime: str,
        endpoint: str,
        length: str,
        warmups: int,
        summary: Dict[str, Any],
        ratio: Any,
        includes_sampling: bool,
        generated_events: Any = "",
        graph_executions: Any = "",
        fixed_contexts: int = 30,
    ) -> None:
        rows.append(
            {
                "comparison_role": role,
                "runtime": runtime,
                "endpoint": endpoint,
                "context_length": length,
                "warmups": warmups,
                "measured_runs": summary["n"],
                "median_ms": summary["median_ms"],
                "q1_ms": summary["q1_ms"],
                "q3_ms": summary["q3_ms"],
                "p95_ms": summary["p95_ms"],
                "mean_ms": summary["mean_ms"],
                "median_ratio_to_pytorch_cpu": ratio,
                "fixed_contexts_cycled": fixed_contexts,
                "includes_sampling": str(includes_sampling).lower(),
                "generated_events": generated_events,
                "graph_executions": graph_executions,
            }
        )

    for length in (12, 24, 48):
        key = str(length)
        add(
            "reference",
            "PyTorch CPU",
            "model_only",
            key,
            python_latency["config"]["warmups"],
            pytorch["steady_state"][key],
            1.0,
            False,
        )
        add(
            "diagnostic",
            "ONNX Runtime Python CPU",
            "model_only",
            key,
            python_latency["config"]["warmups"],
            native_ort["steady_state"][key],
            native_ort["steady_state"][key]["median_ms"] / baselines[key],
            False,
        )
        for runtime, record in (
            ("ONNX Runtime Web Wasm", wasm),
            ("ONNX Runtime Web WebGPU", webgpu),
        ):
            browser = record["latency"]
            add(
                "deployed_pipeline",
                runtime,
                "model_only",
                key,
                browser["warmup_runs"],
                browser["model_only"][key],
                browser["model_only"][key]["median_ms"] / baselines[key],
                False,
            )
            add(
                "deployed_pipeline",
                runtime,
                "sdk_end_to_end_step",
                key,
                browser["warmup_runs"],
                browser["end_to_end_step"][key],
                "",
                False,
            )

    pytorch_trajectory = pytorch["full_trajectory"]
    add(
        "reference",
        "PyTorch CPU",
        "full_trajectory",
        "16+32",
        2,
        pytorch_trajectory,
        1.0,
        True,
        pytorch_trajectory["generated_events"],
        pytorch_trajectory["graph_executions"],
        1,
    )
    for runtime, record in (
        ("ONNX Runtime Web Wasm", wasm),
        ("ONNX Runtime Web WebGPU", webgpu),
    ):
        trajectory = record["latency"]["full_trajectory"]
        add(
            "deployed_pipeline",
            runtime,
            "full_trajectory",
            "16+32",
            2,
            trajectory,
            trajectory["median_ms"] / pytorch_trajectory["median_ms"],
            True,
            trajectory["generated_events"],
            trajectory["graph_executions"],
            1,
        )
    return rows


def startup_rows(
    wasm: Dict[str, Any],
    webgpu: Dict[str, Any],
    wasm_memory: Dict[str, Any],
    webgpu_memory: Dict[str, Any],
    cold_context_length: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record, memory_record in ((wasm, wasm_memory), (webgpu, webgpu_memory)):
        cold = record["cold_start"]
        artifacts = record["artifacts_bytes"]
        process_memory = memory_record["process_memory"]
        snapshots = {
            item["label"]: item
            for item in process_memory["in_page"]["user_agent_specific_memory"]
        }
        rows.append(
            {
                "backend": record["backend"],
                "cold_trials": cold["session_initialization"]["n"],
                "cold_first_inference_context_length": cold_context_length,
                "session_initialization_median_ms": cold["session_initialization"][
                    "median_ms"
                ],
                "session_initialization_q1_ms": cold["session_initialization"]["q1_ms"],
                "session_initialization_q3_ms": cold["session_initialization"]["q3_ms"],
                "session_initialization_p95_ms": cold["session_initialization"]["p95_ms"],
                "session_initialization_mean_ms": cold["session_initialization"]["mean_ms"],
                "first_inference_median_ms": cold["first_inference"]["median_ms"],
                "first_inference_q1_ms": cold["first_inference"]["q1_ms"],
                "first_inference_q3_ms": cold["first_inference"]["q3_ms"],
                "first_inference_p95_ms": cold["first_inference"]["p95_ms"],
                "first_inference_mean_ms": cold["first_inference"]["mean_ms"],
                "steady_state_warmups_per_context_length": record["latency"]["warmup_runs"],
                "steady_state_warmup_executions_total": 6
                * record["latency"]["warmup_runs"],
                "steady_state_warmup_total_ms": record["latency"]["warmup_total_ms"],
                "model_bytes": artifacts["model"],
                "sdk_bytes": artifacts["sdk"],
                "labels_bytes": artifacts["labels"],
                "runtime_javascript_bytes": artifacts["runtime_javascript"],
                "runtime_binary_bytes": artifacts["runtime_wasm"],
                "first_use_payload_bytes": artifacts["first_use_payload"],
                "in_page_after_session_initialization_bytes": snapshots[
                    "after_session_initialization"
                ]["bytes"],
                "in_page_after_32_event_rollout_bytes": snapshots[
                    "after_32_event_rollout"
                ]["bytes"],
                "chrome_idle_rss_bytes": process_memory["idle_rss_bytes"],
                "chrome_peak_rss_bytes": process_memory["peak_rss_bytes"],
                "chrome_incremental_peak_rss_bytes": process_memory[
                    "incremental_peak_rss_bytes"
                ],
                "chrome_rss_samples": process_memory["samples"],
                "in_page_memory_method": "performance.measureUserAgentSpecificMemory()",
                "rss_selection_method": (
                    "Exclude pre-existing Chrome PIDs; sum newly appearing processes "
                    "whose command contains Google Chrome; sample every 100 ms"
                ),
                "rss_caveat": (
                    "Not a parent/child traversal; can include unrelated newly launched Chrome "
                    "processes, double-count shared pages, and does not isolate unified GPU memory"
                ),
            }
        )
    return rows


def main() -> None:
    manifest = load_json(GENERATED / "manifest.json")
    python_fidelity = load_json(RESULTS / "python_fidelity.json")
    python_latency = load_json(RESULTS / "python_latency.json")
    wasm = load_json(RESULTS / "browser_wasm.json")
    webgpu = load_json(RESULTS / "browser_webgpu.json")
    # A full browser run performs the memory workload in its own fresh Chrome
    # process before the fidelity/latency session and embeds that isolated trial.
    # Standalone files support a deliberately narrower --memory-only refresh.
    standalone_wasm_memory = load_json(RESULTS / "browser_memory_wasm.json")
    standalone_webgpu_memory = load_json(RESULTS / "browser_memory_webgpu.json")

    def current_memory_record(
        browser: Dict[str, Any], standalone: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "process_memory" in browser and "evaluation_provenance" in browser:
            return {
                "backend": browser["backend"],
                "timestamp_utc": browser.get("timestamp_utc"),
                "process_memory": browser["process_memory"],
                "evaluation_provenance": browser["evaluation_provenance"],
            }
        return standalone

    wasm_memory = current_memory_record(wasm, standalone_wasm_memory)
    webgpu_memory = current_memory_record(webgpu, standalone_webgpu_memory)

    assert manifest["all_cohort"]["n_cases"] == 7143
    assert len(manifest["stratified"]["selected_patient_ids"]) == 256
    assert len(manifest["stratified"]["cases"]) == 257
    assert wasm["preprocessing"]["exact"] and webgpu["preprocessing"]["exact"]
    assert wasm["backend"] == "wasm" and webgpu["backend"] == "webgpu"
    artifact_paths = {
        "model": ROOT / "delphi.onnx",
        "sdk": ROOT / "delphiSDK.js",
        "labels": ROOT / "delphi_labels_chapters_colours_icd.json",
    }
    for browser in (wasm, webgpu):
        for artifact, artifact_path in artifact_paths.items():
            assert browser["artifacts_bytes"][artifact] == artifact_path.stat().st_size, (
                f"{browser['backend']} result is stale for {artifact}: "
                f"recorded {browser['artifacts_bytes'][artifact]}, current {artifact_path.stat().st_size}"
            )
    expected_provenance = {
        "manifest_sha256": sha256_file(GENERATED / "manifest.json"),
        "upstream_commit": manifest["upstream"]["commit"],
        "validation_sha256": manifest["upstream"]["validation_sha256"],
        "upstream_model_source_sha256": manifest["upstream"]["model_source_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "onnx_sha256": manifest["onnx"]["sha256"],
        "labels_sha256": manifest["labels"]["sha256"],
        "sdk_sha256": sha256_file(ROOT / "delphiSDK.js"),
        "all_final_reference_sha256": sha256_file(
            GENERATED / manifest["all_cohort"]["reference_file"]
        ),
        "stratified_full_reference_sha256": sha256_file(
            GENERATED / manifest["stratified"]["cases"][0]["logits_file"]
        ),
    }
    for browser in (wasm, webgpu):
        assert browser.get("evaluation_provenance") == expected_provenance, (
            f"{browser['backend']} evaluation provenance is missing or stale"
        )
    for memory_record in (wasm_memory, webgpu_memory):
        assert memory_record.get("evaluation_provenance") == expected_provenance, (
            f"{memory_record['backend']} memory provenance is missing or stale"
        )
    expected_python_provenance = {
        key: value for key, value in expected_provenance.items() if key != "sdk_sha256"
    }
    for python_result in (python_fidelity, python_latency):
        assert python_result.get("evaluation_provenance") == expected_python_provenance, (
            "Python evaluation provenance is missing or stale"
        )
    for browser in (wasm, webgpu):
        for logits_key, distributions_key in (
            ("all_cohort_final_logits", "all_cohort_final_distributions"),
            ("stratified_all_position_logits", "stratified_all_position_distributions"),
        ):
            logits = browser["fidelity"][logits_key]
            distributions = browser["fidelity"][distributions_key]
            assert logits["within_tolerance_fraction"] == 1
            assert distributions["top1_agreement"] == 1
            assert not any(distributions["acceptance_failures"].values())
            assert distributions["mask_cell_mismatches"] == 0

    fidelity = fidelity_rows(python_fidelity, wasm, webgpu)
    fidelity_fields = list(fidelity[0].keys())
    write_csv(RESULTS / "fidelity_table.csv", fidelity, fidelity_fields)

    latency = latency_rows(python_latency, wasm, webgpu)
    latency_fields = list(latency[0].keys())
    write_csv(RESULTS / "latency_table.csv", latency, latency_fields)

    cold_context = next(
        case for case in manifest["stratified"]["cases"] if case["context_length"] >= 24
    )
    cold_context_length = cold_context["context_length"]
    startup = startup_rows(
        wasm, webgpu, wasm_memory, webgpu_memory, cold_context_length
    )
    startup_fields = list(startup[0].keys())
    write_csv(RESULTS / "startup_size_memory_table.csv", startup, startup_fields)

    pt = python_latency["results"]["pytorch_cpu"]
    pyort = python_latency["results"]["onnxruntime_python_cpu"]
    wfinal = wasm["fidelity"]["all_cohort_final_logits"]
    gfinal = webgpu["fidelity"]["all_cohort_final_logits"]
    wfinal_dist = wasm["fidelity"]["all_cohort_final_distributions"]
    gfinal_dist = webgpu["fidelity"]["all_cohort_final_distributions"]
    wfull = wasm["fidelity"]["stratified_all_position_logits"]
    gfull = webgpu["fidelity"]["stratified_all_position_logits"]
    wfull_dist = wasm["fidelity"]["stratified_all_position_distributions"]
    gfull_dist = webgpu["fidelity"]["stratified_all_position_distributions"]
    native = python_fidelity["pytorch_vs_python_onnxruntime"]
    acceptance = manifest["acceptance_criteria"]
    legacy = python_fidelity["legacy_javascript_default_mask_diagnostic"]
    wm = wasm_memory["process_memory"]
    gm = webgpu_memory["process_memory"]
    wsnaps = {x["label"]: x["bytes"] for x in wm["in_page"]["user_agent_specific_memory"]}
    gsnaps = {x["label"]: x["bytes"] for x in gm["in_page"]["user_agent_specific_memory"]}
    host = wasm["host"]
    machine_model = host.get("machine_model", "desktop")
    processor = host.get("processor", host["cpu_model"])
    gpu_power_preference = host.get("gpu_power_preference", "not recorded")
    environment = python_fidelity["environment"]
    cohort_cases = manifest["all_cohort"]["cases"]
    raw_lengths = sorted(case["raw_length"] for case in cohort_cases)
    context_lengths = sorted(case["context_length"] for case in cohort_cases)
    female = sum(case["sex"] == "female" for case in cohort_cases)
    male = sum(case["sex"] == "male" for case in cohort_cases)
    tied = sum(case["has_tied_age"] for case in cohort_cases)
    deaths = sum(case["ends_in_death"] for case in cohort_cases)
    raw_records = sum(case["raw_length"] for case in cohort_cases)

    model_table = []
    for runtime, steady in (
        ("PyTorch CPU (reference)", pt["steady_state"]),
        ("ORT Python CPU (diagnostic)", pyort["steady_state"]),
        ("ORT Web Wasm, 1 thread", wasm["latency"]["model_only"]),
        ("ORT Web WebGPU", webgpu["latency"]["model_only"]),
    ):
        model_table.append(
            f"| {runtime} | {interval(steady['12'])} | {interval(steady['24'])} | "
            f"{interval(steady['48'])} |"
        )

    browser_step_table = []
    for runtime, steady in (
        ("ORT Web Wasm, 1 thread", wasm["latency"]["end_to_end_step"]),
        ("ORT Web WebGPU", webgpu["latency"]["end_to_end_step"]),
    ):
        browser_step_table.append(
            f"| {runtime} | {interval(steady['12'])} | {interval(steady['24'])} | "
            f"{interval(steady['48'])} |"
        )

    w_ratio = wasm["latency"]["full_trajectory"]["median_ms"] / pt["full_trajectory"][
        "median_ms"
    ]
    g_ratio = webgpu["latency"]["full_trajectory"]["median_ms"] / pt[
        "full_trajectory"
    ]["median_ms"]
    w48_ratio = wasm["latency"]["model_only"]["48"]["median_ms"] / pt["steady_state"][
        "48"
    ]["median_ms"]
    g48_ratio = webgpu["latency"]["model_only"]["48"]["median_ms"] / pt[
        "steady_state"
    ]["48"]["median_ms"]
    g_vs_w_48 = 1 - webgpu["latency"]["model_only"]["48"]["median_ms"] / wasm[
        "latency"
    ]["model_only"]["48"]["median_ms"]

    report = f"""# Delphi ONNX fidelity and performance evaluation

## Executive result

The browser port reproduced the PyTorch reference to substantially better than the declared probability-level tolerance. Across all 7,143 complete held-out trajectories, ONNX Runtime Web Wasm had logit mean absolute error (MAE) {sci(wfinal['mean_absolute_error'])} and maximum absolute error {sci(wfinal['maximum_absolute_error'])}; WebGPU had MAE {sci(gfinal['mean_absolute_error'])} and maximum {sci(gfinal['maximum_absolute_error'])}. Every one of 9,071,610 logits per backend satisfied `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. The largest conditional event-probability errors were {sci(wfinal_dist['maximum_absolute_probability_error'])} (Wasm) and {sci(gfinal_dist['maximum_absolute_probability_error'])} (WebGPU), maximum total-variation distances were {sci(wfinal_dist['total_variation_distance']['maximum'])} and {sci(gfinal_dist['total_variation_distance']['maximum'])}, and top-1 agreement was 100% for both.

Performance was usable but backend- and context-dependent, so the manuscript should replace an unqualified “near-native” claim with measurements. A fixed 16-event prefix plus 32 generated events took a median {ms(pt['full_trajectory']['median_ms'])} ms in PyTorch CPU, {ms(wasm['latency']['full_trajectory']['median_ms'])} ms in browser Wasm ({w_ratio:.2f}× PyTorch), and {ms(webgpu['latency']['full_trajectory']['median_ms'])} ms in browser WebGPU ({g_ratio:.2f}×). These results apply to one desktop only; they do not establish mobile or general cross-device performance.

## Methods

### Model and reference implementation

We evaluated the locally supplied synthetic-data checkpoint `OriginalModel.pt`, not the inaccessible checkpoint used in the published Delphi-2M study. The checkpoint is intentionally not committed to this repository; its SHA-256 was `{manifest['checkpoint']['sha256']}`, which identifies the exact local input needed to reproduce this run. It contained {manifest['checkpoint']['parameters']:,} parameters (12 layers, 12 attention heads, embedding dimension 120, vocabulary size 1,270, and maximum context 48). Evaluation mode disabled dropout. The tested ONNX artifact SHA-256 was `{manifest['onnx']['sha256']}` and its output was the raw `[batch, position, event]` logit tensor immediately before stochastic event/time sampling.

The source implementation and validation data were pinned to [gerstung-lab/Delphi commit `{manifest['upstream']['commit'][:8]}`](https://github.com/gerstung-lab/Delphi/tree/{manifest['upstream']['commit']}). This comparison therefore establishes equivalence for the synthetic checkpoint in this repository; it cannot establish equivalence to the unavailable published model.

### Fixed evaluation cohort

The upstream synthetic validation file `val.bin` was downloaded from the pinned commit and verified against SHA-256 `{manifest['upstream']['validation_sha256']}`. Patient {manifest['upstream']['split_patient_excluded']} was excluded because the upstream byte-level split bisected that trajectory, leaving only an {manifest['upstream']['split_fragment_rows']}-row validation fragment. The resulting primary cohort contained 7,143 complete held-out trajectories and {raw_records:,} raw records: {female:,} female and {male:,} male trajectories, {tied:,} with tied event ages, and {deaths:,} ending in death. Raw trajectory length ranged from {raw_lengths[0]} to {raw_lengths[-1]} events (median {raw_lengths[len(raw_lengths)//2]}, IQR {raw_lengths[1785]}–{raw_lengths[5357]}). Processed input length ranged from {context_lengths[0]} to {context_lengths[-1]} (median {context_lengths[len(context_lengths)//2]}).

The primary cohort test compared the final-position logits for all 7,143 trajectories (9,071,610 logits per runtime). To exercise intermediate positions and the hand-written browser pipeline, a second fixed set used NumPy seed {manifest['seeds']['selection']} to select 32 trajectories without replacement from each sex-by-raw-length-quartile stratum (2 sexes × 4 quartiles × 32 = 256), plus one authored case containing lifestyle tokens, repeated no-event tokens, and tied ages. This yielded 257 cases, 10,178 evaluated positions, and 12,926,060 logits per runtime. The resolved patient IDs are stored in the generated manifest.

### Preprocessing and probability comparison

Reference contexts followed the pinned upstream fixed-width inference path: the rightmost 49 raw records were placed into a left-masked window; regular no-event records were inserted at day 1 and every 1,826.25 days through the last observed age; the combined records were ordered by age with `torch.argsort`; raw token IDs were increased by one; and the rightmost 49 combined records were retained. The final record was reserved as the prediction target, leaving at most 48 input records. Reproducing the fixed-width intermediate array also reproduced upstream ordering for exact-age ties. Tokens were `int64` and ages were IEEE-754 `float32` days.

The browser check then exercised the public SDK input boundary on the 257 all-position cases: it supplied event names and ages in years to `prepareTrajectoryInputs`, then compared the resulting token IDs and float32 day values with the reference tensors. JavaScript and Python had zero mismatches in 10,178 token cells, zero bit mismatches in 10,178 float32 age cells, and zero event-name mismatches after mapping the tokens back to names, on both browser backends. The SDK does not consume raw `val.bin`; raw cohort-window construction therefore remained a reference-data preparation step, while the SDK-specific tokenization, age conversion, tensor construction, masking, inference, output readback, and event-name mapping were exercised in JavaScript.

Sampling was intentionally excluded from fidelity testing. At each evaluated position, the harness invoked the SDK pre-sampling mask and verified its masked cells against checkpoint `ignore_tokens=[0,2,...,12]` and the previously observed event tokens. It then independently derived and compared (i) raw logits, (ii) the conditional event distribution obtained by normalizing `exp(logit)` over unmasked events, and (iii) `logsumexp(logit)` over unmasked events, which is the log total event rate implied at the exponential-race sampling boundary. Ten repeated fixed-input browser inferences were bitwise identical on each backend. Benchmark seed {manifest['seeds']['benchmark']} controlled stochastic trajectory timing, but sampled trajectories were not treated as a numerical-fidelity endpoint.

### Acceptance criteria

The executable protocol recorded the following criteria before aggregating results:

- every logit must satisfy `abs(error) <= atol + rtol * abs(reference)`, with `atol={acceptance['logits_atol']}` and `rtol={acceptance['logits_rtol']}`;
- maximum per-context total-variation distance, maximum absolute event-probability error, and maximum absolute log-total-rate error must each be no greater than `{acceptance['max_context_total_variation']}`; and
- top-1 event agreement must equal 100%.

The combined absolute-plus-relative logit rule matters for interpreting the WebGPU maximum absolute error ({sci(gfinal['maximum_absolute_error'])}): it exceeds `1e-4` in isolation but occurred on a logit of magnitude about 20 and passed the declared combined tolerance. Probability- and rate-level maximum errors remained below `1e-4` without a relative term.

### Performance protocol

Measurements were made on 13 August 2026 on a {machine_model} running macOS 26.6.1, with {processor} and {host['total_memory_bytes'] / (1024**3):.0f} GiB unified memory, while connected to AC power. PyTorch {environment['torch']} used CPU because MPS was unavailable, with {environment['torch_threads']} intra-op and {environment['torch_interop_threads']} inter-op threads. Browser measurements used headless Google Chrome {environment['chrome'].replace('Google Chrome ', '')} and ONNX Runtime Web 1.27.0. Wasm was explicitly limited to one thread. WebGPU was the only configured execution provider and used the browser's {gpu_power_preference} power preference; ONNX Runtime nevertheless assigned shape/control nodes internally to CPU, as reported by its runtime diagnostics, rather than falling back to a second configured provider. The Python ONNX Runtime 1.19.2 CPU results are reported only as a conversion diagnostic, not as the deployed endpoint.

For each context length (12, 24, and 48), each runtime cycled through the same 30 fixed validation contexts. Each measured endpoint had 10 warm-up executions followed by 200 timed executions. “Model-only” included prebuilt tensors, inference, and output materialization. Browser “SDK end-to-end step” additionally included JavaScript array-to-tensor construction and final-position extraction; model-only and SDK calls were paired with alternating order to limit order and thermal bias. Full-trajectory latency used the same fixed 16-event prefix in Python and JavaScript and comprised exactly 32 sampled events with termination disabled, plus a final all-position inference: 33 graph executions total, with 2 warm-ups and 30 measured runs. Cold initialization and first-inference latency used 10 fresh Chrome processes per backend; first inference used a fixed context of length {cold_context_length}. Runtime, model, and label artifacts were served without caching over loopback. The initialization timer began after the page and SDK module loaded, and therefore captures runtime/model/label fetching, parsing, graph compilation, and session creation—but not HTML navigation, SDK-module loading, or real Internet transfer time.

Memory was measured in a separate browser trial. In-page memory used `performance.measureUserAgentSpecificMemory()`. A second sampler excluded every pre-existing Chrome PID, then summed resident-set size (RSS) every 100 ms across newly appearing processes whose command contained “Google Chrome”; it reports the increment above the dedicated trial's idle page. This is not a parent/child process-tree traversal: unrelated Chrome processes that started during the trial could contaminate the total, and summed RSS can double-count shared pages. Unified-memory macOS also does not expose an isolated GPU-memory peak. These values are feasibility proxies rather than precise model-only allocations.

## Results

### Fidelity: primary browser endpoint

| Backend and scope | Contexts | Logits | Logit MAE | Max abs logit error | Within combined tolerance | Max abs probability error | Max TV distance | Max abs log-total-rate error | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wasm, all final positions | 7,143 | 9,071,610 | {sci(wfinal['mean_absolute_error'])} | {sci(wfinal['maximum_absolute_error'])} | {pct(wfinal['within_tolerance_fraction'])} | {sci(wfinal_dist['maximum_absolute_probability_error'])} | {sci(wfinal_dist['total_variation_distance']['maximum'])} | {sci(wfinal_dist['absolute_log_total_rate_error']['maximum'])} | {pct(wfinal_dist['top1_agreement'])} |
| WebGPU, all final positions | 7,143 | 9,071,610 | {sci(gfinal['mean_absolute_error'])} | {sci(gfinal['maximum_absolute_error'])} | {pct(gfinal['within_tolerance_fraction'])} | {sci(gfinal_dist['maximum_absolute_probability_error'])} | {sci(gfinal_dist['total_variation_distance']['maximum'])} | {sci(gfinal_dist['absolute_log_total_rate_error']['maximum'])} | {pct(gfinal_dist['top1_agreement'])} |
| Wasm, stratified all positions | 10,178 | 12,926,060 | {sci(wfull['mean_absolute_error'])} | {sci(wfull['maximum_absolute_error'])} | {pct(wfull['within_tolerance_fraction'])} | {sci(wfull_dist['maximum_absolute_probability_error'])} | {sci(wfull_dist['total_variation_distance']['maximum'])} | {sci(wfull_dist['absolute_log_total_rate_error']['maximum'])} | {pct(wfull_dist['top1_agreement'])} |
| WebGPU, stratified all positions | 10,178 | 12,926,060 | {sci(gfull['mean_absolute_error'])} | {sci(gfull['maximum_absolute_error'])} | {pct(gfull['within_tolerance_fraction'])} | {sci(gfull_dist['maximum_absolute_probability_error'])} | {sci(gfull_dist['total_variation_distance']['maximum'])} | {sci(gfull_dist['absolute_log_total_rate_error']['maximum'])} | {pct(gfull_dist['top1_agreement'])} |

No browser context exceeded any probability- or rate-level acceptance threshold, and there were zero masking-cell mismatches. Mean top-5 overlap was {pct(wfinal_dist['mean_top5_overlap'], 4)} for Wasm (one boundary-rank difference across the cohort) and 100% for WebGPU; mean top-10 overlap was 100% for both. The pre-evaluation SDK default omitted the checkpoint-level ignored-token mask. That legacy setting changed the PyTorch conditional distribution by mean TV {sci(legacy['total_variation_distance']['mean'])} and maximum TV {sci(legacy['total_variation_distance']['maximum'])}; the SDK was corrected to use `[0,2,...,12]`. This demonstrates why pipeline-level validation was necessary even though the ONNX graph itself was already close.

### Fidelity: intermediate Python ONNX Runtime diagnostic

| Scope | Logit MAE | Max abs logit error | Within combined tolerance | Max abs probability error | Max TV distance | Max abs log-total-rate error | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All final positions | {sci(native['all_cohort_final_logits']['mean_absolute_error'])} | {sci(native['all_cohort_final_logits']['maximum_absolute_error'])} | {pct(native['all_cohort_final_logits']['within_tolerance_fraction'])} | {sci(native['all_cohort_final_distributions']['maximum_absolute_probability_error'])} | {sci(native['all_cohort_final_distributions']['total_variation_distance']['maximum'])} | {sci(native['all_cohort_final_distributions']['absolute_log_total_rate_error']['maximum'])} | {pct(native['all_cohort_final_distributions']['top1_agreement'])} |
| Stratified all positions | {sci(native['stratified_all_position_logits']['mean_absolute_error'])} | {sci(native['stratified_all_position_logits']['maximum_absolute_error'])} | {pct(native['stratified_all_position_logits']['within_tolerance_fraction'])} | {sci(native['stratified_all_position_distributions']['maximum_absolute_probability_error'])} | {sci(native['stratified_all_position_distributions']['total_variation_distance']['maximum'])} | {sci(native['stratified_all_position_distributions']['absolute_log_total_rate_error']['maximum'])} | {pct(native['stratified_all_position_distributions']['top1_agreement'])} |

This diagnostic is consistent with the remaining browser differences arising from backend floating-point variation rather than an export failure. It is not substituted for the browser endpoint in the primary result.

### Steady-state single-step latency

Values are median ms (IQR); p95 ms. Each cell summarizes 200 runs after 10 warm-ups.

| Runtime | Context 12 | Context 24 | Context 48 |
|---|---:|---:|---:|
{chr(10).join(model_table)}

Browser SDK end-to-end step latency, including tensor construction and final-position extraction, was:

| Runtime | Context 12 | Context 24 | Context 48 |
|---|---:|---:|---:|
{chr(10).join(browser_step_table)}

At context 48, browser model-only latency was {w48_ratio:.2f}× PyTorch for Wasm and {g48_ratio:.2f}× for WebGPU. WebGPU was {pct(g_vs_w_48, 1)} faster than Wasm at this length, but its fixed dispatch/readback overhead made it slower at lengths 12 and 24. The Python ONNX Runtime values are implementation diagnostics and should not be used to characterize browser performance.

### Full-trajectory latency

| Runtime | Runs | Median ms (IQR) | p95 ms | Ratio to PyTorch |
|---|---:|---:|---:|---:|
| PyTorch CPU | {pt['full_trajectory']['n']} | {ms(pt['full_trajectory']['median_ms'])} ({ms(pt['full_trajectory']['q1_ms'])}–{ms(pt['full_trajectory']['q3_ms'])}) | {ms(pt['full_trajectory']['p95_ms'])} | 1.00× |
| ORT Web Wasm, 1 thread | {wasm['latency']['full_trajectory']['n']} | {ms(wasm['latency']['full_trajectory']['median_ms'])} ({ms(wasm['latency']['full_trajectory']['q1_ms'])}–{ms(wasm['latency']['full_trajectory']['q3_ms'])}) | {ms(wasm['latency']['full_trajectory']['p95_ms'])} | {w_ratio:.2f}× |
| ORT Web WebGPU | {webgpu['latency']['full_trajectory']['n']} | {ms(webgpu['latency']['full_trajectory']['median_ms'])} ({ms(webgpu['latency']['full_trajectory']['q1_ms'])}–{ms(webgpu['latency']['full_trajectory']['q3_ms'])}) | {ms(webgpu['latency']['full_trajectory']['p95_ms'])} | {g_ratio:.2f}× |

### Cold initialization, artifact size, and memory

| Backend | Session initialization, median (IQR); p95 ms | First inference, median (IQR); p95 ms | Model MB | Raw first-use payload MB |
|---|---:|---:|---:|---:|
| Wasm | {interval(wasm['cold_start']['session_initialization'])} | {interval(wasm['cold_start']['first_inference'])} | {mb(wasm['artifacts_bytes']['model'])} | {mb(wasm['artifacts_bytes']['first_use_payload'])} |
| WebGPU | {interval(webgpu['cold_start']['session_initialization'])} | {interval(webgpu['cold_start']['first_inference'])} | {mb(webgpu['artifacts_bytes']['model'])} | {mb(webgpu['artifacts_bytes']['first_use_payload'])} |

The ONNX model itself was {wasm['artifacts_bytes']['model']:,} bytes ({mb(wasm['artifacts_bytes']['model'])} decimal MB). The raw first-use totals include the model, {wasm['artifacts_bytes']['sdk']:,}-byte SDK, {wasm['artifacts_bytes']['labels']:,}-byte labels, and the backend-specific runtime JavaScript and Wasm binary; they are uncompressed file sizes, not content-encoded network transfer sizes. Across the 10 WebGPU cold trials, the maximum first-inference time was {ms(max(webgpu['cold_start']['first_inference']['samples_ms']))} ms, compared with median {ms(webgpu['cold_start']['first_inference']['median_ms'])} ms and p95 {ms(webgpu['cold_start']['first_inference']['p95_ms'])} ms.

The 60 browser steady-state warm-up executions (10 model-only and 10 SDK end-to-end calls at each of the three context lengths) took {ms(wasm['latency']['warmup_total_ms'])} ms in Wasm and {ms(webgpu['latency']['warmup_total_ms'])} ms in WebGPU in the full benchmark sessions. These totals are reported for procedural completeness; the fresh-browser length-{cold_context_length} first-inference distribution better characterizes one-time warm-up cost because the 60-execution totals mix two endpoints and three context lengths.

| Backend | In-page after session MB | In-page after rollout MB | Idle Chrome RSS MB | Peak Chrome RSS MB | Incremental peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Wasm | {mb(wsnaps['after_session_initialization'])} | {mb(wsnaps['after_32_event_rollout'])} | {mb(wm['idle_rss_bytes'])} | {mb(wm['peak_rss_bytes'])} | {mb(wm['incremental_peak_rss_bytes'])} |
| WebGPU | {mb(gsnaps['after_session_initialization'])} | {mb(gsnaps['after_32_event_rollout'])} | {mb(gm['idle_rss_bytes'])} | {mb(gm['peak_rss_bytes'])} | {mb(gm['incremental_peak_rss_bytes'])} |

## Publication-ready manuscript text

### Methods insertion

> We assessed numerical fidelity against the PyTorch implementation using a locally supplied synthetic-data checkpoint (`OriginalModel.pt`, SHA-256 `{manifest['checkpoint']['sha256']}`); the original Delphi-2M paper checkpoint was unavailable, and the local checkpoint is not committed to the conversion repository. We pinned the upstream synthetic validation data and preprocessing code to Delphi commit `{manifest['upstream']['commit'][:8]}` and verified the validation file by SHA-256. After excluding one patient whose trajectory was bisected by the upstream byte-level train/validation split, the test cohort comprised 7,143 complete held-out trajectories. Preprocessing reproduced the upstream fixed-width path: it placed the rightmost 49 raw records in a left-masked window, inserted a no-event token every 5 years, ordered the combined records by age using `torch.argsort` (including its exact-age tie behavior), shifted token IDs by one, retained the rightmost 49 combined records, reserved the final record as the target, and supplied at most 48 input records as int64 tokens and float32 ages in days. We compared final-position logits for every trajectory and all-position logits for a fixed seed-1337 stratified subset of 256 trajectories (32 per sex-by-length-quartile stratum) plus one authored lifestyle/no-event edge case. Because sampling is stochastic, fidelity was evaluated immediately before sampling. After applying the checkpoint’s ignored-token and previously-seen-event masks, we compared raw logits, conditional event probabilities (masked softmax), and log total event rate (masked log-sum-exp). Acceptance required all logits to satisfy `|Δ| <= 10^-4 + 10^-4|reference|`, maximum event-probability error, total-variation distance, and log-total-rate error no greater than `10^-4`, and 100% top-1 agreement. The actual ONNX Runtime Web Wasm and WebGPU paths were the primary endpoints; ONNX Runtime Python was an intermediate conversion diagnostic.
>
> Performance was measured on one {machine_model} with {processor} and 18 GiB unified memory using PyTorch 2.8.0 CPU, ONNX Runtime Web 1.27.0 Wasm (one thread), and WebGPU in headless Chrome 151. At each context length (12, 24, and 48), all runtimes cycled through the same 30 fixed validation contexts; each endpoint had 10 warm-up executions followed by 200 timed executions. We recorded model-only latency (prebuilt tensors through materialized output), browser SDK end-to-end step latency (including tensor construction and final-position extraction), and 30 complete 32-event rollouts from the same 16-event prefix (33 graph executions, after two warm-ups). Ten fresh-browser trials measured session initialization and length-{cold_context_length} first-inference latency. Artifact sizes are raw uncompressed bytes. Memory was estimated both in-page and as the increment above an idle page in sampled summed RSS of newly launched Chrome processes from a separate trial.

### Results insertion

> ONNX Runtime Web reproduced the PyTorch reference within the declared tolerance. Across 9,071,610 final-position logits, Wasm and WebGPU logit MAE values were {sci(wfinal['mean_absolute_error'])} and {sci(gfinal['mean_absolute_error'])}, with maximum absolute errors {sci(wfinal['maximum_absolute_error'])} and {sci(gfinal['maximum_absolute_error'])}, respectively; 100% satisfied the combined `atol=rtol=10^-4` criterion. Maximum conditional event-probability errors were {sci(wfinal_dist['maximum_absolute_probability_error'])} and {sci(gfinal_dist['maximum_absolute_probability_error'])}, maximum total-variation distances were {sci(wfinal_dist['total_variation_distance']['maximum'])} and {sci(gfinal_dist['total_variation_distance']['maximum'])}, maximum log-total-rate errors were {sci(wfinal_dist['absolute_log_total_rate_error']['maximum'])} and {sci(gfinal_dist['absolute_log_total_rate_error']['maximum'])}, and top-1 agreement was 100% for both. The 10,178-position stratified test likewise passed all criteria. JavaScript and Python preprocessing agreed exactly in all 10,178 token and age cells, with no event-name postprocessing mismatches. This evaluation identified and corrected an SDK default mask that had omitted checkpoint-ignored tokens.
>
> On the tested desktop, median model-only latency at context lengths 12/24/48 was {ms(pt['steady_state']['12']['median_ms'])}/{ms(pt['steady_state']['24']['median_ms'])}/{ms(pt['steady_state']['48']['median_ms'])} ms for PyTorch CPU, {ms(wasm['latency']['model_only']['12']['median_ms'])}/{ms(wasm['latency']['model_only']['24']['median_ms'])}/{ms(wasm['latency']['model_only']['48']['median_ms'])} ms for browser Wasm, and {ms(webgpu['latency']['model_only']['12']['median_ms'])}/{ms(webgpu['latency']['model_only']['24']['median_ms'])}/{ms(webgpu['latency']['model_only']['48']['median_ms'])} ms for browser WebGPU. A 32-event rollout took a median {ms(pt['full_trajectory']['median_ms'])} ms in PyTorch, {ms(wasm['latency']['full_trajectory']['median_ms'])} ms in Wasm ({w_ratio:.2f}×), and {ms(webgpu['latency']['full_trajectory']['median_ms'])} ms in WebGPU ({g_ratio:.2f}×). The ONNX artifact was {mb(wasm['artifacts_bytes']['model'])} MB; raw first-use payloads were {mb(wasm['artifacts_bytes']['first_use_payload'])} MB (Wasm) and {mb(webgpu['artifacts_bytes']['first_use_payload'])} MB (WebGPU). Median cold session initialization/length-{cold_context_length} first-inference times were {ms(wasm['cold_start']['session_initialization']['median_ms'])}/{ms(wasm['cold_start']['first_inference']['median_ms'])} ms for Wasm and {ms(webgpu['cold_start']['session_initialization']['median_ms'])}/{ms(webgpu['cold_start']['first_inference']['median_ms'])} ms for WebGPU; 60 browser steady-state warm-up executions took {ms(wasm['latency']['warmup_total_ms'])} and {ms(webgpu['latency']['warmup_total_ms'])} ms. Incremental peak newly launched Chrome-process RSS was {mb(wm['incremental_peak_rss_bytes'])} MB and {mb(gm['incremental_peak_rss_bytes'])} MB, respectively. These measurements demonstrate sub-200-ms median 32-event rollouts on the tested desktop, but not an unqualified “near-native” or mobile-performance claim.

## Draft response to reviewers

### Comment 1: fidelity

Thank you; we agree that conversion alone was insufficient evidence of correctness. We added a deterministic, end-to-end comparison against the PyTorch implementation using the synthetic-data checkpoint, since the checkpoint from the original Delphi-2M publication is not accessible. The primary endpoint is the actual JavaScript/ONNX Runtime Web path, not Python ONNX Runtime. We compared 9,071,610 final-position logits from all 7,143 complete held-out validation trajectories and 12,926,060 all-position logits from a fixed stratified set of 256 trajectories plus one authored lifestyle/no-event edge case. Sampling was excluded so that both implementations could be compared immediately before the stochastic boundary; instead, we compared the resulting masked event-probability distributions and log total event rates.

For Wasm, logit MAE/max absolute error were {sci(wfinal['mean_absolute_error'])}/{sci(wfinal['maximum_absolute_error'])}; for WebGPU they were {sci(gfinal['mean_absolute_error'])}/{sci(gfinal['maximum_absolute_error'])}. All logits satisfied `|Δ| <= 10^-4 + 10^-4|reference|`. Maximum probability error, total-variation distance, and log-total-rate error were {sci(wfinal_dist['maximum_absolute_probability_error'])}, {sci(wfinal_dist['total_variation_distance']['maximum'])}, and {sci(wfinal_dist['absolute_log_total_rate_error']['maximum'])} for Wasm, and {sci(gfinal_dist['maximum_absolute_probability_error'])}, {sci(gfinal_dist['total_variation_distance']['maximum'])}, and {sci(gfinal_dist['absolute_log_total_rate_error']['maximum'])} for WebGPU. Top-1 agreement was 100% for both. JavaScript preprocessing matched Python exactly in every tested token and float32 age cell, and event-name postprocessing had zero mismatches. Importantly, the exercise exposed an SDK default-mask discrepancy, which we corrected; we now describe and test the checkpoint’s ignored-token mask explicitly. We added these methods, quantitative results, tolerances, fixed seeds, and reproducibility artifacts to the manuscript and repository.

### Comment 2: performance

Thank you; we replaced qualitative performance assertions with measurements and limited our inference to the tested desktop. On an Apple M3 Pro, browser model-only latency at context lengths 12/24/48 was {ms(wasm['latency']['model_only']['12']['median_ms'])}/{ms(wasm['latency']['model_only']['24']['median_ms'])}/{ms(wasm['latency']['model_only']['48']['median_ms'])} ms for one-thread Wasm and {ms(webgpu['latency']['model_only']['12']['median_ms'])}/{ms(webgpu['latency']['model_only']['24']['median_ms'])}/{ms(webgpu['latency']['model_only']['48']['median_ms'])} ms for WebGPU, compared with {ms(pt['steady_state']['12']['median_ms'])}/{ms(pt['steady_state']['24']['median_ms'])}/{ms(pt['steady_state']['48']['median_ms'])} ms for PyTorch CPU. A fixed 32-event rollout required median {ms(pt['full_trajectory']['median_ms'])} ms in PyTorch, {ms(wasm['latency']['full_trajectory']['median_ms'])} ms in Wasm, and {ms(webgpu['latency']['full_trajectory']['median_ms'])} ms in WebGPU. We also report 10-trial cold initialization and length-{cold_context_length} first-inference times, 60-execution browser warm-up totals, the {mb(wasm['artifacts_bytes']['model'])}-MB model, backend-specific raw first-use payloads of {mb(wasm['artifacts_bytes']['first_use_payload'])} and {mb(webgpu['artifacts_bytes']['first_use_payload'])} MB, and memory proxies.

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
"""
    (EVALUATION / "PUBLICATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
