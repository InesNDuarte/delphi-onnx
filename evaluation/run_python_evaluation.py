#!/usr/bin/env python3
"""Create fixed Delphi fidelity fixtures and benchmark native Python runtimes.

The browser evaluator consumes the generated manifest and little-endian float32
reference files. Large reproducible files live under evaluation/generated/ and
are intentionally excluded from version control.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_COMMIT = "fb72166be6b29d8db819227a59487e51c1df1454"
VAL_URL = (
    "https://raw.githubusercontent.com/gerstung-lab/Delphi/"
    f"{UPSTREAM_COMMIT}/data/ukb_simulated_data/val.bin"
)
VAL_SHA256 = "f57f6a63e339f0c3643709f80443a75ecb05850986a24b91fb2d1910c1d11484"
MODEL_URL = (
    "https://raw.githubusercontent.com/gerstung-lab/Delphi/"
    f"{UPSTREAM_COMMIT}/model.py"
)
MODEL_SHA256 = "1f7e9349c09bb83fe74fe5cb48cac8845fccd9a81c3dcff3f3fa73f120d6c906"
SPLIT_PATIENT_ID = 427985
SELECTION_SEED = 1337
BENCHMARK_SEED = 20260813
YEAR_DAYS = 365.25
MASK_AGE = -10000.0
VOCAB_SIZE = 1270
BLOCK_SIZE = 48
ATOL = 1e-4
RTOL = 1e-4
SELECTION_FIXTURE = ROOT / "evaluation" / "fixtures" / "stratified_patient_ids.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_val_data(explicit: Path | None, cache: Path) -> Path:
    path = explicit or cache
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pinned validation data to {path}", flush=True)
        urllib.request.urlretrieve(VAL_URL, path)
    actual = sha256(path)
    if actual != VAL_SHA256:
        raise ValueError(f"val.bin SHA-256 mismatch: expected {VAL_SHA256}, got {actual}")
    return path


def ensure_upstream_model(explicit: Path | None, cache: Path) -> Path:
    path = explicit or cache
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pinned upstream model definition to {path}", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    actual = sha256(path)
    if actual != MODEL_SHA256:
        raise ValueError(f"model.py SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}")
    return path


def load_upstream_model_classes(path: Path) -> Tuple[Any, Any]:
    specification = importlib.util.spec_from_file_location("pinned_delphi_model", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to import pinned model definition from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.Delphi, module.DelphiConfig


def load_checkpoint(path: Path, model_source: Path) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    Delphi, DelphiConfig = load_upstream_model_classes(model_source)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_args = dict(checkpoint["model_args"])
    allowed = DelphiConfig.__dataclass_fields__.keys()
    config = DelphiConfig(**{key: value for key, value in model_args.items() if key in allowed})
    model = Delphi(config)
    state = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_args, checkpoint


def load_labels(path: Path) -> Tuple[List[str], Dict[str, int]]:
    records = json.loads(path.read_text())
    records.sort(key=lambda item: int(item["index"]))
    names = [record["name"] for record in records]
    if len(names) != VOCAB_SIZE or len(set(names)) != VOCAB_SIZE:
        raise ValueError("Expected 1,270 unique labels")
    return names, {name: index for index, name in enumerate(names)}


def group_rows(data: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    starts = np.r_[0, np.flatnonzero(data[1:, 0] != data[:-1, 0]) + 1]
    ends = np.r_[starts[1:], len(data)]
    return [(int(data[start, 0]), data[start:end]) for start, end in zip(starts, ends)]


def prepare_context(rows: np.ndarray, block_size: int = BLOCK_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror upstream ``get_batch(..., select='right', padding='regular')``.

    The fixed-width, left-masked raw window is retained before ``torch.argsort``.
    That detail affects the deterministic order of exact-age ties, because
    ``torch.argsort`` is not stable.  Reproducing the intermediate width makes
    this function bit-exact with the upstream batch loader, including tied ages.
    """

    raw_width = block_size + 1
    window = rows[-raw_width:]
    raw_tokens = np.full(raw_width, -1, dtype=np.int64)
    raw_ages = np.full(raw_width, MASK_AGE, dtype=np.float32)
    raw_tokens[-len(window) :] = window[:, 2].astype(np.int64, copy=False)
    raw_ages[-len(window) :] = window[:, 1].astype(np.float32, copy=False)

    pad_ages = (
        torch.arange(0, 36525, YEAR_DAYS * 5, dtype=torch.float32) + 1
    )
    tokens = torch.cat(
        (torch.from_numpy(raw_tokens), torch.zeros(len(pad_ages), dtype=torch.int64))
    )
    ages = torch.cat((torch.from_numpy(raw_ages), pad_ages))
    maximum_age = torch.from_numpy(raw_ages).max()
    beyond_trajectory = ages > maximum_age
    tokens = tokens.masked_fill(beyond_trajectory, -1)
    ages = ages.masked_fill(beyond_trajectory, MASK_AGE)

    order = torch.argsort(ages)
    tokens = tokens[order] + 1
    ages = ages[order]
    cut_margin = int(torch.sum(tokens == 0))
    tokens = tokens[cut_margin:]
    ages = ages[cut_margin:]
    if len(tokens) > raw_width:
        tokens = tokens[-raw_width:]
        ages = ages[-raw_width:]

    return np.ascontiguousarray(tokens[:-1].numpy(), dtype=np.int64), np.ascontiguousarray(
        ages[:-1].numpy(), dtype=np.float32
    )


def authored_case(name_to_token: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    names = [
        "Male", "B01 Varicella [chickenpox]", "L20 Atopic dermatitis", "No event",
        "No event", "No event", "No event", "G43 Migraine", "E73 Lactose intolerance",
        "B27 Infectious mononucleosis", "No event", "J11 Influenza, virus not identified",
        "No event", "No event", "No event", "Smoking low", "BMI mid", "Alcohol low",
        "No event",
    ]
    years = [0, 2, 3, 5, 10, 15, 20, 20, 21, 22, 25, 28, 30, 35, 40, 41, 41, 41, 42]
    return (
        np.asarray([name_to_token[name] for name in names], dtype=np.int64),
        np.asarray(years, dtype=np.float32) * np.float32(YEAR_DAYS),
        names,
    )


def sex_name(rows: np.ndarray) -> str:
    raw = int(rows[0, 2])
    if raw == 1:
        return "female"
    if raw == 2:
        return "male"
    return "atypical"


def select_stratified(groups: Sequence[Tuple[int, np.ndarray]]) -> Tuple[List[int], List[float]]:
    lengths = np.asarray([len(rows) for _, rows in groups], dtype=np.int64)
    boundaries = [float(value) for value in np.quantile(lengths, [0.25, 0.5, 0.75])]
    strata: Dict[Tuple[str, int], List[int]] = {}
    for patient_id, rows in groups:
        sex = sex_name(rows)
        if sex not in ("female", "male"):
            continue
        quartile = int(np.searchsorted(boundaries, len(rows), side="left"))
        strata.setdefault((sex, quartile), []).append(patient_id)
    rng = np.random.default_rng(SELECTION_SEED)
    selected: List[int] = []
    for sex in ("female", "male"):
        for quartile in range(4):
            candidates = np.asarray(sorted(strata[(sex, quartile)]), dtype=np.int64)
            if len(candidates) < 32:
                raise ValueError(f"Stratum {(sex, quartile)} has fewer than 32 cases")
            selected.extend(int(value) for value in rng.choice(candidates, 32, replace=False))
    return selected, boundaries


def stable_softmax(logits: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    values = logits.astype(np.float64, copy=True)
    values[mask] = -np.inf
    maximum = float(np.max(values))
    exponentials = np.exp(values - maximum)
    exponentials[mask] = 0.0
    total = float(exponentials.sum())
    return exponentials / total, maximum + math.log(total)


def production_mask(tokens: np.ndarray, ignore_tokens: Sequence[int]) -> np.ndarray:
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[np.asarray(ignore_tokens, dtype=np.int64)] = True
    seen = tokens[tokens > 1]
    mask[seen] = True
    return mask


class NumericMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.absolute_sum = 0.0
        self.square_sum = 0.0
        self.maximum = 0.0
        self.tolerance_count = 0
        self._errors: List[np.ndarray] = []

    def add(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        self.count += int(error.size)
        self.absolute_sum += float(error.sum())
        self.square_sum += float(np.square(error).sum())
        self.maximum = max(self.maximum, float(error.max(initial=0.0)))
        self.tolerance_count += int(
            np.count_nonzero(error <= ATOL + RTOL * np.abs(reference.astype(np.float64)))
        )
        self._errors.append(error.astype(np.float32))

    def summary(self) -> Dict[str, Any]:
        errors = np.concatenate(self._errors) if self._errors else np.zeros(1, dtype=np.float32)
        return {
            "count": self.count,
            "mean_absolute_error": self.absolute_sum / self.count,
            "root_mean_square_error": math.sqrt(self.square_sum / self.count),
            "median_absolute_error": float(np.quantile(errors, 0.5)),
            "p95_absolute_error": float(np.quantile(errors, 0.95)),
            "p99_absolute_error": float(np.quantile(errors, 0.99)),
            "maximum_absolute_error": self.maximum,
            "atol": ATOL,
            "rtol": RTOL,
            "within_tolerance_fraction": self.tolerance_count / self.count,
        }


class DistributionMetrics:
    def __init__(self) -> None:
        self.contexts = 0
        self.probability_count = 0
        self.probability_absolute_sum = 0.0
        self.probability_maximum = 0.0
        self.tv: List[float] = []
        self.log_rate_error: List[float] = []
        self.js_divergence: List[float] = []
        self.top1_matches = 0
        self.top5_overlap: List[float] = []
        self.top10_overlap: List[float] = []

    def add(self, ref_logits: np.ndarray, candidate_logits: np.ndarray, mask: np.ndarray) -> None:
        ref_prob, ref_log_rate = stable_softmax(ref_logits, mask)
        candidate_prob, candidate_log_rate = stable_softmax(candidate_logits, mask)
        delta = np.abs(ref_prob - candidate_prob)
        midpoint = 0.5 * (ref_prob + candidate_prob)
        ref_term = np.zeros_like(ref_prob)
        candidate_term = np.zeros_like(candidate_prob)
        ref_positive = ref_prob > 0
        candidate_positive = candidate_prob > 0
        ref_term[ref_positive] = ref_prob[ref_positive] * np.log(
            ref_prob[ref_positive] / midpoint[ref_positive]
        )
        candidate_term[candidate_positive] = candidate_prob[candidate_positive] * np.log(
            candidate_prob[candidate_positive] / midpoint[candidate_positive]
        )
        self.contexts += 1
        self.probability_count += VOCAB_SIZE
        self.probability_absolute_sum += float(delta.sum())
        self.probability_maximum = max(self.probability_maximum, float(delta.max()))
        self.tv.append(float(0.5 * delta.sum()))
        self.log_rate_error.append(abs(ref_log_rate - candidate_log_rate))
        self.js_divergence.append(float(0.5 * (ref_term.sum() + candidate_term.sum())))
        self.top1_matches += int(np.argmax(ref_prob) == np.argmax(candidate_prob))
        for k, target in ((5, self.top5_overlap), (10, self.top10_overlap)):
            left = set(np.argpartition(ref_prob, -k)[-k:].tolist())
            right = set(np.argpartition(candidate_prob, -k)[-k:].tolist())
            target.append(len(left & right) / k)

    @staticmethod
    def _distribution(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "maximum": float(array.max(initial=0.0)),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "contexts": self.contexts,
            "mean_absolute_probability_error": self.probability_absolute_sum
            / self.probability_count,
            "maximum_absolute_probability_error": self.probability_maximum,
            "total_variation_distance": self._distribution(self.tv),
            "jensen_shannon_divergence": self._distribution(self.js_divergence),
            "absolute_log_total_rate_error": self._distribution(self.log_rate_error),
            "top1_agreement": self.top1_matches / self.contexts,
            "mean_top5_overlap": float(np.mean(self.top5_overlap)),
            "mean_top10_overlap": float(np.mean(self.top10_overlap)),
        }


def percentile_summary(samples: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(values),
        "median_ms": float(np.median(values)),
        "q1_ms": float(np.quantile(values, 0.25)),
        "q3_ms": float(np.quantile(values, 0.75)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
        "samples_ms": [float(value) for value in values],
    }


def timed_runs(function, warmups: int, runs: int) -> Dict[str, Any]:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return percentile_summary(samples)


def system_metadata() -> Dict[str, Any]:
    chrome = None
    chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome_path.exists():
        chrome = subprocess.run([str(chrome_path), "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime_python": ort.__version__,
        "numpy": np.__version__,
        "chrome": chrome,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def evaluation_provenance(
    manifest_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    labels_path: Path,
) -> Dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    reference_directory = manifest_path.parent
    return {
        "manifest_sha256": sha256(manifest_path),
        "upstream_commit": manifest["upstream"]["commit"],
        "validation_sha256": manifest["upstream"]["validation_sha256"],
        "upstream_model_source_sha256": manifest["upstream"]["model_source_sha256"],
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx_sha256": sha256(onnx_path),
        "labels_sha256": sha256(labels_path),
        "all_final_reference_sha256": sha256(
            reference_directory / manifest["all_cohort"]["reference_file"]
        ),
        "stratified_full_reference_sha256": sha256(
            reference_directory / manifest["stratified"]["cases"][0]["logits_file"]
        ),
    }


def benchmark_native(
    model: Any,
    session: ort.InferenceSession,
    contexts: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    runs: int,
    warmups: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "definitions": {
            "steady_state": "Prebuilt tensors through inference with output materialized",
            "full_trajectory": (
                "Actual PyTorch generate(): 16-event prefix plus 32 sampled steps and its final "
                "all-position inference (33 graph executions); termination disabled"
            ),
        },
        "pytorch_cpu": {"steady_state": {}},
        "onnxruntime_python_cpu": {"steady_state": {}},
    }
    for length, cases in contexts.items():
        numpy_inputs = [
            (tokens[-length:][None, :], ages[-length:][None, :])
            for tokens, ages in cases
        ]
        torch_inputs = [
            (torch.from_numpy(idx), torch.from_numpy(age))
            for idx, age in numpy_inputs
        ]
        pytorch_index = 0
        ort_index = 0

        def pytorch_run() -> None:
            nonlocal pytorch_index
            idx, age = torch_inputs[pytorch_index % len(torch_inputs)]
            pytorch_index += 1
            with torch.inference_mode():
                output = model(idx, age)[0]
                _ = float(output[0, -1, 0])

        def ort_run() -> None:
            nonlocal ort_index
            idx_np, age_np = numpy_inputs[ort_index % len(numpy_inputs)]
            ort_index += 1
            output = session.run(["logits"], {"idx": idx_np, "age": age_np})[0]
            _ = float(output[0, -1, 0])

        results["pytorch_cpu"]["steady_state"][str(length)] = timed_runs(
            pytorch_run, warmups, runs
        )
        results["onnxruntime_python_cpu"]["steady_state"][str(length)] = timed_runs(
            ort_run, warmups, runs
        )

    prefix_tokens, prefix_ages = contexts[48][0]
    prefix_tokens = torch.from_numpy(prefix_tokens[-48:-32][None, :])
    prefix_ages = torch.from_numpy(prefix_ages[-48:-32][None, :])

    def trajectory_run() -> None:
        torch.manual_seed(BENCHMARK_SEED)
        with torch.inference_mode():
            output = model.generate(
                prefix_tokens.clone(),
                prefix_ages.clone(),
                max_new_tokens=32,
                max_age=1e12,
                termination_tokens=[-1],
            )
            _ = int(output[0][0, -1])

    results["pytorch_cpu"]["full_trajectory"] = timed_runs(
        trajectory_run, min(2, warmups), max(10, min(30, runs // 4))
    )
    results["pytorch_cpu"]["full_trajectory"]["graph_executions"] = 33
    results["pytorch_cpu"]["full_trajectory"]["generated_events"] = 32
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "OriginalModel.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "delphi.onnx")
    parser.add_argument("--labels", type=Path, default=ROOT / "delphi_labels_chapters_colours_icd.json")
    parser.add_argument("--val-bin", type=Path)
    parser.add_argument(
        "--upstream-model",
        type=Path,
        help="Pinned upstream model.py; downloaded and hash-verified when omitted",
    )
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "evaluation" / "generated")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Reuse an existing generated manifest and run only native latency benchmarks",
    )
    args = parser.parse_args()

    torch.manual_seed(BENCHMARK_SEED)
    np.random.seed(BENCHMARK_SEED)
    random.seed(BENCHMARK_SEED)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    references_dir = args.generated_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model_source = ensure_upstream_model(
        args.upstream_model, ROOT / "evaluation" / ".cache" / "upstream_model.py"
    )

    if args.latency_only:
        manifest_path = args.generated_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("--latency-only requires evaluation/generated/manifest.json")
        model, _, _ = load_checkpoint(args.checkpoint, model_source)
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        manifest = json.loads(manifest_path.read_text())
        context_lookup = {case["id"]: case for case in manifest["stratified"]["cases"]}
        latency_contexts: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for length in (12, 24, 48):
            latency_contexts[length] = []
            for case_id in manifest["benchmark_context_ids"][str(length)]:
                case = context_lookup[case_id]
                latency_contexts[length].append(
                    (
                        np.asarray(case["tokens"], dtype=np.int64),
                        np.asarray(case["ages_days"], dtype=np.float32),
                    )
                )
        latency = {
            "environment": system_metadata(),
            "evaluation_provenance": evaluation_provenance(
                manifest_path, args.checkpoint, args.onnx, args.labels
            ),
            "config": {"warmups": args.warmups, "runs_per_length": args.latency_runs},
            "results": benchmark_native(
                model, session, latency_contexts, args.latency_runs, args.warmups
            ),
        }
        (args.results_dir / "python_latency.json").write_text(json.dumps(latency, indent=2) + "\n")
        print(f"Wrote {args.results_dir / 'python_latency.json'}", flush=True)
        return

    val_path = ensure_val_data(args.val_bin, ROOT / "evaluation" / ".cache" / "val.bin")
    data = np.fromfile(val_path, dtype="<u4")
    if data.size % 3:
        raise ValueError("val.bin does not contain three uint32 columns")
    data = data.reshape(-1, 3)
    original_groups = group_rows(data)
    split_rows = next(rows for patient, rows in original_groups if patient == SPLIT_PATIENT_ID)
    groups = [(patient, rows) for patient, rows in original_groups if patient != SPLIT_PATIENT_ID]
    names, name_to_token = load_labels(args.labels)
    model, model_args, checkpoint = load_checkpoint(args.checkpoint, model_source)
    if model_args["vocab_size"] != VOCAB_SIZE or model_args["block_size"] != BLOCK_SIZE:
        raise ValueError("Evaluation constants do not match checkpoint metadata")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    selected_ids, quartile_boundaries = select_stratified(groups)
    if SELECTION_FIXTURE.exists():
        fixed_ids = json.loads(SELECTION_FIXTURE.read_text())["patient_ids"]
        if selected_ids != fixed_ids:
            raise ValueError(
                "Seeded stratified selection differs from the tracked patient-ID fixture"
            )
    selected_set = set(selected_ids)
    rows_by_id = {patient: rows for patient, rows in groups}
    prepared = {patient: prepare_context(rows) for patient, rows in groups}
    all_final_path = references_dir / "pytorch_all_final.f32"
    full_path = references_dir / "pytorch_stratified_full.f32"

    native_final_numeric = NumericMetrics()
    native_final_distributions = DistributionMetrics()
    legacy_tv: List[float] = []
    legacy_top1 = 0
    all_cases: List[Dict[str, Any]] = []
    print(f"Evaluating {len(groups):,} complete held-out trajectories", flush=True)
    with all_final_path.open("wb") as reference_handle, torch.inference_mode():
        for index, (patient_id, rows) in enumerate(groups):
            tokens, ages = prepared[patient_id]
            pt = model(torch.from_numpy(tokens[None, :]), torch.from_numpy(ages[None, :]))[0]
            pt_final = pt[0, -1].cpu().numpy().astype("<f4", copy=False)
            pt_final.tofile(reference_handle)
            candidate = session.run(
                ["logits"], {"idx": tokens[None, :], "age": ages[None, :]}
            )[0][0, -1]
            native_final_numeric.add(pt_final, candidate)
            mask = production_mask(tokens, model_args["ignore_tokens"])
            native_final_distributions.add(pt_final, candidate, mask)
            reference_prob, _ = stable_softmax(pt_final, mask)
            legacy_mask = np.zeros(VOCAB_SIZE, dtype=bool)
            legacy_mask[tokens[tokens > 1]] = True
            legacy_prob, _ = stable_softmax(pt_final, legacy_mask)
            legacy_tv.append(float(0.5 * np.abs(reference_prob - legacy_prob).sum()))
            legacy_top1 += int(np.argmax(reference_prob) == np.argmax(legacy_prob))
            all_cases.append(
                {
                    "patient_id": patient_id,
                    "raw_length": len(rows),
                    "context_length": len(tokens),
                    "sex": sex_name(rows),
                    "has_tied_age": bool(np.any(np.diff(rows[:, 1]) == 0)),
                    "ends_in_death": bool(int(rows[-1, 2]) == 1268),
                    "tokens": tokens.tolist(),
                    "ages_days": [float(value) for value in ages],
                    "reference_row": index,
                }
            )
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1:,}/{len(groups):,}", flush=True)

    full_cases: List[Dict[str, Any]] = []
    full_offset = 0
    native_full_numeric = NumericMetrics()
    native_full_distributions = DistributionMetrics()
    selected_order = [patient for patient in selected_ids]
    with full_path.open("wb") as reference_handle, torch.inference_mode():
        for patient_id in selected_order:
            rows = rows_by_id[patient_id]
            tokens, ages = prepared[patient_id]
            output = model(torch.from_numpy(tokens[None, :]), torch.from_numpy(ages[None, :]))[0]
            reference = output[0].cpu().numpy().astype("<f4", copy=False)
            reference.tofile(reference_handle)
            candidate = session.run(
                ["logits"], {"idx": tokens[None, :], "age": ages[None, :]}
            )[0][0]
            native_full_numeric.add(reference, candidate)
            for position in range(len(tokens)):
                mask = production_mask(tokens[: position + 1], model_args["ignore_tokens"])
                native_full_distributions.add(reference[position], candidate[position], mask)
            events = [names[int(token)] for token in tokens]
            element_count = int(reference.size)
            full_cases.append(
                {
                    "id": str(patient_id),
                    "source": "upstream_validation",
                    "patient_id": patient_id,
                    "sex": sex_name(rows),
                    "raw_length": len(rows),
                    "context_length": len(tokens),
                    "has_tied_age": bool(np.any(np.diff(rows[:, 1]) == 0)),
                    "ends_in_death": bool(int(rows[-1, 2]) == 1268),
                    "tokens": tokens.tolist(),
                    "ages_days": [float(value) for value in ages],
                    "events": events,
                    "ages_years": [float(np.float64(value) / YEAR_DAYS) for value in ages],
                    "logits_file": "references/pytorch_stratified_full.f32",
                    "logits_element_offset": full_offset,
                    "logits_element_count": element_count,
                    "logits_shape": [1, len(tokens), VOCAB_SIZE],
                    "logits_dtype": "float32-le",
                }
            )
            full_offset += element_count

        tokens, ages, events = authored_case(name_to_token)
        output = model(torch.from_numpy(tokens[None, :]), torch.from_numpy(ages[None, :]))[0]
        reference = output[0].cpu().numpy().astype("<f4", copy=False)
        reference.tofile(reference_handle)
        candidate = session.run(["logits"], {"idx": tokens[None, :], "age": ages[None, :]})[0][0]
        native_full_numeric.add(reference, candidate)
        for position in range(len(tokens)):
            mask = production_mask(tokens[: position + 1], model_args["ignore_tokens"])
            native_full_distributions.add(reference[position], candidate[position], mask)
        full_cases.append(
            {
                "id": "authored_lifestyle",
                "source": "README_authored_edge_case",
                "context_length": len(tokens),
                "tokens": tokens.tolist(),
                "ages_days": [float(value) for value in ages],
                "events": events,
                "ages_years": [float(np.float64(value) / YEAR_DAYS) for value in ages],
                "logits_file": "references/pytorch_stratified_full.f32",
                "logits_element_offset": full_offset,
                "logits_element_count": int(reference.size),
                "logits_shape": [1, len(tokens), VOCAB_SIZE],
                "logits_dtype": "float32-le",
            }
        )

    # Fixed real contexts for latency. Prefer selected cases long enough for each length.
    benchmark_contexts: Dict[str, List[str]] = {}
    for length in (12, 24, 48):
        eligible = [case["id"] for case in full_cases[:-1] if case["context_length"] >= length]
        benchmark_contexts[str(length)] = eligible[:30]
        if not eligible:
            raise ValueError(f"No benchmark context available for length {length}")

    manifest = {
        "schema_version": 1,
        "seeds": {"selection": SELECTION_SEED, "benchmark": BENCHMARK_SEED},
        "upstream": {
            "repository": "https://github.com/gerstung-lab/Delphi",
            "commit": UPSTREAM_COMMIT,
            "validation_url": VAL_URL,
            "validation_sha256": VAL_SHA256,
            "validation_bytes": val_path.stat().st_size,
            "split_patient_excluded": SPLIT_PATIENT_ID,
            "split_fragment_rows": len(split_rows),
            "model_source_url": MODEL_URL,
            "model_source_sha256": MODEL_SHA256,
        },
        "preprocessing": {
            "raw_columns": ["patient_id", "age_days", "raw_token_id"],
            "raw_token_shift": 1,
            "regular_no_event_interval_days": YEAR_DAYS * 5,
            "regular_no_event_first_age_days": 1,
            "window": "right",
            "final_record_reserved_as_target": True,
            "block_size": BLOCK_SIZE,
            "age_dtype": "float32",
            "token_dtype": "int64",
        },
        "checkpoint": {
            "path": args.checkpoint.name,
            "sha256": sha256(args.checkpoint),
            "model_args": model_args,
            "iteration": checkpoint.get("iter_num"),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "onnx": {"path": args.onnx.name, "sha256": sha256(args.onnx), "bytes": args.onnx.stat().st_size},
        "labels": {
            "path": args.labels.name,
            "sha256": sha256(args.labels),
            "bytes": args.labels.stat().st_size,
            "vocabulary_size": len(names),
        },
        "acceptance_criteria": {
            "logits_atol": ATOL,
            "logits_rtol": RTOL,
            "max_context_total_variation": 1e-4,
            "max_event_probability_error": 1e-4,
            "max_absolute_log_total_rate_error": 1e-4,
            "top1_agreement": 1.0,
        },
        "all_cohort": {
            "n_cases": len(all_cases),
            "reference_file": "references/pytorch_all_final.f32",
            "reference_shape": [len(all_cases), VOCAB_SIZE],
            "reference_dtype": "float32-le",
            "cases": all_cases,
        },
        "stratified": {
            "seed": SELECTION_SEED,
            "design": "32 cases per sex (female/male) by raw-length quartile",
            "raw_length_quartile_boundaries": quartile_boundaries,
            "selected_patient_ids": selected_ids,
            "cases": full_cases,
        },
        "benchmark_context_ids": benchmark_contexts,
    }
    (args.generated_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path = args.generated_dir / "manifest.json"
    provenance = evaluation_provenance(
        manifest_path, args.checkpoint, args.onnx, args.labels
    )

    legacy_array = np.asarray(legacy_tv, dtype=np.float64)
    fidelity = {
        "environment": system_metadata(),
        "evaluation_provenance": provenance,
        "scope": (
            "Synthetic-data checkpoint OriginalModel.pt; this is not the inaccessible "
            "published Delphi-2M checkpoint"
        ),
        "cohort": {
            "complete_held_out_trajectories": len(groups),
            "excluded_split_patient": SPLIT_PATIENT_ID,
            "raw_records": int(sum(len(rows) for _, rows in groups)),
            "stratified_full_position_cases": len(full_cases),
        },
        "pytorch_vs_python_onnxruntime": {
            "role": "Intermediate conversion diagnostic, not the browser fidelity endpoint",
            "all_cohort_final_logits": native_final_numeric.summary(),
            "all_cohort_final_distributions": native_final_distributions.summary(),
            "stratified_all_position_logits": native_full_numeric.summary(),
            "stratified_all_position_distributions": native_full_distributions.summary(),
        },
        "legacy_javascript_default_mask_diagnostic": {
            "description": (
                "Distribution change caused solely by the pre-evaluation SDK default "
                "ignoreTokens=[] rather than checkpoint ignore_tokens=[0,2,...,12]"
            ),
            "contexts": len(groups),
            "total_variation_distance": {
                "mean": float(legacy_array.mean()),
                "median": float(np.median(legacy_array)),
                "p95": float(np.quantile(legacy_array, 0.95)),
                "maximum": float(legacy_array.max()),
            },
            "top1_agreement": legacy_top1 / len(groups),
            "corrected_in_sdk": True,
        },
    }
    (args.results_dir / "python_fidelity.json").write_text(json.dumps(fidelity, indent=2) + "\n")

    if not args.skip_latency:
        context_lookup = {case["id"]: case for case in full_cases}
        latency_contexts: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for length in (12, 24, 48):
            latency_contexts[length] = []
            for case_id in benchmark_contexts[str(length)]:
                case = context_lookup[case_id]
                latency_contexts[length].append(
                    (
                        np.asarray(case["tokens"], dtype=np.int64),
                        np.asarray(case["ages_days"], dtype=np.float32),
                    )
                )
        latency = {
            "environment": system_metadata(),
            "evaluation_provenance": provenance,
            "config": {"warmups": args.warmups, "runs_per_length": args.latency_runs},
            "results": benchmark_native(
                model, session, latency_contexts, args.latency_runs, args.warmups
            ),
        }
        (args.results_dir / "python_latency.json").write_text(json.dumps(latency, indent=2) + "\n")

    print(f"Wrote {args.generated_dir / 'manifest.json'}", flush=True)
    print(f"Wrote results under {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
