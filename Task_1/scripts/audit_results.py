# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Audit downloaded Task 1 evidence without importing model/GPU libraries."""

import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Task_1.common.artifacts import ArtifactStore, atomic_json, read_json, sha256_file
from Task_1.common.config import load_configuration
from Task_1.common.paths import resolve_paths
from Task_1.common.replay import validate_replay
from Task_1.data.prepare import TASK_FILES, read_gzip_records, validate_task_data, verify_supplied_inputs
from Task_1.training.protocol import adaptive_pairs, experiment_conditions


def require(condition, message):
    """Stop the audit with the failed invariant, including under optimized Python."""
    if not condition:
        raise ValueError(message)


def audit_results(project_root=None):
    """Check source/data identity, budgets, scorer diagnostics, timings and coverage."""
    project = Path(project_root or Path(__file__).resolve().parents[2])
    configuration = load_configuration(project / "Task_1")
    paths = resolve_paths(configuration, project)
    root = paths["results"]
    manifest = validate_replay(configuration, paths)
    store = ArtifactStore(paths, configuration)
    stages = ["data", "allocation_plan", *[row["id"] for row in experiment_conditions()]]
    checks = {"stage_manifests": {stage: store.completed(stage) for stage in stages}}
    require(all(checks["stage_manifests"].values()), "A required stage is incomplete.")
    index = read_json(root / "backup_index.json")["files"]
    verified, locally_formatted, optional_missing = 0, [], []
    allowed_local = {"manifests/run.json", "figures/style_manifest.json",
                     *[relative for question in manifest["research_questions"].values()
                       for relative in question["figures"]]}
    for relative, expected in index.items():
        path = store.path(relative)
        if not path.is_file() and relative.startswith("checkpoints/"):
            optional_missing.append(relative)
            continue
        require(path.is_file() and path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
                f"Saved backup checksum differs: {relative}")
        if expected.get("source") == "local figure formatting":
            require(relative in allowed_local, f"A numerical artifact was labeled as local formatting: {relative}")
            locally_formatted.append(relative)
        else:
            verified += 1
    checks["cloud_files_verified"] = verified
    checks["locally_formatted_files_verified"] = locally_formatted
    if locally_formatted:
        from Task_1.analysis.presentation import valid_presentation
        require(valid_presentation(root, read_json(root / "figures/style_manifest.json")),
                "The main figure style/input hashes failed verification.")
    checks["optional_recovery_files_not_downloaded"] = optional_missing
    data = {name: read_gzip_records(root / "splits/materialized" / filename) for name, filename in TASK_FILES.items()}
    checks["data_protocol"] = validate_task_data(data, configuration, project / "Task_1/resources/course_data")
    selection = read_json(root / "splits/selection.json")
    require(selection["supplied_input_hashes"] == verify_supplied_inputs(project / "Task_1/resources/course_data"),
            "Supplied source/manifests differ from the run's recorded inputs.")
    profiles = read_json(root / "metrics/sensitivity_profiles.json")
    inventory = read_json(root / "metrics/model_inventory.json")
    require(len(profiles) == len(inventory["projections"]) == 196, "Projection coverage is incomplete.")
    plans = read_json(root / "configs/allocations_locked.json")
    checks["pruning_allocations"] = {}
    for target, plan in plans["pruning"].items():
        allocated = sum(item["zero_count"] for item in plan["allocations"].values())
        required = round(float(target) * inventory["eligible_parameters"])
        require(allocated == required == plan["zero_count"], "Adaptive pruning violates its global zero-count budget.")
        require(set(plan["allocations"]) == set(profiles), "Adaptive pruning omits eligible projections.")
        require(all(.10 - 1e-8 <= item["sparsity"] <= .85 + 1e-8 for item in plan["allocations"].values()),
                "Adaptive pruning exceeds the permitted per-projection bounds.")
        candidates = plan["candidate_results"]
        best = min(candidates, key=lambda row: (row["development"]["mean_nll"], row["allocation"]["temperature"]))
        require(best["allocation"]["temperature"] == plan["temperature"], "Development selection differs from its fixed rule.")
        checks["pruning_allocations"][target] = {"zeros": allocated, "development_candidates": len(candidates),
                                                 "selected_temperature": plan["temperature"]}
    checks["quantization_allocations"] = {}
    for bits, plan in plans["quantization"].items():
        require(set(plan["allocations"]) == set(profiles) and set(plan["allocations"].values()).issubset({2, 3, 4, 8}),
                "Mixed precision has missing projections or unsupported bit widths.")
        actual = read_json(root / f"metrics/q3_budget_int{bits}/summary.json")["storage"]["estimated_packed_model_bytes"]
        reference = read_json(root / f"metrics/q2_int{bits}/summary.json")["storage"]["estimated_packed_model_bytes"]
        require(actual == plan["packed_bytes"] <= reference == plan["reference_budget_bytes"],
                "Mixed precision violates its metadata-inclusive byte budget.")
        checks["quantization_allocations"][bits] = {"actual_bytes": actual, "reference_bytes": reference,
                                                    "bit_counts": dict(Counter(plan["allocations"].values()))}
    reference = read_json(root / "metrics/reference_diagnostics.json")
    rescaling = read_json(root / "metrics/awq_full_model_rescaling.json")
    local_error = max(profile["awq"][str(bits)]["rescaling_equivalence"]["relative_l2"]
                      for profile in profiles.values() for bits in [2, 3, 4, 8])
    workload = read_json(root / "metrics/gpu_workload_plan.json")
    require(reference["status"] == "passed" and reference["absolute_error"] <= .003, "The causal scorer reference check failed.")
    require(set(rescaling) == {"2", "3", "4", "8"} and max(row["relative_l2"] for row in rescaling.values()) <= .02,
            "The whole-model FP16 rescaling check failed.")
    require(local_error <= 2e-5 and workload["batch_reference_maximum_mean_nll_error"] <= .003,
            "Local rescaling or batched scorer equivalence failed.")
    checks["scorer_and_rescaling"] = {"causal_reference_error": reference["absolute_error"],
                                     "maximum_local_rescaling_relative_l2": local_error,
                                     "maximum_whole_model_fp16_rescaling_relative_l2": max(row["relative_l2"] for row in rescaling.values()),
                                     "batched_reference_mean_nll_error": workload["batch_reference_maximum_mean_nll_error"]}
    checks["conditions"] = []
    for condition in experiment_conditions():
        name = condition["id"]
        summary = read_json(root / f"metrics/{name}/summary.json")
        require(summary["status"] == "complete" and summary["runtime"] == manifest["runtime"],
                f"Incomplete or incompatible runtime for {name}.")
        for length in ["512", "2048"]:
            timing = summary["latency"]["shapes"][length]
            require(all(len(timing[key]) >= 5 and all(math.isfinite(value) and value > 0 for value in timing[key])
                        for key in ["prefill_ms", "decode_64_steps_ms"]), f"Incomplete or invalid timings for {name}.")
            require(bool(timing["sdpa_observed_operators"]) and timing["peak_allocated_bytes"] > 0,
                    f"Missing measured GPU/backend evidence for {name}.")
        checks["conditions"].append({"condition": name, "status": summary["status"],
                                     "wikitext2_targets": summary["quality"]["wikitext2"]["scored_tokens"],
                                     "c4_targets": summary["quality"]["c4"]["scored_tokens"],
                                     "hellaswag_examples": summary["quality"]["hellaswag"]["examples"]})
    intervals = read_json(root / "metrics/paired_comparisons.json")
    require(len(intervals) == len(adaptive_pairs()) * 6 and all(row["status"] == "complete" and row["resamples"] >= 2000
            and row["ci_low"] <= row["ci_high"] for row in intervals), "Incomplete paired bootstrap evidence.")
    checks["paired_intervals"] = len(intervals)
    checks["registered_artifacts_verified"] = len(manifest["required_artifacts"])
    checks["checkpoint_files_verified"] = len(list((root / "checkpoints").rglob("*.pt")))
    audit = {"audited_on": "Local ATML; saved evidence only", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
             "run_id": manifest["run_id"], "computed_on": manifest["computed_on"], "gpu": manifest["runtime"]["gpu"],
             "status": "passed", "gpu_rerun_required": False, "checks": checks,
             "interpretation_notes": ["Retain unsuccessful adaptive outcomes; do not choose replacements using held-out results.",
                                      "This audit checks saved evidence and does not repeat pretrained inference."]}
    atomic_json(root / "metrics/local_run_audit.json", audit)
    print(f"PASS: {len(checks['conditions'])} conditions; {verified} unchanged Drive files; "
          f"{len(locally_formatted)} locally formatted files; {len(intervals)} paired intervals. No GPU rerun required.")
    return audit


if __name__ == "__main__":
    audit_results()
