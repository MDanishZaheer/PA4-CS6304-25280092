# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build required Task 1 tables, plots, paired intervals, and aligned examples."""

import math

import numpy as np
import pandas as pd

from Task_1.common.artifacts import read_json, read_jsonl, sha256_file
from Task_1.evaluation.statistics import align_records, paired_bootstrap
from Task_1.training.protocol import adaptive_pairs, experiment_conditions


def save_table(store, relative, rows):
    """Save a machine-readable table before displaying or plotting it."""
    path = store.path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8")
    store.backup([relative])
    return frame


def summary_table(store, summaries):
    """Compare every method with FP16 on the same HellaSwag IDs and token windows."""
    rows = []
    baseline = summaries["fp16"]
    baseline_predictions = read_jsonl(store.path("predictions/fp16/hellaswag.jsonl"))
    for condition in experiment_conditions():
        name = condition["id"]
        summary = summaries[name]
        row = {"condition": name, "family": condition["family"], "status": summary["status"],
               "hellaswag_required_examples": condition["hellaswag_count"],
               "compression_seconds": summary.get("compression_seconds_including_restore"),
               "condition_peak_allocated_bytes": summary.get("condition_peak_allocated_bytes"),
               "calibration_relative_mse": summary.get("mean_projection_calibration_relative_mse")}
        storage = summary.get("storage", {})
        for key in ["achieved_sparsity", "estimated_packed_model_bytes", "nominal_bits_per_eligible_weight", "effective_nonzeros"]:
            row[key] = storage.get(key)
        quality = summary.get("quality", {})
        for suite in ["wikitext2", "c4"]:
            metrics = quality.get(suite, {})
            row[f"{suite}_scored_tokens"] = metrics.get("scored_tokens")
            row[f"{suite}_mean_nll"] = metrics.get("mean_nll")
            row[f"{suite}_perplexity"] = metrics.get("perplexity")
            row[f"{suite}_perplexity_overflow"] = metrics.get("perplexity_overflow", False)
            baseline_nll = baseline.get("quality", {}).get(suite, {}).get("mean_nll")
            row[f"{suite}_nll_delta_from_fp16"] = metrics["mean_nll"] - baseline_nll if "mean_nll" in metrics and baseline_nll is not None else None
        hellaswag = quality.get("hellaswag", {})
        row["hellaswag_accuracy"] = hellaswag.get("accuracy")
        row["hellaswag_accuracy_normalized"] = hellaswag.get("accuracy_normalized")
        row["hellaswag_evaluated_examples"] = hellaswag.get("examples", 0)
        row["latency_benchmark_batch_size"] = summary.get("latency", {}).get("batch_size", store.configuration["benchmark"]["batch_size"])
        predictions = read_jsonl(store.path(f"predictions/{name}/hellaswag.jsonl"))
        diagnostic = predictions[:512]
        if len(diagnostic) == 512:
            row["hellaswag_diagnostic_accuracy"] = np.mean([item["correct"] for item in diagnostic])
            row["hellaswag_diagnostic_accuracy_normalized"] = np.mean([item["correct_normalized"] for item in diagnostic])
        if len(predictions) == condition["hellaswag_count"] and baseline_predictions:
            pairs = align_records(predictions, baseline_predictions)
            row["hellaswag_paired_accuracy_delta"] = np.mean([a["correct"] - b["correct"] for a, b in pairs])
            row["hellaswag_paired_normalized_accuracy_delta"] = np.mean([a["correct_normalized"] - b["correct_normalized"] for a, b in pairs])
            row["hellaswag_paired_baseline_examples"] = len(pairs)
        for length in [512, 2048]:
            latency = summary.get("latency", {}).get("shapes", {}).get(str(length), {})
            row[f"prefill_{length}_median_ms"] = latency.get("prefill_median_ms")
            row[f"decode_{length}_64_steps_median_ms"] = latency.get("decode_median_ms")
            row[f"peak_{length}_allocated_bytes"] = latency.get("peak_allocated_bytes")
        rows.append(row)
    return save_table(store, "tables/all_conditions.csv", rows)


def confidence_intervals(store, summaries):
    """Produce every required adaptive-minus-uniform paired 95% interval."""
    rows = []
    for method, reference in adaptive_pairs():
        for suite, metrics in [("wikitext2", ["mean_nll", "perplexity"]),
                               ("c4", ["mean_nll", "perplexity"]),
                               ("hellaswag", ["accuracy", "accuracy_normalized"])]:
            left = read_jsonl(store.path(f"predictions/{method}/{suite}.jsonl"))
            right = read_jsonl(store.path(f"predictions/{reference}/{suite}.jsonl"))
            left_complete = suite in summaries[method].get("quality", {})
            right_complete = suite in summaries[reference].get("quality", {})
            for metric in metrics:
                if not left_complete or not right_complete:
                    rows.append({"method": method, "reference": reference, "suite": suite, "metric": metric,
                                 "status": "unavailable_due_to_recorded_condition_failure"})
                    continue
                result = paired_bootstrap(left, right, metric,
                    store.configuration["evaluation"]["bootstrap_resamples"], store.configuration["seed"])
                rows.append({"method": method, "reference": reference, "suite": suite,
                             "status": "overflow" if result.get("overflow") else "complete", **result})
    store.save_json("metrics/paired_comparisons.json", rows)
    return save_table(store, "tables/paired_comparisons.csv", rows)


def allocation_tables(store, profiles, plans):
    """Save both sensitivity maps and the final sparsity/precision assignments."""
    rows = []
    for name, profile in profiles.items():
        row = {"projection": name, "layer": profile["layer"], "component": name.split(".")[-1],
               "elements": profile["elements"], "pruning_sensitivity": profile["pruning_sensitivity"],
               "quantization_sensitivity": profile["quantization_sensitivity"],
               "probe_seconds": profile["probe_seconds"],
               "activation_norm_median": profile["activation_norms"]["quantiles"][4],
               "activation_norm_p99": profile["activation_norms"]["quantiles"][7],
               "magnitude_median": profile["magnitude_scores"]["quantiles"][4],
               "wanda_score_median": profile["wanda_scores"]["quantiles"][4]}
        for target in ["0.5", "0.75"]:
            row[f"p3_sparsity_{target}"] = plans["pruning"][target]["allocations"][name]["sparsity"]
        for bits in ["4", "3"]:
            row[f"q3_bits_budget_int{bits}"] = plans["quantization"][bits]["allocations"][name]
        rows.append(row)
    frame = save_table(store, "tables/sensitivity_and_allocations.csv", rows)
    curve_rows = []
    for name, profile in profiles.items():
        for family in ["magnitude", "wanda"]:
            for sparsity, error in profile[family].items():
                curve_rows.append({"projection": name, "family": family, "sparsity": float(sparsity), **error})
    curves = save_table(store, "tables/pruning_local_errors.csv", curve_rows)
    budget_rows = []
    for target, plan in plans["pruning"].items():
        actual = sum(item["zero_count"] for item in plan["allocations"].values())
        budget_rows.append({"method": f"p3_s{round(float(target) * 100)}", "target": float(target),
                            "required_zeros": plan["zero_count"], "allocated_zeros": actual,
                            "temperature": plan["temperature"], "development_mean_nll": plan["development"]["mean_nll"],
                            "matched": actual == plan["zero_count"]})
    for bits, plan in plans["quantization"].items():
        budget_rows.append({"method": f"q3_budget_int{bits}", "reference_bytes": plan["reference_budget_bytes"],
                            "allocated_bytes": plan["packed_bytes"], "unused_bytes": plan["unused_bytes"],
                            "matched": plan["packed_bytes"] <= plan["reference_budget_bytes"]})
    save_table(store, "tables/budget_checks.csv", budget_rows)
    return frame, curves


def qualitative_examples(store):
    """Use fixed cases plus the first paired adaptive disagreements in manifest order."""
    by_condition = {condition["id"]: read_jsonl(store.path(f"predictions/{condition['id']}/hellaswag.jsonl"))
                    for condition in experiment_conditions()}
    fixed = set(store.configuration["evaluation"]["qualitative_orders"])
    extra = set()
    for method, reference in adaptive_pairs():
        if not by_condition[method] or not by_condition[reference]:
            continue
        disagreements = [(a, b) for a, b in align_records(by_condition[method], by_condition[reference])
                         if a["prediction_normalized"] != b["prediction_normalized"]]
        extra.update(a["order"] for a, _ in disagreements[:4])
    examples = []
    for name, records in by_condition.items():
        for record in records:
            if record["order"] in fixed | extra:
                examples.append({**record, "selection": "fixed_manifest_order" if record["order"] in fixed else "first_four_adaptive_pair_disagreements"})
    store.save_json("examples/hellaswag_aligned.json", {
        "selection_rule": "Fixed orders from configuration plus first four normalized-prediction disagreements per declared adaptive/reference pair, in manifest order.",
        "examples": examples})
    table_rows = [{key: record[key] for key in ["condition", "order", "gold", "prediction", "prediction_normalized", "correct", "correct_normalized", "selection"]}
                  for record in examples]
    save_table(store, "tables/qualitative_cases.csv", table_rows)


def create_figures(store, frame, maps, curves, profiles, *, font_family="DejaVu Serif",
                   font_size=10, output_directory="figures", uniform_text=False,
                   style_records=None):
    """Draw compact scientific figures entirely from saved numerical evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text

    plt.rcParams.update({"font.family": font_family, "font.size": font_size, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    if uniform_text:
        font_manager.findfont(font_manager.FontProperties(family=font_family), fallback_to_default=False)
        plt.rcParams.update({"axes.titlesize": font_size, "axes.labelsize": font_size,
                             "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                             "legend.fontsize": font_size, "figure.titlesize": font_size,
                             "mathtext.fontset": "custom", "mathtext.rm": font_family,
                             "mathtext.it": f"{font_family}:italic",
                             "mathtext.bf": f"{font_family}:bold", "mathtext.default": "regular",
                             "mathtext.fallback": None})
    generated = []
    def save(figure, filename):
        """Register a finished PNG with the durable evidence set."""
        relative = f"{output_directory}/{filename}.png"
        figure.canvas.draw()
        if uniform_text:
            for text in figure.findobj(Text):
                text.set_fontfamily(font_family)
                text.set_fontsize(font_size)
        figure.tight_layout()
        figure.canvas.draw()
        if style_records is not None:
            texts = [text for text in figure.findobj(Text) if text.get_text()]
            style_records[relative] = {
                "text_objects": len(texts),
                "font_sizes_pt": sorted({float(text.get_fontsize()) for text in texts}),
                "font_families": sorted({family for text in texts for family in text.get_fontfamily()}),
                "resolved_font_files": sorted({font_manager.findfont(text.get_fontproperties(),
                    fallback_to_default=False) for text in texts}),
            }
            if uniform_text and (style_records[relative]["font_sizes_pt"] != [float(font_size)]
                                 or style_records[relative]["font_families"] != [font_family]):
                raise ValueError(f"Figure text does not match the requested style: {relative}")
        figure.savefig(store.path(relative), dpi=300, bbox_inches="tight")
        plt.close(figure)
        store.backup([relative])
        generated.append(relative)

    store.path(output_directory).mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for family, group in curves.groupby("family"):
        pooled = group.groupby("sparsity")[["squared_error", "squared_reference"]].sum()
        axes[0].plot(pooled.index, pooled["squared_error"] / pooled["squared_reference"], marker="o", label=family)
    axes[0].set(xlabel="Per-projection sparsity", ylabel="Pooled local output relative MSE")
    axes[0].legend()
    for name, profile in profiles.items():
        delta = profile["magnitude"]["0.5"]["relative_mse"] - profile["wanda"]["0.5"]["relative_mse"]
        norms = profile["activation_norms"]["quantiles"]
        axes[1].scatter(norms[7] / max(norms[4], 1e-12), delta, s=10, alpha=.5)
    axes[1].axhline(0, color="grey", linewidth=.8)
    axes[1].set(xlabel="Input activation norm p99 / median", ylabel="Magnitude minus Wanda local MSE (50%)")
    save(figure, "rq1_pruning_mechanism")

    fields = ["pruning_sensitivity", "quantization_sensitivity", "p3_sparsity_0.5", "p3_sparsity_0.75", "q3_bits_budget_int4", "q3_bits_budget_int3"]
    figure, axes = plt.subplots(3, 2, figsize=(10, 10))
    for axis, field in zip(axes.flat, fields):
        matrix = maps.pivot(index="layer", columns="component", values=field)
        values = matrix.to_numpy(dtype=float)
        if "sensitivity" in field:
            values = np.log10(np.maximum(values, 1e-12))
        graphic = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(0, len(matrix.index), 4), matrix.index[::4])
        axis.set(ylabel="Transformer layer", title=field.replace("_", " "))
        figure.colorbar(graphic, ax=axis, label="log10 relative MSE" if "sensitivity" in field else "sparsity" if "sparsity" in field else "bits")
    save(figure, "rq2_sensitivity_and_allocations")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for family in ["P1", "P2", "P3", "structured"]:
        group = frame[frame["family"] == family].sort_values("achieved_sparsity")
        axes[0, 0].plot(group["achieved_sparsity"], group["wikitext2_perplexity"], marker="o", label=family)
        if "hellaswag_diagnostic_accuracy_normalized" in group:
            axes[1, 0].plot(group["achieved_sparsity"], 100 * group["hellaswag_diagnostic_accuracy_normalized"], marker="o", label=family)
    for family in ["Q1", "Q2", "Q3"]:
        group = frame[frame["family"] == family].sort_values("estimated_packed_model_bytes")
        axes[0, 1].plot(group["estimated_packed_model_bytes"] / 2**30, group["wikitext2_perplexity"], marker="o", label=family)
        if "hellaswag_diagnostic_accuracy_normalized" in group:
            axes[1, 1].plot(group["estimated_packed_model_bytes"] / 2**30, 100 * group["hellaswag_diagnostic_accuracy_normalized"], marker="o", label=family)
    for axis in axes[0]:
        axis.set_yscale("log")
        axis.set_ylabel("WikiText-2 perplexity (log axis)")
    for axis in axes[1]:
        axis.set_ylabel("HellaSwag acc_norm, paired 512 IDs (%)")
    for axis in axes[:, 0]:
        axis.set_xlabel("Achieved eligible-weight sparsity")
    for axis in axes[:, 1]:
        axis.set_xlabel("Estimated packed model storage (GiB)")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    save(figure, "rq3_resource_quality")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for family, group in frame.groupby("family"):
        axes[0].scatter(group["calibration_relative_mse"], group["wikitext2_nll_delta_from_fp16"], label=family)
        axes[1].scatter(group["calibration_relative_mse"], group["c4_nll_delta_from_fp16"], label=family)
    for axis, suite in zip(axes, ["WikiText-2", "C4 validation"]):
        axis.set(xlabel="Mean projection calibration relative MSE", ylabel=f"{suite} NLL change (nats/token)")
        axis.legend(fontsize=7)
    save(figure, "rq3_calibration_vs_heldout")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    positions = np.arange(len(frame))
    axes[0].bar(positions - .18, frame["estimated_packed_model_bytes"] / 2**30, width=.36, label="Estimated packed storage")
    axes[0].bar(positions + .18, frame["peak_2048_allocated_bytes"] / 2**30, width=.36, label="Measured peak allocated, dense FP16")
    axes[0].set(ylabel="GiB", xticks=positions, xticklabels=frame["condition"])
    axes[0].tick_params(axis="x", labelrotation=65)
    axes[0].legend(fontsize=8)
    axes[1].plot(positions, frame["prefill_512_median_ms"], marker="o", label="512-token prefill")
    axes[1].plot(positions, frame["prefill_2048_median_ms"], marker="o", label="2048-token prefill")
    axes[1].plot(positions, frame["decode_2048_64_steps_median_ms"], marker="o", label="64 cached decode steps after 2048")
    batch_size = store.configuration["benchmark"]["batch_size"]
    axes[1].set(ylabel=f"Dense FP16 latency (ms/batch; B={batch_size})", xticks=positions, xticklabels=frame["condition"])
    axes[1].tick_params(axis="x", labelrotation=65)
    axes[1].legend(fontsize=8)
    save(figure, "rq4_storage_memory_latency")
    return generated


def build_evidence(store):
    """Register a complete evidence manifest, retaining failed required conditions."""
    summaries = {condition["id"]: read_json(store.path(f"metrics/{condition['id']}/summary.json"))
                 for condition in experiment_conditions()}
    profiles = read_json(store.path("metrics/sensitivity_profiles.json"))
    plans = read_json(store.path("configs/allocations_locked.json"))
    frame = summary_table(store, summaries)
    confidence_intervals(store, summaries)
    maps, curves = allocation_tables(store, profiles, plans)
    qualitative_examples(store)
    cost = {"activation_collection": read_json(store.path("metrics/calibration.json")),
            "projection_probe_seconds": sum(profile["probe_seconds"] for profile in profiles.values()),
            "projection_probe_peak_allocated_bytes": max(profile.get("probe_peak_allocated_bytes", 0) for profile in profiles.values()),
            "development_candidate_seconds": sum(candidate["seconds"] for plan in plans["pruning"].values() for candidate in plan["candidate_results"]),
            "precision_allocation": "deterministic greedy calibration distortion reduction per byte; no test selection",
            "selection_limitation": "Isolated projection errors ignore accumulated cross-layer and folding interactions."}
    if store.path("metrics/gpu_workload_plan.json").is_file():
        workload = read_json(store.path("metrics/gpu_workload_plan.json"))
        cost["gpu_workload_tuning_seconds"] = workload["tuning_seconds"]
        cost["gpu_workload"] = {key: workload[key] for key in ["batch_sequences", "calibration_batch_sequences", "head_chunk_tokens",
                                                               "selected_probe_peak_bytes", "target_memory_fraction", "latency_benchmark_batch_size"]}
    store.save_json("metrics/calibration_and_selection_cost.json", cost)
    correlations = []
    for family, group in frame.groupby("family"):
        for suite in ["wikitext2", "c4"]:
            subset = group[["calibration_relative_mse", f"{suite}_nll_delta_from_fp16"]].dropna()
            correlation = subset.corr().iloc[0, 1] if len(subset) >= 3 else float("nan")
            correlations.append({"family": family, "suite": suite, "conditions": len(subset),
                                 "pearson_correlation": float(correlation) if math.isfinite(correlation) else None,
                                 "note": "descriptive across conditions; not a causal or generalization claim"})
    store.save_json("metrics/calibration_vs_quality.json", correlations)
    figures = create_figures(store, frame, maps, curves, profiles)
    questions = {
        "rq1": {"tables": ["tables/pruning_local_errors.csv"], "figures": [figures[0]], "metrics": []},
        "rq2": {"tables": ["tables/sensitivity_and_allocations.csv", "tables/budget_checks.csv"], "figures": [figures[1]], "metrics": []},
        "rq3": {"tables": ["tables/all_conditions.csv", "tables/paired_comparisons.csv", "tables/qualitative_cases.csv"],
                "figures": [figures[2], figures[3]], "metrics": ["metrics/calibration_vs_quality.json"]},
        "rq4": {"tables": ["tables/all_conditions.csv"], "figures": [figures[4]], "metrics": ["metrics/calibration_and_selection_cost.json"]},
    }
    lock = read_json(store.path("manifests/run_lock.json"))
    required = {}
    for path in sorted(store.root.rglob("*")):
        relative = path.relative_to(store.root).as_posix()
        if path.is_file() and not relative.startswith("checkpoints/") and relative not in {"manifests/run.json", "backup_index.json"} and not relative.endswith((".partial", ".uploading", ".previous")):
            required[relative] = sha256_file(path)
    failures = {name: summary["status"] for name, summary in summaries.items() if summary["status"] != "complete"}
    manifest = {"task": "Task_1", "run_id": lock["run_id"], "config_id": lock["config_id"],
                "data_id": lock["data_id"], "computed_on": "Google Colab", "runtime": lock["runtime"],
                "status": "complete_with_recorded_failures" if failures else "complete", "recorded_failures": failures,
                "conditions": [condition["id"] for condition in experiment_conditions()],
                "required_artifacts": required, "research_questions": questions,
                "report_analysis": "Student-authored interpretation required; these artifacts contain measurements and diagnostics."}
    store.save_json("manifests/run.json", manifest)
    return {"status": manifest["status"], "conditions": len(summaries), "recorded_failures": failures,
            "tables": 6, "figures": len(figures), "results": str(store.root)}
