# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate Task 2 tables and eight main figures from authentic saved evidence."""

import math
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from Task_2.analysis.fonts import configure_fonts, save_figure
from Task_2.common.artifacts import read_json, read_jsonl, sha256_file
from Task_2.common.config import fingerprint
from Task_2.data.mqar import read_gzip_records
from Task_2.training.protocol import evaluation_conditions, training_conditions


FIGURE_NAMES = ["rq1_exact_attention", "rq2_learning_curves", "rq2_distance_load", "rq2_regime_controls",
                "rq3_kv_fidelity", "rq3_kv_factorial", "rq4_packed_cache_systems", "rq4_resources_and_overheads"]


def save_table(store, relative, records, columns=None):
    """Persist machine-readable values before using the same data in a figure."""
    frame = pd.DataFrame(records, columns=columns if not records else None)
    store.path(relative).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(store.path(relative), index=False, encoding="utf-8")
    store.backup([relative])
    return frame


def collect_tables(store):
    """Collect all required conditions, seed variation, factorial cells and system costs."""
    attention = []
    for row in read_json(store.path("metrics/attention/all_conditions.json")):
        attention.append({key: row[key] for key in ["method", "dtype", "length", "status"]} | {
            "observed_operators": "; ".join(row.get("observed_operators", [])),
            "reference_status": row["reference_status"], "output_relative_l2": row.get("output_error", {}).get("relative_l2"),
            "gradient_maximum_relative_l2": max((value["relative_l2"] for value in row.get("gradient_errors", [])), default=None),
            "forward_ms": row.get("forward", {}).get("median_ms"), "forward_backward_ms": row.get("forward_backward", {}).get("median_ms"),
            "forward_peak_allocated_bytes": row.get("forward", {}).get("peak_allocated_bytes"),
            "forward_backward_peak_allocated_bytes": row.get("forward_backward", {}).get("peak_allocated_bytes"),
            "forward_backward_peak_reserved_bytes": row.get("forward_backward", {}).get("peak_reserved_bytes"),
            "timing_repetitions": row.get("forward", {}).get("timing_repetitions", 0)})
    a = save_table(store, "tables/attention_conditions.csv", attention)
    training, histories = [], []
    for condition in training_conditions(store.configuration):
        row = read_json(store.path(f"metrics/mqar_training/{condition['id']}.json"))
        history = read_jsonl(store.path(row["history"]))
        training.append({**condition, **{key: row[key] for key in ["status", "completed_updates", "trainable_parameters", "training_seconds",
            "skipped_optimizer_attempts", "peak_allocated_bytes", "initialization_sha256", "final_sha256"]},
            "final_training_loss": history[-1]["loss"] if history else None})
        histories.extend({**condition, **entry} for entry in history)
    train = save_table(store, "tables/mqar_training.csv", training)
    curves = save_table(store, "tables/mqar_learning_curves.csv", histories,
                        columns=["method", "train_K", "train_regime", "seed", "update", "loss"])
    recall, distances = [], []
    for condition in evaluation_conditions(store.configuration):
        row = read_json(store.path(f"metrics/mqar_test/{condition['id']}.json"))
        cells = row["cells"] if row["status"] == "complete" else [
            {"test_K": count, "regime": regime, "examples": 0, "queries": 0, "correct": 0,
             "accuracy": None, "mean_query_nll": None, "chance_label_space": 1 / 64,
             "chance_present_values": 1 / count,
             "distance_bins": {name: {"queries": 0, "correct": 0, "accuracy": None}
                               for name in ["0_32", "32_64", "64_128", "128_256"]}}
            for count in [4, 8, 16] for regime in ["near", "far", "mixed"]]
        for cell in cells:
            metadata = {**condition, "status": row["status"], "test_K": cell["test_K"], "regime": cell["regime"]}
            recall.append({**metadata, **{key: value for key, value in cell.items() if key != "distance_bins"}})
            for name, values in cell["distance_bins"].items():
                distances.append({**metadata, "distance_bin": name, **values})
    r = save_table(store, "tables/mqar_accuracy.csv", recall)
    d = save_table(store, "tables/mqar_distance_bins.csv", distances)
    if not r.empty:
        seed_groups = ["method", "train_K", "train_regime", "test_method", "test_only_mask", "test_K", "regime"]
        variation = r.groupby(seed_groups, dropna=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
        save_table(store, "tables/mqar_seed_dispersion.csv", variation.to_dict("records"))
    else:
        save_table(store, "tables/mqar_seed_dispersion.csv", [],
                   columns=["method", "train_K", "train_regime", "test_method", "test_only_mask", "test_K", "regime", "mean", "std", "count"])
    inference = []
    for row in read_json(store.path("metrics/mqar_inference/all_conditions.json")):
        inference.append({key: value for key, value in row.items() if key != "timing"} | {
            "median_ms": row.get("timing", {}).get("median_ms"), "peak_allocated_bytes": row.get("timing", {}).get("peak_allocated_bytes"),
            "additional_peak_allocated_bytes": row.get("timing", {}).get("additional_peak_allocated_bytes")})
    sweep = save_table(store, "tables/mqar_operator_resources.csv", inference)
    records = [read_json(path) for path in sorted(store.path("predictions/kv").glob("*/*.json"))]
    summary, cells = [], []
    for method in store.configuration["kv"]["methods"]:
        group = [row for row in records if row["method"] == method]
        complete = [row for row in group if row["status"] == "complete"]
        def statistics(rows):
            """Use prompt denominators for accuracy and pooled answer-token NLL."""
            good = [row for row in rows if row["status"] == "complete"]
            reference_correct = [row for row in rows if row.get("fp16_reference_correct")]
            return {"prompts": len(rows), "completed_prompts": len(good), "accuracy": sum(row.get("correct", False) for row in rows) / len(rows),
                "conditional_on_fp16_correct_accuracy": sum(row.get("correct", False) for row in reference_correct) / len(reference_correct) if reference_correct else None,
                "fp16_correct_prompts": len(reference_correct),
                "reference_answer_mean_nll": sum(row["reference_answer_nll"]["nll_sum"] for row in good) / sum(row["reference_answer_nll"]["tokens"] for row in good) if good else None,
                **{field: float(np.mean([row[field] for row in good])) if good else None for field in ["post_prefill_logit_kl", "reference_answer_teacher_forced_kl",
                    "actual_serialized_prefix_bytes", "fraction_full_fp16_bytes", "evidence_retention_fraction", "prefill_seconds", "question_importance_seconds",
                    "packed_encode_seconds", "question_suffix_seconds", "teacher_forced_scoring_seconds", "peak_allocated_bytes"]},
                "mean_decode_seconds": float(np.mean([row["response"]["decode_seconds"] for row in good])) if good else None,
                "truncated_prompts": sum(row["response"]["truncated"] for row in good)}
        summary.append({"method": method, **statistics(group)})
        for length in [2048, 4096]:
            for position in ["start", "middle", "end"]:
                for distractors in [0, 2, 8]:
                    selected = [row for row in group if (row["target_tokens"], row["position"], row["distractors"]) == (length, position, distractors)]
                    cells.append({"method": method, "target_tokens": length, "position": position, "distractors": distractors,
                                  "interpretation": "descriptive ten-prompt cell", **statistics(selected)})
    kv = save_table(store, "tables/kv_summary.csv", summary)
    factorial = save_table(store, "tables/kv_factorial.csv", cells)
    pairs = save_table(store, "tables/kv_paired_intervals.csv", read_json(store.path("metrics/kv/paired_comparisons.json")))
    systems = []
    for row in read_json(store.path("metrics/kv_systems/all_conditions.json")):
        systems.append({key: row[key] for key in ["method", "batch", "context_length", "status"]} | {
            "serialized_bytes": row.get("serialized_bytes"), "calculated_bytes": row.get("calculated_bytes"),
            **{f"{part}_ms": row.get(part, {}).get("median_ms") for part in ["encode", "unpack", "decode_step"]},
            **{f"{part}_peak_allocated_bytes": row.get(part, {}).get("peak_allocated_bytes") for part in ["encode", "unpack", "decode_step"]}})
    system_frame = save_table(store, "tables/kv_packed_systems.csv", systems)
    cases = []
    tokenized = read_gzip_records(store.path("splits/retrieval_tokenized.jsonl.gz"))
    fixed = {tokenized[index]["id"]: tokenized[index] for index in store.configuration["kv"]["qualitative_case_indices"]}
    for row in records:
        if row["id"] in fixed:
            selected = set(row.get("selected_fp16_positions", []))
            cases.append({**row, "messages": fixed[row["id"]]["messages"], "evidence_text": fixed[row["id"]]["evidence_text"],
                          "fp16_selection_mask": [position in selected for position in range(row["prefix_tokens"])],
                          "selection_rule": "fixed supplied case indices [0,30,60,90,120,150]; every method retained"})
    store.save_json("examples/kv_aligned.json", cases)
    return {"attention": a, "training": train, "curves": curves, "recall": r, "distances": d,
            "inference": sweep, "kv": kv, "factorial": factorial, "pairs": pairs, "systems": system_frame}


def create_figures(store, frames):
    """Render eight canonical Times New Roman 10 pt figures, including seed variation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_fonts(store.paths["task"])
    styles, figures = {}, []
    colors = {"dense": "#1f77b4", "window": "#ff7f0e", "linear": "#2ca02c",
              "dilated": "#d62728", "local_content": "#9467bd"}
    def save(figure, name):
        figures.append(save_figure(store, figure, name, styles))
    frame = frames["attention"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    fields = [("output_relative_l2", "Output relative L2 error"), ("gradient_maximum_relative_l2", "Gradient relative L2 error"),
              ("forward_ms", "Forward (ms/batch)"), ("forward_backward_ms", "Forward + backward (ms/batch)"),
              ("forward_backward_peak_allocated_bytes", "Peak allocated (GiB)"), ("forward_backward_peak_reserved_bytes", "Peak reserved (GiB)")]
    for (method, dtype), group in frame[frame["status"] == "complete"].groupby(["method", "dtype"]):
        for axis, (field, label) in zip(axes.flat, fields):
            data = group.sort_values("length")
            values = data[field] / 2 ** 30 if "bytes" in field else data[field]
            values = values.clip(lower=1e-12) if "relative_l2" in field else values
            axis.plot(data["length"], values, marker="o", linestyle="--" if dtype == "float16" else "-", label=f"{method}, {dtype}")
            axis.set(xlabel="Sequence length", ylabel=label)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
    axes[0, 0].legend()
    save(figure, "rq1_exact_attention")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    curves = frames["curves"]
    for index, (count, regime) in enumerate([(4, "mixed"), (8, "mixed"), (16, "mixed"), (4, "near"), (8, "near"), (16, "near")]):
        axis = axes.flat[index]
        subset = curves[(curves["train_K"] == count) & (curves["train_regime"] == regime)]
        for method, group in subset.groupby("method"):
            pivot = group.pivot(index="update", columns="seed", values="loss").rolling(50, min_periods=1).mean()
            mean, deviation = pivot.mean(axis=1), pivot.std(axis=1).fillna(0)
            axis.plot(mean.index, mean, label=method, color=colors[method])
            axis.fill_between(mean.index, mean - deviation, mean + deviation, alpha=.15, color=colors[method])
        axis.set(title=f"Train K={count}, {regime}", xlabel="Optimizer update", ylabel="Query CE (50-update mean)")
        if not subset.empty:
            axis.legend()
    save(figure, "rq2_learning_curves")

    distances = frames["distances"].copy()
    figure, axis = plt.subplots(figsize=(12, 9))
    if not distances.empty:
        distances["model"] = distances.apply(lambda row: f"{row['method']}, train K={row['train_K']} {row['train_regime']}" + (", test W64" if row["test_only_mask"] else ""), axis=1)
        pooled = distances.groupby(["model", "seed", "test_K", "distance_bin"])[["correct", "queries"]].sum().reset_index()
        pooled["accuracy"] = pooled["correct"] / pooled["queries"].replace(0, np.nan)
        average = pooled.groupby(["model", "test_K", "distance_bin"])["accuracy"].mean().reset_index()
        average["column"] = average.apply(lambda row: f"K{row['test_K']}:{row['distance_bin']}", axis=1)
        matrix = average.pivot(index="model", columns="column", values="accuracy")
        ordered_columns = [f"K{count}:{band}" for count in [4, 8, 16] for band in ["0_32", "32_64", "64_128", "128_256"]]
        matrix = matrix.reindex(columns=ordered_columns)
        matrix.columns = [column.replace(":", ":[").replace("_", ",") + ")" for column in matrix.columns]
        graphic = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        figure.colorbar(graphic, ax=axis, label="Query accuracy, mean across training seeds")
    axis.set(title="Distance-by-load recall; row labels identify training load", xlabel="Test load and true source-distance bin")
    save(figure, "rq2_distance_load")

    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    recall = frames["recall"].copy()
    for axis, count in zip(axes, [4, 8, 16]):
        subset = recall[(recall["test_K"] == count) & (recall["train_K"] == count)].copy() if not recall.empty else recall
        if not subset.empty:
            subset["model"] = subset.apply(lambda row: f"{row['method']}/{row['train_regime']}" + ("/test W64" if row["test_only_mask"] else ""), axis=1)
            labels = sorted(subset["model"].unique())
            for offset, regime in zip([-.25, 0, .25], ["near", "far", "mixed"]):
                selected = subset[subset["regime"] == regime].groupby("model")["accuracy"].agg(["mean", "std"]).reindex(labels)
                axis.bar(np.arange(len(labels)) + offset, selected["mean"], width=.25,
                         yerr=selected["std"].fillna(0), capsize=2, label=regime)
            axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
            axis.legend(ncol=3)
        axis.axhline(1 / 64, color="grey", linestyle=":", label="1/64 labels")
        axis.axhline(1 / count, color="black", linestyle="--", linewidth=.8, label="1/K present values")
        if not subset.empty:
            axis.legend(ncol=3)
        axis.set(title=f"Matched training/test load K={count}; 1,024 sequences per regime per seed", ylabel="Query accuracy", ylim=(0, 1.05))
    save(figure, "rq2_regime_controls")

    kv = frames["kv"].set_index("method").reindex(store.configuration["kv"]["methods"]).apply(pd.to_numeric, errors="coerce")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    positions = np.arange(len(kv))
    axes[0, 0].bar(positions - .18, kv["accuracy"], width=.36, label="Unconditional, n=180")
    axes[0, 0].bar(positions + .18, kv["conditional_on_fp16_correct_accuracy"], width=.36, label="Conditional on FP16 correct")
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Retrieval accuracy")
    for axis, field, label in [(axes[0, 1], "reference_answer_mean_nll", "FP16-reference answer NLL (nats/token)"),
                              (axes[1, 0], "post_prefill_logit_kl", "Post-prefill KL from FP16 (nats)"),
                              (axes[1, 1], "evidence_retention_fraction", "Evidence fraction (hybrid: selected FP16)")]:
        axis.bar(positions, kv[field])
        axis.set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks(positions, kv.index, rotation=35, ha="right")
    save(figure, "rq3_kv_fidelity")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (length, distractors) in zip(axes.flat, [(length, distractors) for length in [2048, 4096] for distractors in [0, 2, 8]]):
        selected = frames["factorial"][(frames["factorial"]["target_tokens"] == length) & (frames["factorial"]["distractors"] == distractors)]
        matrix = selected.pivot(index="method", columns="position", values="accuracy").reindex(index=store.configuration["kv"]["methods"], columns=["start", "middle", "end"])
        graphic = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1)
        axis.set_xticks(range(3), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title(f"Target ~{length}, distractors={distractors}; n=10/cell")
        figure.colorbar(graphic, ax=axis, label="Accuracy")
    save(figure, "rq3_kv_factorial")

    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, (length, batch) in zip(axes.flat, [(length, batch) for length in [2048, 4096] for batch in [1, 8]]):
        subset = frames["systems"][(frames["systems"]["context_length"] == length) & (frames["systems"]["batch"] == batch)].set_index("method").reindex(kv.index)
        for offset, part in zip([-.25, 0, .25], ["encode", "unpack", "decode_step"]):
            axis.bar(positions + offset, subset[f"{part}_ms"], width=.25, label=part)
        axis.set_xticks(positions, kv.index, rotation=35, ha="right")
        axis.set(title=f"B={batch}, L={length}; 10 timed repetitions", ylabel="Median ms/batch")
        axis.legend()
    save(figure, "rq4_packed_cache_systems")

    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    sweep = frames["inference"]
    for method, group in sweep[sweep["status"] == "complete"].groupby("method"):
        group = group.sort_values("length")
        axes[0, 0].plot(group["length"], group["median_ms"], marker="o", label=method, color=colors[method])
        axes[0, 1].plot(group["length"], group["additional_peak_allocated_bytes"] / 2 ** 20, marker="o", label=method, color=colors[method])
    for axis, label in [(axes[0, 0], "Forward-only ms/batch"), (axes[0, 1], "Additional peak allocated MiB")]:
        axis.set(xlabel="MQAR operator sequence length", ylabel=label)
        axis.set_xscale("log", base=2)
        axis.legend()
    axes[1, 0].bar(positions - .18, kv["fraction_full_fp16_bytes"], width=.36, label="Serialized prefix/full FP16")
    fp16_peak = kv.loc["fp16", "peak_allocated_bytes"]
    axes[1, 0].bar(positions + .18, kv["peak_allocated_bytes"] / fp16_peak, width=.36, label="Measured peak/full FP16 peak")
    axes[1, 0].set_ylabel("Mean ratio (different quantities)")
    axes[1, 0].legend()
    bottom = np.zeros(len(kv))
    for field, label in [("prefill_seconds", "Shared full prefill"), ("question_importance_seconds", "Question scoring"),
                         ("packed_encode_seconds", "Pack"), ("question_suffix_seconds", "Question suffix"), ("mean_decode_seconds", "Greedy decode")]:
        values = kv[field].fillna(0).to_numpy()
        axes[1, 1].bar(positions, values, bottom=bottom, label=label)
        bottom += values
    axes[1, 1].set_ylabel("Mean seconds/request (quality run)")
    axes[1, 1].legend()
    for axis in axes[1]:
        axis.set_xticks(positions, kv.index, rotation=35, ha="right")
    save(figure, "rq4_resources_and_overheads")
    return figures, styles


def required_stages(configuration):
    """List the stages that must have completed attempts before report evidence exists."""
    return ["data", "part_a", "intervention", "mqar_final", "mqar_inference", "kv_reference",
            "kv_policy_eviction", "kv_policy_quantization", "kv_policy_hybrid", "kv_quality", "kv_systems",
            *[f"train_{condition['id']}" for condition in training_conditions(configuration)],
            *[f"mqar_test_{condition['id']}" for condition in evaluation_conditions(configuration)]]


def build_evidence(store):
    """Register all authentic evidence and qualify any unsuccessful required conditions."""
    missing = [stage for stage in required_stages(store.configuration) if not store.completed(stage)]
    if missing:
        raise RuntimeError(f"Required Task 2 stages are incomplete: {missing}")
    frames = collect_tables(store)
    figures, styles = create_figures(store, frames)
    lock = read_json(store.path("manifests/run_lock.json"))
    style = {"font": configure_fonts(store.paths["task"]), "figures": styles,
             "rendered_at_utc": datetime.now(timezone.utc).isoformat(), "rendered_on": platform.platform(),
             "original_runtime": lock["runtime"], "computed_on": "Google Colab",
             "table_hashes": {path.relative_to(store.root).as_posix(): sha256_file(path) for path in sorted(store.path("tables").glob("*.csv"))}}
    store.save_json("figures/style_manifest.json", style)
    failures = []
    for name in ["attention", "training", "systems", "inference"]:
        frame = frames[name]
        for row in frame[frame["status"] != "complete"].to_dict("records"):
            failures.append({"suite": name, "id": row.get("id", row.get("method")), "status": row["status"],
                             "length": row.get("length", row.get("context_length")), "dtype": row.get("dtype")})
    for path in sorted(store.path("predictions/kv").glob("*/*.json")):
        row = read_json(path)
        if row["status"] != "complete":
            failures.append({"suite": "kv_quality", "id": row["id"], "method": row["method"], "status": row["status"]})
    for condition in evaluation_conditions(store.configuration):
        row = read_json(store.path(f"metrics/mqar_test/{condition['id']}.json"))
        if row["status"] != "complete":
            failures.append({"suite": "mqar_test", "id": condition["id"], "status": row["status"]})
    questions = {
        "rq1": {"tables": ["tables/attention_conditions.csv"], "figures": [figures[0]], "metrics": ["metrics/attention/correctness.json"]},
        "rq2": {"tables": ["tables/mqar_training.csv", "tables/mqar_accuracy.csv", "tables/mqar_seed_dispersion.csv"],
                "figures": figures[1:4], "metrics": ["configs/intervention_locked.json"]},
        "rq3": {"tables": ["tables/kv_summary.csv", "tables/kv_paired_intervals.csv", "tables/kv_factorial.csv"],
                "figures": figures[4:6], "metrics": ["metrics/kv/reference_checks.json"]},
        "rq4": {"tables": ["tables/mqar_operator_resources.csv", "tables/kv_packed_systems.csv", "tables/kv_summary.csv"],
                "figures": figures[6:], "metrics": ["metrics/kv/model_geometry.json"]},
    }
    required = {}
    for path in sorted(store.root.rglob("*")):
        relative = path.relative_to(store.root).as_posix()
        if path.is_file() and not relative.startswith("checkpoints/") and relative not in {"backup_index.json", "manifests/run.json"} and not path.name.endswith((".partial", ".uploading", ".previous", ".restoring")):
            required[relative] = sha256_file(path)
    manifest = {"task": "Task_2", "run_id": lock["run_id"], "config_id": lock["config_id"], "data_id": lock["data_id"],
        "computed_on": "Google Colab", "runtime": lock["runtime"], "status": "complete_with_recorded_failures" if failures else "complete",
        "recorded_failures": failures, "required_artifacts": required, "research_questions": questions,
        "counts": {"part_a_conditions": 36, "mqar_training_runs": 42, "mqar_test_variants": 54, "kv_prompts": 180,
                   "kv_methods": 6, "kv_systems_conditions": 24, "figures": len(figures)},
        "report_analysis": "Student-authored report language, interpretation and analysis required."}
    store.save_json("manifests/run.json", manifest)
    return {"status": manifest["status"], "figures": len(figures), "recorded_failures": len(failures), "results": str(store.root)}
