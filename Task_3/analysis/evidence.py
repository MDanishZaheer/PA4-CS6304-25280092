# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Derive numerical evidence and nine canonical figures from completed saved experiments."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from Task_3.analysis.fonts import configure_fonts, save_figure
from Task_3.analysis.costs import account_upfront_cost
from Task_3.common.artifacts import read_json, read_jsonl, sha256_file
from Task_3.common.config import bank_names, draft_ids, trained_variants
from Task_3.data.prepare import read_gzip_records
from Task_3.evaluation.statistics import break_even_acceptance, deployment_break_even, paired_interval, predicted_speedup


FIGURE_NAMES = ["sampler_correctness", "distillation_curves", "common_prefix_alignment", "workload_position",
                "quality_and_lengths", "acceptance_verification", "predicted_measured_speedup", "execution_costs", "training_amortization"]


def required_stages(configuration):
    """Require every mandatory experiment, including all four seeds/methods and own banks."""
    return ["data", "sampler", "benchmark_baseline", "teacher_pool_seed6304", "teacher_pool_seed6305", "banks", "alignment",
        "benchmark_primary", "benchmark_fixed64", "quality", "isolated_unmerged_off_policy_seed6304",
        *[f"train_{x['id']}" for x in trained_variants(configuration)],
        *[f"merge_{x['id']}" for x in trained_variants(configuration)], *[f"isolated_{name}" for name in draft_ids(configuration)]]


def save_table(store, name, records):
    """Use the same machine-readable values for both tables and plots."""
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    relative = f"tables/{name}.csv"
    store.path(relative).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(store.path(relative), index=False, encoding="utf-8", lineterminator="\n")
    store.backup([relative])
    return frame


def collect_tables(store):
    """Recompute paired summaries, actual token denominators and amortization from raw records."""
    configuration = store.configuration
    toy = read_json(store.path("metrics/sampler_markov.json"))
    frames = {"markov": save_table(store, "sampler_conditions", toy["conditions"]),
              "joint": save_table(store, "sampler_joint_probabilities", toy["joint_categories"])}
    training, histories, costs = [], [], []
    ledger = read_jsonl(store.path("metrics/stage_wall_costs.jsonl"))
    for variant in trained_variants(configuration):
        name = variant["id"]
        row = read_json(store.path(f"metrics/training/{name}.json"))
        merge = read_json(store.path(f"metrics/merging/{name}.json"))
        costs.append({"draft": name, "method": variant["method"], "seed": variant["seed"],
            **{key: row[key] for key in ["teacher_pool_generation_seconds", "student_rollout_seconds", "teacher_scoring_seconds", "student_backward_and_optimizer_seconds"]},
            "merge_compute_seconds": merge["merge_compute_seconds"], "merge_write_and_backup_seconds": merge["merge_write_and_backup_seconds"],
            **account_upfront_cost(variant, row, merge, ledger)})
        training.append({**variant, **{key: row[key] for key in ["trainable_parameters", "total_parameters", "successful_optimizer_updates", "overflow_attempts", "adapter_sha256"]},
            "math_scored_tokens": row["teacher_scored_tokens"]["math"], "short_scored_tokens": row["teacher_scored_tokens"]["short"],
            "reported_training_peak_allocated_bytes": row.get("peak_allocated_bytes"),
            "reported_training_peak_reserved_bytes": row.get("peak_reserved_bytes"),
            "training_memory_scope": "entire uninterrupted training session" if variant["method"] == "off_policy" else
                "observed peak since last rollout reset; not certified as the whole-run maximum",
            "merge_maximum_relative_l2": max(item["relative_l2"] for item in merge["numerical_checks"])})
        histories.extend({**variant, **item} for item in read_jsonl(store.path(row["history"])))
    frames["training"] = save_table(store, "distillation_summary", training)
    frames["curves"] = save_table(store, "distillation_learning_curves", histories)
    frames["costs"] = save_table(store, "upfront_costs", costs)
    alignment = [row for path in sorted(store.path("alignment").rglob("*.jsonl.gz")) for row in read_gzip_records(path)]
    frames["alignment"] = save_table(store, "common_prefix_metrics", alignment)
    test_prompts = read_gzip_records(store.path("splits/test_prompts.jsonl.gz"))
    prompt_lookup = {row["id"]: row for row in test_prompts}
    fixed_ids = {test_prompts[index]["id"] for index in configuration["evaluation"]["fixed_qualitative_prompt_indices"]}
    aligned_examples = []
    for bank in bank_names(configuration):
        for pid in sorted(fixed_ids):
            frozen = read_json(store.path(f"banks/{bank}/{pid}.json"))
            selected_ids = set(frozen["prefix_ids"][:3])
            for row in alignment:
                if row["prefix_id"] in selected_ids:
                    aligned_examples.append({**row, "prompt_text": prompt_lookup[pid]["text"],
                        "response_prefix_token_ids": frozen["response_token_ids"][:row["position"] - 1],
                        "selection": "fixed prompt indices [0,30,60,90], first three saved positions, every draft retained"})
    store.save_json("examples/aligned_prefixes.json", aligned_examples)
    metrics = ["forward_kl", "reverse_kl", "tv", "expected_acceptance", "greedy_agreement"]
    grouped = frames["alignment"].groupby(["draft", "bank", "workload", "position_bin"], sort=False)
    aggregate = grouped[metrics].mean().reset_index()
    aggregate["prefix_positions"] = grouped.size().to_numpy()
    frames["alignment_summary"] = save_table(store, "common_prefix_summary", aggregate)
    consistency = []
    original = frames["alignment"][frames["alignment"]["draft"] == "original"]
    for (workload, band), group in original.groupby(["workload", "position_bin"]):
        for metric in ["forward_kl", "tv", "expected_acceptance"]:
            left = group[group.bank == "exact_spec_original_gamma4"].groupby("prompt_id")[metric].mean().to_dict()
            right = group[group.bank == "teacher"].groupby("prompt_id")[metric].mean().to_dict()
            paired = set(left) & set(right)
            if paired:
                consistency.append({"workload": workload, "position_bin": band, "metric": metric,
                    **paired_interval({key: left[key] for key in paired}, {key: right[key] for key in paired}, configuration["evaluation"]["bootstrap_resamples"]),
                    "comparison": "speculative minus teacher bank; independent target-law realizations paired by prompt, not identical token histories"})
    frames["consistency"] = save_table(store, "target_speculative_consistency", consistency)
    systems, system_lookup = [], {}
    for path in sorted(store.path("systems").glob("*.json")):
        row = read_json(path)
        systems.append({"draft": row["draft"], "context_length": row["context_length"], "unmerged_wrapper": row["unmerged_wrapper"],
            "target_step_ms": row["target_single_step"]["median_ms"], "draft_step_ms": row["draft_single_step"]["median_ms"], "c": row["c"],
            **{f"v_gamma{gamma}": row["verification"][str(gamma)]["v_gamma"] for gamma in [2, 4]}})
        if not row["unmerged_wrapper"]:
            system_lookup[(row["draft"], row["context_length"])] = row
    frames["systems"] = save_table(store, "isolated_decode_costs", systems)
    expected = frames["alignment"][(frames["alignment"].bank == "teacher") & (frames["alignment"].position <= 64)]
    expected_map = expected.groupby(["draft", "prompt_id"])[["expected_acceptance", "forward_kl", "tv"]].mean().to_dict("index")
    speed, examples = [], []
    prompts = read_gzip_records(store.path("splits/test_prompts.jsonl.gz"))
    fixed = {prompts[index]["id"] for index in configuration["evaluation"]["fixed_qualitative_prompt_indices"]}
    for suite in ["baseline", "primary", "fixed64"]:
        for path in sorted(store.path(f"benchmarks/{suite}").rglob("*.json")):
            record = read_json(path)
            speculative, target = record["speculative"], record["target"]
            row = {key: record[key] for key in ["prompt_id", "draft", "gamma", "repetition", "workload", "suite", "end_to_end_tps_ratio", "steady_decode_tps_ratio", "request_latency_ratio"]}
            row.update({key: speculative[key] for key in ["accepted_examined_fraction", "accepted_proposed_fraction", "mean_accepted_prefix", "target_passes_per_output_token",
                "draft_forward_passes", "prefill_seconds", "proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds", "host_overhead_seconds", "peak_allocated_bytes"]})
            row.update(output_tokens=speculative["new_tokens"], target_output_tokens=target["new_tokens"], eos=speculative["eos"], truncated=speculative["truncated"],
                spec_elapsed_seconds=speculative["elapsed_seconds"], target_elapsed_seconds=target["elapsed_seconds"],
                seconds_saved_per_request=target["elapsed_seconds"] - speculative["elapsed_seconds"],
                actual_output_length_bin="1_16" if speculative["new_tokens"] <= 16 else "17_32" if speculative["new_tokens"] <= 32 else "33_64")
            lengths = [length for name, length in system_lookup if name == row["draft"]]
            nearest = min(lengths, key=lambda length: abs(length - record["prompt_tokens"]))
            system = system_lookup[(row["draft"], nearest)]
            alpha = expected_map[(row["draft"], row["prompt_id"])]
            row.update(alpha, isolated_context_length=nearest, c=system["c"], v_gamma=system["verification"][str(row["gamma"])]["v_gamma"])
            row["projected_speedup"] = predicted_speedup(row["expected_acceptance"], row["gamma"], row["c"], row["v_gamma"])
            row["projected_using_empirical_acceptance"] = predicted_speedup(row["accepted_examined_fraction"], row["gamma"], row["c"], row["v_gamma"])
            row["break_even_acceptance"] = break_even_acceptance(row["gamma"], row["c"], row["v_gamma"])
            speed.append(row)
            if suite == "primary" and record["prompt_id"] in fixed and record["repetition"] == 0:
                examples.append(record)
    frames["speed"] = save_table(store, "speculative_per_prompt", speed)
    store.save_json("examples/matched_continuations.json", examples)
    summaries, intervals, correlations, deployment = [], [], [], []
    measures = ["accepted_examined_fraction", "accepted_proposed_fraction", "mean_accepted_prefix", "target_passes_per_output_token",
        "draft_forward_passes", "output_tokens", "eos", "truncated", "expected_acceptance", "forward_kl", "tv", "projected_speedup", "projected_using_empirical_acceptance",
        "c", "v_gamma", "break_even_acceptance", "prefill_seconds", "proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds", "host_overhead_seconds", "peak_allocated_bytes"]
    for (suite, name, gamma), all_group in frames["speed"].groupby(["suite", "draft", "gamma"]):
        for workload in ["all", *sorted(all_group.workload.unique())]:
            group = all_group if workload == "all" else all_group[all_group.workload == workload]
            metadata = {"suite": suite, "draft": name, "gamma": int(gamma), "workload": workload}
            averages = group.groupby("prompt_id").mean(numeric_only=True)
            row = {**metadata, "paired_prompts": len(averages), **{key: float(group[key].mean()) for key in measures},
                "end_to_end_tps_ratio": float(np.exp(np.log(averages.end_to_end_tps_ratio).mean())),
                "steady_decode_tps_ratio": float(np.exp(np.log(averages.steady_decode_tps_ratio).mean())),
                "request_latency_ratio": float(np.exp(np.log(averages.request_latency_ratio).mean()))}
            summaries.append(row)
            for field in ["end_to_end_tps_ratio", "steady_decode_tps_ratio", "request_latency_ratio", "accepted_examined_fraction", "accepted_proposed_fraction"]:
                intervals.append({**metadata, "metric": field, **paired_interval(averages[field].to_dict(), resamples=configuration["evaluation"]["bootstrap_resamples"], geometric="ratio" in field)})
            if suite == "primary":
                for metric in ["forward_kl", "tv", "expected_acceptance"]:
                    correlation = float(averages[metric].corr(averages.accepted_examined_fraction)) if averages[metric].std() > 0 and averages.accepted_examined_fraction.std() > 0 else None
                    correlations.append({**metadata, "metric": metric, "pearson_r": correlation, "prompts": len(averages),
                        "scope": "common teacher-bank expectations versus separate speculative trajectories, prompt means"})
                if name != "original":
                    cost = next(item for item in costs if item["draft"] == name)
                    upfront = cost["accounted_upfront_seconds"]
                    savings = float(averages.seconds_saved_per_request.mean())
                    deployment.append({**metadata, "upfront_seconds": upfront, "seconds_saved_per_request": savings,
                        "upfront_is_lower_bound": cost["upfront_is_lower_bound"],
                        "break_even_requests": deployment_break_even(upfront, savings),
                        "finite_break_even": savings > 0, "comparison": "same-condition adjacent target-only request latency"})
    frames["speed_summary"] = save_table(store, "speculative_summary", summaries)
    frames["intervals"] = save_table(store, "paired_prompt_intervals", intervals)
    frames["correlations"] = save_table(store, "acceptance_prediction_correlations", correlations)
    frames["deployment"] = save_table(store, "deployment_break_even", deployment)
    # Explicit paired trained-vs-original and off-vs-on acceptance/throughput comparisons.
    comparisons = []
    primary = frames["speed"][frames["speed"].suite == "primary"]
    pairs = [(name, "original") for name in draft_ids(configuration) if name != "original"] + [(f"on_policy_seed{seed}", f"off_policy_seed{seed}") for seed in [6304, 6305]]
    for gamma in [2, 4]:
        for first, second in pairs:
            for field in ["accepted_examined_fraction", "end_to_end_tps_ratio"]:
                a = primary[(primary.draft == first) & (primary.gamma == gamma)].groupby("prompt_id")[field].mean().to_dict()
                b = primary[(primary.draft == second) & (primary.gamma == gamma)].groupby("prompt_id")[field].mean().to_dict()
                comparisons.append({"draft": first, "reference": second, "gamma": gamma, "metric": field,
                    **paired_interval(a, b, configuration["evaluation"]["bootstrap_resamples"])})
    frames["comparisons"] = save_table(store, "paired_draft_comparisons", comparisons)
    length_groups = primary.groupby(["draft", "gamma", "workload", "actual_output_length_bin"])
    frames["lengths"] = save_table(store, "output_length_diagnostics", length_groups[["accepted_examined_fraction", "end_to_end_tps_ratio", "target_passes_per_output_token"]].mean().reset_index())
    quality = [read_json(path) for path in sorted(store.path("quality").rglob("*.json"))]
    frame = pd.DataFrame({**{key: row[key] for key in ["draft", "prompt_id", "workload", "repetition", "correct", "extractable", "format_compliant"]},
                          "truncated": row["response"]["truncated"], "output_tokens": row["response"]["new_tokens"]} for row in quality)
    frames["quality"] = save_table(store, "standalone_quality_per_prompt", frame)
    grouped = frame.groupby(["draft", "workload"])
    summary = grouped[["correct", "extractable", "format_compliant", "truncated", "output_tokens"]].mean().reset_index()
    summary["responses"] = grouped.size().to_numpy()
    frames["quality_summary"] = save_table(store, "standalone_quality_summary", summary)
    return frames


def create_figures(store, frames):
    """Render nine main PNGs with every text object verified Times New Roman 10 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_fonts(store.paths["task"])
    names = draft_ids(store.configuration)
    labels = {"original": "Original", "target": "Target", **{name: name.replace("off_policy_seed", "Off ").replace("on_policy_seed", "On ") for name in names if name != "original"}}
    colors = dict(zip(names, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    styles, figures = {}, []
    def save(figure, name):
        figures.append(save_figure(store, figure, name, styles))
    def bars(axis, frame, field, ylabel, title=None):
        frame = frame.set_index("draft").reindex(names)
        axis.bar(np.arange(len(names)), frame[field], color=[colors[name] for name in names])
        axis.set_xticks(range(len(names)), [labels[name] for name in names], rotation=30, ha="right")
        axis.set(ylabel=ylabel, title=title or "")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    joint = frames["joint"]
    correct = joint[~joint.negative_control]
    categories = list(correct[correct.gamma == 2].sequence.astype(str).str.zfill(3))
    axes[0].plot(range(27), correct[correct.gamma == 2].target_probability, label="Exact target joint law", color="black")
    for gamma, marker in [(2, "o"), (4, "x")]:
        axes[0].scatter(range(27), correct[correct.gamma == gamma].observed_probability, marker=marker, label=f"Sampler gamma={gamma}", s=14)
    axes[0].set_xticks(range(27), categories, rotation=90)
    axes[0].set(xlabel="Three-symbol output sequence", ylabel="Joint sequence probability", title="100,000 trials per sampler condition")
    axes[0].legend()
    toy = frames["markov"]
    axes[1].bar(range(len(toy)), toy.chi_square, color=["#d62728" if x else "#1f77b4" for x in toy.negative_control])
    axes[1].set_xticks(range(len(toy)), [f"gamma={row.gamma}, {'wrong residual' if row.negative_control else 'correct'}" for row in toy.itertuples()], rotation=25, ha="right")
    axes[1].set(yscale="log", ylabel="Pearson chi-square statistic", title="Correct samplers pass; wrong residual controls reject")
    save(figure, "sampler_correctness")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, group in frames["curves"].groupby("id"):
        axes[0].plot(group.cumulative_tokens, group.forward_kl, label=labels[name], color=colors[name])
    axes[0].set(xlabel="Cumulative teacher-scored response tokens", ylabel="Forward KL (nats/token)", title="Every seed and method; full-vocabulary soft targets")
    axes[0].legend()
    training = frames["training"].set_index("id")
    positions = np.arange(len(training))
    axes[1].bar(positions, training.math_scored_tokens, label="Mathematics")
    axes[1].bar(positions, training.short_scored_tokens, bottom=training.math_scored_tokens, label="Short form")
    axes[1].set_xticks(positions, [labels[name] for name in training.index], rotation=30, ha="right")
    axes[1].set(ylabel="Teacher-scored tokens", title="30,000 mathematics + 10,000 short form per run")
    axes[1].legend()
    save(figure, "distillation_curves")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    alignment = frames["alignment"]
    short = alignment[alignment.position <= 64]
    banks = bank_names(store.configuration)
    bank_labels = ["Teacher", "Original", "Exact spec", "Own off6304", "Own on6304", "Own off6305", "Own on6305"]
    for axis, field, title in [(axes[0, 0], "forward_kl", "Common prefixes: forward KL"), (axes[0, 1], "tv", "Common prefixes: total variation")]:
        matrix = short.groupby(["draft", "bank"])[field].mean().unstack().reindex(index=names, columns=banks)
        graphic = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, cmap="viridis")
        axis.set_xticks(range(len(banks)), bank_labels, rotation=40, ha="right")
        axis.set_yticks(range(len(names)), [labels[name] for name in names])
        axis.set_title(title)
        figure.colorbar(graphic, ax=axis, label="Nats/token" if field == "forward_kl" else "TV")
    consistency = frames["consistency"]
    subset = consistency[consistency.metric == "tv"].reset_index(drop=True)
    axes[1, 0].errorbar(subset.estimate, range(len(subset)), xerr=[subset.estimate - subset.ci_low, subset.ci_high - subset.estimate], fmt="o", capsize=2)
    axes[1, 0].set_yticks(range(len(subset)), subset.workload + " " + subset.position_bin)
    axes[1, 0].axvline(0, color="black", linestyle="--")
    axes[1, 0].set(xlabel="Speculative minus teacher bank mean TV", title="Original draft; prompt-paired 95% intervals")
    matrix = short.groupby(["draft", "bank"]).greedy_agreement.mean().unstack().reindex(index=names, columns=banks)
    graphic = axes[1, 1].imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1, 1].set_xticks(range(len(banks)), bank_labels, rotation=40, ha="right")
    axes[1, 1].set_yticks(range(len(names)), [labels[name] for name in names])
    axes[1, 1].set_title("Greedy agreement on the same saved prefix IDs")
    figure.colorbar(graphic, ax=axes[1, 1], label="Agreement fraction")
    save(figure, "common_prefix_alignment")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    order = ["1_16", "17_32", "33_64", "65_192"]
    teacher_bank = alignment[alignment.bank == "teacher"]
    for column, workload in enumerate(["math", "familiar", "shifted"]):
        for name in names:
            group = teacher_bank[(teacher_bank.draft == name) & (teacher_bank.workload == workload)].groupby("position_bin")[["forward_kl", "expected_acceptance"]].mean().reindex(order)
            for axis, field in [(axes[0, column], "forward_kl"), (axes[1, column], "expected_acceptance")]:
                axis.plot(range(4), group[field], marker="o", label=labels[name], color=colors[name])
                axis.set_xticks(range(4), ["1-16", "17-32", "33-64", "65-192"], rotation=25)
        axes[0, column].set(title=workload, ylabel="Forward KL (nats/token)")
        axes[1, column].set(xlabel="Response position", ylabel="Expected one-token acceptance", ylim=(0, 1.02))
    axes[0, 0].legend()
    save(figure, "workload_position")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    quality_names = ["target", *names]
    for axis, workload in zip(axes.flat[:3], ["math", "familiar", "shifted"]):
        data = frames["quality_summary"][frames["quality_summary"].workload == workload].set_index("draft").reindex(quality_names)
        x = np.arange(len(data))
        axis.bar(x - .18, data.correct, width=.36, label="Standalone correctness")
        axis.bar(x + .18, data.truncated if workload == "math" else data.format_compliant, width=.36,
                 label="Length capped" if workload == "math" else "Format compliant")
        axis.set_xticks(x, [labels[name] for name in data.index], rotation=30, ha="right")
        axis.set(title=f"Standalone {workload}; cap {'512' if workload == 'math' else '32'}", ylabel="Response fraction", ylim=(0, 1.05))
        axis.legend()
    primary = frames["speed_summary"][(frames["speed_summary"].suite == "primary") & (frames["speed_summary"].gamma == 4) & (frames["speed_summary"].workload == "all")]
    bars(axes[1, 1], primary, "eos", "EOS-terminated fraction", "Primary speed cap=64, gamma=4")
    save(figure, "quality_and_lengths")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    summaries = frames["speed_summary"]
    primary = summaries[(summaries.suite == "primary") & (summaries.workload == "all")]
    for axis, field, label in zip(axes.flat,
        ["accepted_examined_fraction", "accepted_proposed_fraction", "mean_accepted_prefix", "target_passes_per_output_token"],
        ["Accepted / examined proposals", "Accepted / all proposed", "Mean accepted prefix per verification", "Target passes / output token (including prefill)"]):
        for offset, gamma in [(-.18, 2), (.18, 4)]:
            subset = primary[primary.gamma == gamma].set_index("draft").reindex(names)
            limits = frames["intervals"][(frames["intervals"].suite == "primary") & (frames["intervals"].workload == "all") &
                (frames["intervals"].gamma == gamma) & (frames["intervals"].metric == field)].set_index("draft").reindex(names)
            uncertainty = [np.maximum(0, subset[field] - limits.ci_low), np.maximum(0, limits.ci_high - subset[field])] if not limits.empty and "ci_low" in limits and limits.ci_low.notna().any() else None
            axis.bar(np.arange(len(names)) + offset, subset[field], width=.36, label=f"gamma={gamma}", yerr=uncertainty, capsize=2)
        axis.set_xticks(range(len(names)), [labels[name] for name in names], rotation=30, ha="right")
        axis.set_ylabel(label)
        axis.legend()
    save(figure, "acceptance_verification")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, gamma in zip(axes, [2, 4]):
        subset = primary[primary.gamma == gamma].set_index("draft").reindex(names)
        for name in names:
            axis.scatter(subset.loc[name, "projected_speedup"], subset.loc[name, "steady_decode_tps_ratio"], label=labels[name], color=colors[name])
        maximum = max(1.1, subset.projected_speedup.max(), subset.steady_decode_tps_ratio.max()) * 1.1
        axis.plot([0, maximum], [0, maximum], color="black", linestyle="--")
        axis.axhline(1, color="grey", linestyle=":")
        axis.set(title=f"gamma={gamma}", xlabel="Projected token speedup using common-bank overlap", ylabel="Measured steady decode TPS ratio", xlim=(0, maximum), ylim=(0, maximum))
        axis.legend()
    save(figure, "predicted_measured_speedup")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, gamma in zip(axes[0], [2, 4]):
        subset = primary[primary.gamma == gamma].set_index("draft").reindex(names)
        bottom = np.zeros(len(names))
        for field, label in [("prefill_seconds", "Prefill"), ("proposal_seconds", "Draft passes"), ("verification_seconds", "Target verification"),
                             ("sampling_seconds", "Accept/residual"), ("maintenance_seconds", "Cache crop/advance"), ("host_overhead_seconds", "Other host")]:
            values = subset[field].to_numpy()
            axis.bar(range(len(names)), values, bottom=bottom, label=label)
            bottom += values
        axis.set_xticks(range(len(names)), [labels[name] for name in names], rotation=30, ha="right")
        axis.set(title=f"gamma={gamma}", ylabel="Mean seconds/request")
        axis.legend(ncol=2)
    systems = frames["systems"]
    for name in names:
        group = systems[(systems.draft == name) & ~systems.unmerged_wrapper].sort_values("context_length")
        axes[1, 0].plot(group.context_length, group.c, marker="o", label=labels[name], color=colors[name])
    axes[1, 0].set(xlabel="Isolated context length", ylabel="c = draft step / target step")
    axes[1, 0].legend()
    wrapper_name = "off_policy_seed6304"
    for wrapper, label in [(False, "Merged FP16"), (True, "Unmerged adapter")]:
        group = systems[(systems.draft == wrapper_name) & (systems.unmerged_wrapper == wrapper)].sort_values("context_length")
        axes[1, 1].plot(group.context_length, group.draft_step_ms, marker="o", label=label)
    axes[1, 1].set(xlabel="Isolated context length", ylabel="Draft single-step ms", title="Fixed off-policy seed 6304 wrapper control")
    axes[1, 1].legend()
    save(figure, "execution_costs")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    costs = frames["costs"].set_index("draft")
    learned = [name for name in names if name != "original"]
    bottom = np.zeros(len(learned))
    for field, label in [("teacher_pool_generation_seconds", "Teacher rollout pool"), ("student_rollout_seconds", "Student rollouts"),
                         ("teacher_scoring_seconds", "Teacher scoring"), ("student_backward_and_optimizer_seconds", "Student updates"),
                         ("merge_compute_seconds", "Merge compute"), ("merge_write_and_backup_seconds", "Merge write/backup")]:
        values = costs.reindex(learned)[field].to_numpy()
        axes[0, 0].bar(range(len(learned)), values, bottom=bottom, label=label)
        bottom += values
    axes[0, 0].set_xticks(range(len(learned)), [labels[name] for name in learned], rotation=30, ha="right")
    axes[0, 0].set(ylabel="Measured component seconds", title="Upfront work, all four independent runs")
    axes[0, 0].legend(ncol=2)
    axes[0, 1].bar(range(len(learned)), costs.reindex(learned).accounted_upfront_seconds)
    axes[0, 1].set_xticks(range(len(learned)), [labels[name] + ('*' if costs.loc[name, 'upfront_is_lower_bound'] else '') for name in learned], rotation=30, ha="right")
    axes[0, 1].set(ylabel="Accounted upfront seconds", title="Stage wall costs; * marks an interrupted-run lower bound")
    deployment = frames["deployment"][frames["deployment"].workload == "all"]
    for offset, gamma in [(-.18, 2), (.18, 4)]:
        group = deployment[deployment.gamma == gamma].set_index("draft").reindex(learned)
        axes[1, 0].bar(np.arange(len(learned)) + offset, group.seconds_saved_per_request, width=.36, label=f"gamma={gamma}")
        values = pd.to_numeric(group.break_even_requests, errors="coerce")
        axes[1, 1].scatter(np.arange(len(learned)) + offset, values, label=f"gamma={gamma}")
        for index, value in enumerate(values):
            if not np.isfinite(value):
                axes[1, 1].text(index + offset, .05 + .1 * (gamma == 4), "None", transform=axes[1, 1].get_xaxis_transform(), rotation=90, ha="center")
    for axis in axes[1]:
        axis.set_xticks(range(len(learned)), [labels[name] for name in learned], rotation=30, ha="right")
        axis.legend()
    axes[1, 0].axhline(0, color="black", linestyle="--")
    axes[1, 0].set(ylabel="Target minus speculative seconds/request", title="Negative savings have no deployment break-even")
    axes[1, 1].set(ylabel="Requests to amortize accounted upfront time", title="No finite value when per-request savings are nonpositive")
    if not pd.to_numeric(deployment.break_even_requests, errors="coerce").notna().any():
        axes[1, 1].clear()
        axes[1, 1].set_axis_off()
        axes[1, 1].text(.5, .56, "No finite deployment break-even", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].text(.5, .44, "All drafts have nonpositive request savings at gamma=2 and 4.", ha="center", va="center", transform=axes[1, 1].transAxes)
    save(figure, "training_amortization")
    return figures, styles


def build_evidence(store, rendered_on="Google Colab"):
    """Finalize only a complete set of required attempts and immutable replay evidence."""
    missing = [stage for stage in required_stages(store.configuration) if not store.completed(stage)]
    if missing:
        raise RuntimeError(f"Task 3 stages remain incomplete: {missing}")
    frames = collect_tables(store)
    figures, styles = create_figures(store, frames)
    store.save_json("figures/style_manifest.json", {"font": configure_fonts(store.paths["task"]), "figures": styles,
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(), "computed_on": "Google Colab",
        "rendered_on": rendered_on,
        "table_hashes": {path.relative_to(store.root).as_posix(): sha256_file(path) for path in store.path("tables").glob("*.csv")}})
    questions = {
        "rq1": {"tables": ["tables/sampler_conditions.csv", "tables/distillation_summary.csv", "tables/common_prefix_summary.csv", "tables/target_speculative_consistency.csv"],
                "figures": figures[:3], "metrics": ["metrics/sampler_markov.json", "metrics/cache_checks.json"]},
        "rq2": {"tables": ["tables/standalone_quality_summary.csv", "tables/output_length_diagnostics.csv"], "figures": figures[3:5]},
        "rq3": {"tables": ["tables/speculative_summary.csv", "tables/paired_prompt_intervals.csv", "tables/paired_draft_comparisons.csv", "tables/acceptance_prediction_correlations.csv"], "figures": figures[5:7]},
        "rq4": {"tables": ["tables/isolated_decode_costs.csv", "tables/upfront_costs.csv", "tables/deployment_break_even.csv"], "figures": figures[7:]}}
    required = {path.relative_to(store.root).as_posix(): sha256_file(path) for path in sorted(store.root.rglob("*"))
        if path.is_file() and not path.relative_to(store.root).as_posix().startswith("checkpoints/")
        and not (path.parent == store.path("metrics") and path.name.startswith("local_"))
        and path.name not in {"backup_index.json", "run.json", ".local_replay_connection.json"}
        and not path.name.endswith((".partial", ".previous", ".uploading", ".restoring"))}
    lock = read_json(store.path("manifests/run_lock.json"))
    manifest = {"task": "Task_3", "run_id": lock["run_id"], "config_id": lock["config_id"], "data_id": lock["data_id"],
        "runtime": lock["runtime"], "computed_on": "Google Colab", "status": "complete", "required_artifacts": required,
        "research_questions": questions, "counts": {"training_runs": 4, "scored_tokens_per_run": 40000, "banks": 7,
            "drafts": 5, "test_prompts": 120, "baseline_pairs": 480, "primary_pairs": 2400, "fixed64_pairs": 480,
            "quality_records": 1440, "figures": 9}, "report_analysis": "Student-authored PDF report interpretation and language required."}
    store.save_json("manifests/run.json", manifest)
    return {"status": "complete", "figures": 9, "registered_artifacts": len(required)}
