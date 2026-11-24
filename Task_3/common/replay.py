# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Deep validation and local display of cloud results without model-library imports."""

import math
from collections import Counter

import numpy as np

from Task_3.common.artifacts import read_json, read_jsonl, sha256_file
from Task_3.common.config import bank_names, draft_ids, fingerprint, protocol_id, trained_variants
from Task_3.data.prepare import read_gzip_records, verify_supplied_inputs
from Task_3.data.prefixes import selected_positions
from Task_3.evaluation.quality import score_answer
from Task_3.evaluation.statistics import paired_interval


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)


def validate_generation(row, cap, ignore_eos=False):
    require(row["new_tokens"] == len(row["token_ids"]) and 1 <= row["new_tokens"] <= cap, "Generation length/cap differs.")
    require(row["ignore_eos"] == ignore_eos, "An EOS-aware condition silently ignored EOS.")
    require(row["new_tokens"] == cap if ignore_eos else row["eos"] or row["new_tokens"] == cap, "Generation ended without EOS or the cap.")
    require(all(math.isfinite(row[key]) and row[key] > 0 for key in ["prefill_seconds", "decode_seconds", "elapsed_seconds"]), "Generation time is invalid.")


def validate_rounds(result, prompt_length, gamma):
    """Recalculate every counter and the one-pending-token invariant from emitted tokens."""
    output, accepted, examined, proposed, calls = [], 0, 0, 0, 0
    for index, row in enumerate(result["rounds"]):
        n, a, e = row["proposed"], row["accepted"], row["examined"]
        require(0 <= a <= e <= n <= gamma and e > 0 and n == len(row["proposed_tokens"]), "Invalid proposal/acceptance counters.")
        require(row["output_offset"] == len(output), "Speculative rounds have output gaps or duplicate tokens.")
        require(row["emitted_tokens"][:a] == row["proposed_tokens"][:a], "An accepted prefix changed token IDs.")
        if row["rejected"]:
            require(e == a + 1 and len(row["emitted_tokens"]) == a + 1 and not row["bonus"], "Incorrect residual/bonus path.")
        else:
            require(e == a == n and len(row["emitted_tokens"]) == n + int(row["bonus"]), "Incorrect full-acceptance path.")
        output.extend(row["emitted_tokens"])
        require(row["cache_length"] == prompt_length + len(output) - 1 and row["pending_token"] == output[-1], "Saved cache advancement is inconsistent.")
        require(row["draft_forward_passes"] == n + int(index > 0) and row["target_verification_passes"] == 1
                and row["target_verification_input_tokens"] == n + int(index > 0), "Actual draft/target passes were miscounted.")
        require(all(math.isfinite(row[key]) and row[key] >= 0 for key in ["proposal_seconds", "verification_seconds", "sampling_seconds", "maintenance_seconds"]), "Invalid per-round timing decomposition.")
        accepted += a
        examined += e
        proposed += n
        calls += row["draft_forward_passes"]
    require(output == result["token_ids"] and result["accepted"] == accepted and result["examined"] == examined
            and result["proposed"] == proposed and result["draft_forward_passes"] == calls, "Speculative totals differ from raw rounds.")
    require(close(result["accepted_examined_fraction"], accepted / examined) and close(result["accepted_proposed_fraction"], accepted / proposed), "Acceptance denominators differ.")
    require(result["target_passes"] == len(result["rounds"]) + 1 and close(result["target_passes_per_output_token"], result["target_passes"] / len(output)), "Target passes/output token differ.")


def validate_replay(configuration, paths, deep=True):
    """Require complete matching evidence before any local notebook outputs are displayed."""
    root = paths["results"]
    if not (root / "manifests/run.json").is_file():
        raise FileNotFoundError(f"No completed Task 3 run at {root}. Execute on Colab and download Task_3/results/ with its run folder intact. Local replay never starts experiments.")
    manifest = read_json(root / "manifests/run.json")
    require(manifest["task"] == "Task_3" and manifest["status"] == "complete" and manifest["config_id"] == protocol_id(configuration, paths["task"]), "Saved Task 3 scientific identity/completion differs.")
    print(f"Validating {len(manifest['required_artifacts'])} Task 3 evidence files...", flush=True)
    for relative, expected in manifest["required_artifacts"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and sha256_file(path) == expected, f"Missing/corrupt Task 3 artifact: {relative}")
    lock = read_json(root / "manifests/run_lock.json")
    data = read_json(root / "splits/data_manifest.json")
    require(manifest["config_id"] == lock["config_id"] and fingerprint(data) == manifest["data_id"] == lock["data_id"], "Run/data identities differ.")
    require(data["supplied_hashes"] == verify_supplied_inputs(paths["resources"]), "The supplied source data differ from the compute run.")
    train, test = [read_gzip_records(root / f"splits/{split}_prompts.jsonl.gz") for split in ["train", "test"]]
    require(len(train) == 800 and len(test) == 120 and len({x["id"] for x in train + test}) == 920, "The prescribed train/test prompt counts differ.")
    require(not ({x["text"] for x in train} & {x["text"] for x in test}), "Training and held-out text overlap.")
    require(Counter(x["workload"] for x in test) == {"math": 60, "familiar": 30, "shifted": 30}, "Held-out workload balance differs.")
    toy = read_json(root / "metrics/sampler_markov.json")
    require(len(toy["conditions"]) == 4 and len(toy["joint_categories"]) == 108, "Joint Markov evidence is incomplete.")
    for condition in toy["conditions"]:
        require(condition["trials"] == 100000 and condition["minimum_expected_count"] >= 10 and condition["passed"], "Markov trial/count/gate requirement failed.")
        require(condition["p_value"] < .01 if condition["negative_control"] else condition["p_value"] >= .01, "Correct/incorrect Markov controls have invalid conclusions.")
        rows = [x for x in toy["joint_categories"] if (x["gamma"], x["negative_control"]) == (condition["gamma"], condition["negative_control"])]
        require(len(rows) == 27 and sum(x["observed_count"] for x in rows) == 100000, "Joint sequence counts do not cover all trials.")
        statistic = sum((row["observed_count"] - row["expected_count"]) ** 2 / row["expected_count"] for row in rows)
        require(close(statistic, condition["chi_square"]), "Pearson statistic differs from saved joint counts.")
    require(read_json(root / "metrics/cache_checks.json")["status"] == "passed", "Real cache validation did not pass.")
    sources = {row["id"]: row for row in train}
    initializations, training = {}, {}
    for variant in trained_variants(configuration):
        name = variant["id"]
        row = read_json(root / f"metrics/training/{name}.json")
        training[name] = row
        require(row["status"] == "complete" and row["teacher_scored_tokens"] == {"math": 30000, "short": 10000}
                and row["successful_optimizer_updates"] == 157, "A distillation run violates the fixed successful-update/token budget.")
        previous = initializations.setdefault(variant["seed"], row["initialization_sha256"])
        require(previous == row["initialization_sha256"], "Off/on policy methods did not share seeded adapter initialization.")
        history = read_jsonl(root / row["history"])
        require([x["update"] for x in history] == list(range(1, 158)) and [x["teacher_scored_tokens"] for x in history] == [256] * 156 + [64], "Distillation accumulation/update sequence differs.")
        require(all(close(x["lr"], .0003) and math.isfinite(x["forward_kl"]) for x in history), "Invalid distillation learning rates/losses.")
        counts = Counter()
        for relative in row["annotations"]:
            for annotation in read_gzip_records(root / relative):
                source = sources[annotation["source_id"]]
                require(annotation["prompt_sha256"] == source["prompt_sha256"] and annotation["domain"] == source["domain"], "A training annotation uses a held-out or changed prompt.")
                first, count = annotation["first_scored_response_index"], annotation["teacher_scored_tokens"]
                require(count > 0 and first >= 0 and first + count <= len(annotation["response"]) <= configuration["training"]["response_caps"][source["domain"]], "Training response cap/scored prefix changed.")
                counts[source["domain"]] += count
        require(dict(counts) == row["teacher_scored_tokens"], "Saved presentations do not cover the teacher-scored quota.")
        if variant["method"] == "off_policy":
            for source in train:
                pool = read_json(root / f"rollouts/teacher_pool_seed{variant['seed']}/{source['id']}.json")
                require(pool["source_id"] == source["id"] and pool["prompt_sha256"] == source["prompt_sha256"], "Frozen teacher pool omitted or changed a training prompt.")
        merge = read_json(root / f"metrics/merging/{name}.json")
        require(merge["fresh_fp16_backbone"] and not merge["wrapper_in_primary_timing"] and merge["numerical_checks"], "Primary timing is not a verified merged-backbone comparison.")
        require(all(item["relative_l2"] <= configuration["evaluation"]["merge_relative_l2_tolerance"] for item in merge["numerical_checks"]), "Adapter merge numerical error exceeds tolerance.")
    print("Verified four 40,000-token runs, training annotations and adapter merges.", flush=True)
    for bank in bank_names(configuration):
        for prompt in test:
            frozen = read_json(root / f"banks/{bank}/{prompt['id']}.json")
            response = frozen["response_token_ids"]
            require(frozen["prompt_sha256"] == prompt["prompt_sha256"] and frozen["workload"] == prompt["workload"], "A bank changed a held-out prompt.")
            expected = selected_positions(len(response), prompt["id"], bank, configuration)
            require(frozen["selected_positions"] == expected and frozen["prefix_ids"] == [f"{bank}:{prompt['id']}:p{p:03d}" for p in expected], "Saved prefix positions/IDs changed.")
            producer = frozen["producer"]
            expected_checkpoint = configuration["models"]["teacher_revision"] if producer == "teacher" else configuration["models"]["student_revision"] if producer == "original" else training[producer]["adapter_sha256"]
            require(frozen["producer_checkpoint"] == expected_checkpoint, "Own-policy bank producing checkpoint differs.")
            for draft in draft_ids(configuration):
                rows = read_gzip_records(root / f"alignment/{bank}/{draft}/{prompt['id']}.jsonl.gz")
                require([x["prefix_id"] for x in rows] == frozen["prefix_ids"], "Drafts were compared on unmatched prefixes.")
                for row in rows:
                    require(row["draft"] == draft and row["producer_checkpoint"] == expected_checkpoint, "Alignment model/checkpoint identity differs.")
                    require(all(math.isfinite(row[key]) and row[key] >= 0 for key in ["forward_kl", "reverse_kl", "tv", "expected_acceptance"])
                            and close(row["expected_acceptance"], 1 - row["tv"]), "Full-vocabulary alignment or 1-TV identity is invalid.")
    print("Verified seven frozen prefix banks and all five drafts on identical prefix IDs.", flush=True)
    for suite, names, prompts in [("baseline", ["original"], test), ("primary", draft_ids(configuration), test),
                                  ("fixed64", draft_ids(configuration), [x for x in test if x["domain"] == "math"][:24])]:
        expected_paths = set()
        for name in names:
            for gamma in [2, 4]:
                for repetition in [0, 1]:
                    for prompt in prompts:
                        relative = f"benchmarks/{suite}/{name}/gamma{gamma}/rep{repetition}/{prompt['id']}.json"
                        expected_paths.add(relative)
                        row = read_json(root / relative)
                        require((row["draft"], row["gamma"], row["repetition"], row["prompt_sha256"]) == (name, gamma, repetition, prompt["prompt_sha256"]), "A paired benchmark changed its condition.")
                        require(row["adjacent_target_reference"], "A speculative benchmark lacks its adjacent target control.")
                        validate_generation(row["target"], 64, suite == "fixed64")
                        validate_generation(row["speculative"], 64, suite == "fixed64")
                        validate_rounds(row["speculative"], len(prompt["prompt_token_ids"]), gamma)
                        p, q = row["target"], row["speculative"]
                        require(close(row["end_to_end_tps_ratio"], (q["new_tokens"] / q["elapsed_seconds"]) / (p["new_tokens"] / p["elapsed_seconds"])), "Paired throughput ratio differs from measured tokens/times.")
        actual = {p.relative_to(root).as_posix() for p in (root / f"benchmarks/{suite}").rglob("*.json")}
        require(actual == expected_paths, "A required benchmark condition is absent, duplicated or extraneous.")
        print(f"Verified {suite}: {len(expected_paths)} adjacent-control pairs.", flush=True)
    for name in ["target", *draft_ids(configuration)]:
        for repetition in [0, 1]:
            for prompt in test:
                row = read_json(root / f"quality/{name}/rep{repetition}/{prompt['id']}.json")
                validate_generation(row["response"], configuration["sampling"]["quality_caps"][prompt["domain"]])
                require(row["prompt_sha256"] == prompt["prompt_sha256"] and all(row[key] == value for key, value in score_answer(prompt, row["text"]).items()), "Standalone answer scoring or prompt differs.")
    systems = list((root / "systems").glob("*.json"))
    require(len(systems) == 18, "Isolated merged/unmerged context-length cost matrix is incomplete.")
    for path in systems:
        row = read_json(path)
        for timing in [row["target_single_step"], row["draft_single_step"], *row["verification"].values()]:
            require(len(timing["samples_ms"]) >= 10 and timing["warmup"] >= 1 and close(timing["median_ms"], float(np.median(timing["samples_ms"]))), "Isolated timing lacks warmed-up raw samples.")
    if deep:
        import pandas as pd
        speed = pd.read_csv(root / "tables/speculative_per_prompt.csv")
        intervals = pd.read_csv(root / "tables/paired_prompt_intervals.csv")
        for row in intervals.to_dict("records"):
            selected = speed[(speed.suite == row["suite"]) & (speed.draft == row["draft"]) & (speed.gamma == row["gamma"])]
            if row["workload"] != "all":
                selected = selected[selected.workload == row["workload"]]
            values = selected.groupby("prompt_id")[row["metric"]].mean().to_dict()
            expected = paired_interval(values, resamples=configuration["evaluation"]["bootstrap_resamples"], geometric="ratio" in row["metric"])
            require(all(close(row[key], expected[key]) for key in ["estimate", "ci_low", "ci_high"]), "Prompt-paired interval differs from its saved inputs.")
    style = read_json(root / "figures/style_manifest.json")
    require(len(style["figures"]) == len(list((root / "figures").rglob("*.png"))) == 9, "Exactly nine canonical main PNGs are required.")
    require(all(x["font_families"] == ["Times New Roman"] and x["font_sizes_pt"] == [10.] for x in style["figures"].values()), "A main figure is not Times New Roman 10 pt throughout.")
    return manifest


class ReplaySession:
    """Present the same seven-stage notebook interface using cloud artifacts only."""

    def __init__(self, configuration, paths):
        self.configuration, self.paths = configuration, paths
        self.manifest = validate_replay(configuration, paths)
        print(f"Replay only | original computation: Google Colab | {self.manifest['run_id']}")

    def prepare_data(self):
        return read_json(self.paths["results"] / "splits/data_manifest.json")

    def validate_sampler(self):
        return read_json(self.paths["results"] / "metrics/sampler_markov.json")["conditions"]

    def baseline(self):
        return {"status": "saved pretrained baseline", "paired_records": 480}

    def train_method(self, method):
        return [read_json(self.paths["results"] / f"metrics/training/{x['id']}.json") for x in trained_variants(self.configuration) if x["method"] == method]

    def alignment(self):
        return {"status": "saved paired-prefix evidence", "banks": 7, "drafts": 5}

    def merged_execution(self):
        return {"status": "saved merged-draft benchmarks", "paired_records": 2400}

    def workload_evaluation(self):
        return {"status": "saved workload/quality evidence", "fixed64_pairs": 480, "standalone_quality_records": 1440}

    def build_evidence(self):
        return {"status": "saved evidence", "figures": 9, "font": "Times New Roman, 10 pt"}

    def finish(self):
        manifest = validate_replay(self.configuration, self.paths)
        return {"status": manifest["status"], "computed_on": "Google Colab", "run_id": manifest["run_id"]}


def display_question(session, question, kind):
    """Display only registered saved tables and canonical main images."""
    import pandas as pd
    from IPython.display import Image, display
    root = session.paths["results"]
    manifest = read_json(root / "manifests/run.json")
    for relative in manifest["research_questions"][question].get(kind, []):
        print(relative)
        if kind == "figures":
            display(Image(filename=str(root / relative)))
        elif kind == "tables":
            frame = pd.read_csv(root / relative)
            if len(frame) > 36:
                print(f"Showing 24/{len(frame)} saved rows; full data are in the CSV.")
                frame = frame.head(24)
            display(frame)
        else:
            display(read_json(root / relative))
