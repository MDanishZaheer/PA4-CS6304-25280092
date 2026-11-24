# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate genuine Task 2 results without importing Torch or loading checkpoints."""

import math
from collections import Counter

import numpy as np

from Task_2.common.artifacts import read_json, read_jsonl, sha256_file
from Task_2.common.config import fingerprint, protocol_id
from Task_2.data.mqar import load_evaluation_stream, read_gzip_records, verify_supplied_inputs
from Task_2.data.retrieval import extract_code, load_retrieval_cases
from Task_2.evaluation.recall import aggregate_recall
from Task_2.evaluation.statistics import KV_PAIRS, paired_accuracy_interval
from Task_2.training.protocol import evaluation_conditions, training_conditions


def require(condition, message):
    """Reject an invalid saved invariant even when Python optimization is enabled."""
    if not condition:
        raise ValueError(message)


def close(first, second):
    """Compare recalculated metrics without accepting NaN or infinity."""
    return math.isfinite(float(first)) and math.isfinite(float(second)) and math.isclose(float(first), float(second), rel_tol=1e-8, abs_tol=1e-8)


def validate_replay(configuration, paths, deep=True):
    """Check all required artifacts, counts, paired labels, budgets and canonical fonts."""
    root = paths["results"]
    path = root / "manifests/run.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed Task 2 evidence at {path}. Run main_task_2.ipynb on Colab, download Task_2/results/, and copy it here with the run folder preserved. Replay never starts computation.")
    manifest = read_json(path)
    require(manifest.get("task") == "Task_2" and manifest.get("config_id") == protocol_id(configuration, paths["task"]),
            "Task 2 saved scientific configuration/source identity differs.")
    require(manifest.get("status") in {"complete", "complete_with_recorded_failures"}, "Task 2 run is not finalized.")
    print(f"Validating Task 2 saved evidence: {len(manifest['required_artifacts'])} registered files...", flush=True)
    for relative, expected in manifest["required_artifacts"].items():
        artifact = (root / relative).resolve()
        require(artifact.is_relative_to(root.resolve()) and artifact.is_file() and sha256_file(artifact) == expected,
                f"Missing or corrupt Task 2 replay artifact: {relative}")
    lock = read_json(root / "manifests/run_lock.json")
    data = read_json(root / "splits/data_manifest.json")
    require(lock["config_id"] == manifest["config_id"] and fingerprint(data) == lock["data_id"] == manifest["data_id"], "Task 2 data/run identities differ.")
    require(data["supplied_hashes"] == verify_supplied_inputs(paths["resources"]), "Task 2 course assets differ from the original locked run.")
    intervention = read_json(root / "configs/intervention_locked.json")
    require(fingerprint(intervention) == lock["intervention_id"] and intervention["final_test_accessed"] is False,
            "The intervention is not the frozen development-only selection.")
    require(lock["mqar_final_test_started"] is True, "The final MQAR evaluation was never opened.")
    part_a = read_json(root / "metrics/attention/all_conditions.json")
    expected_a = {(method, dtype, length) for method in ["dense", "tiled", "sdpa"]
                  for dtype in ["float16", "float32"] for length in [256, 512, 1024, 2048, 4096, 8192]}
    require(len(part_a) == 36 and {(row["method"], row["dtype"], row["length"]) for row in part_a} == expected_a, "Part A condition matrix is incomplete.")
    for row in part_a:
        if row["status"] == "complete":
            for field in ["forward", "forward_backward"]:
                measure = row[field]
                require(measure["warmup_repetitions"] >= 1 and len(measure["samples_ms"]) >= 10
                        and all(math.isfinite(value) and value > 0 for value in measure["samples_ms"]), "Part A timing repetitions are incomplete.")
            require(bool(row["observed_operators"]), "Part A backend provenance is missing.")
            if row["length"] <= configuration["attention"]["reference_required_through"]:
                require("output_error" in row and len(row.get("gradient_errors", [])) == 3, "Part A output/gradient comparison is missing.")
    initializations = {}
    trained = {}
    checked_maps = {}
    for condition in training_conditions(configuration):
        row = read_json(root / f"metrics/mqar_training/{condition['id']}.json")
        trained[condition["id"]] = row
        require(all(row[key] == condition[key] for key in condition), "A saved MQAR run changes its seed/load/training regime.")
        if row["status"] == "complete":
            require(row["completed_updates"] == 4500 and row["checkpoint_updates"] == [500, 1000, 1500, 2000, 3000, 4500], "A successful MQAR run violates its training budget.")
            history = read_jsonl(root / row["history"])
            require([entry["update"] for entry in history] == list(range(1, 4501)), "MQAR training curve has missing or duplicated optimizer updates.")
            require(all(math.isfinite(entry["loss"]) for entry in history), "MQAR training history contains non-finite loss.")
        require(abs(row["trainable_parameters"] / row["baseline_trainable_parameters"] - 1) <= .05, "MQAR intervention exceeds its parameter budget.")
        previous = initializations.setdefault(condition["seed"], row["initialization_sha256"])
        require(previous == row["initialization_sha256"], "Matched architectures did not start from identical seeded parameters.")
        maps = row.get("training_query_maps", [])
        expected_index = (condition["seed"] - 6304) * 4500 * 32
        map_examples = 0
        for relative in maps:
            if not relative.endswith(".npz"):
                continue
            receipt = relative[:-4] + ".json"
            require(receipt in maps and relative in manifest["required_artifacts"], "MQAR training query-map receipt is missing.")
            if relative not in checked_maps:
                metadata = read_json(root / receipt)
                require(metadata["sha256"] == sha256_file(root / relative)
                        and metadata["generator_sha256"] == data["supplied_hashes"]["scripts/mqar_generator.py"], "MQAR query-map identity differs from the staff generator.")
                with np.load(root / relative, allow_pickle=False) as compressed:
                    array = compressed["queries"]
                    require(array.dtype == np.uint8 and array.shape == (metadata["examples"], metadata["K"], 4), "MQAR saved query annotations have invalid dimensions/dtype.")
                    require(int(compressed["start_index"][0]) == metadata["start_index"], "MQAR query-map starting index differs from its receipt.")
                    require(np.all(array[..., 0].astype(np.int16) - array[..., 2] == array[..., 3]), "Saved MQAR query-source distances are wrong.")
                    require(np.all(np.sort(array[..., 2], axis=-1) == np.arange(0, metadata["K"] * 2, 2)[None]), "Saved MQAR source positions are missing or duplicated.")
                    require(np.all((array[..., 1] >= 128) & (array[..., 1] < 192)) and np.all(np.diff(array[..., 0].astype(np.int16), axis=-1) > 0), "Saved MQAR answer IDs/query positions are invalid.")
                    lower, upper = {"near": (1, 63), "far": (128, 250), "mixed": (1, 250)}[metadata["regime"]]
                    require(np.all((array[..., 3] >= lower) & (array[..., 3] <= upper)), "Saved MQAR training distances violate their regime.")
                checked_maps[relative] = metadata
            metadata = checked_maps[relative]
            require((metadata["seed"], metadata["K"], metadata["regime"]) == (condition["seed"], condition["train_K"], condition["train_regime"]), "MQAR training query map belongs to another stream.")
            require(metadata["start_index"] == expected_index and metadata["stop_index"] - metadata["start_index"] == metadata["examples"], "MQAR training query-map index ranges have gaps or duplicates.")
            expected_index = metadata["stop_index"]
            map_examples += metadata["examples"]
        require(map_examples == row["completed_updates"] * 32, "MQAR training query annotations do not cover every completed batch.")
    print("Verified 42 training conditions and their saved query annotations.", flush=True)
    source_test = {row["id"]: row for row in load_evaluation_stream(paths["resources"], "test")} if deep else None
    for index, condition in enumerate(evaluation_conditions(configuration), 1):
        row = read_json(root / f"metrics/mqar_test/{condition['id']}.json")
        require(all(row[key] == condition[key] for key in condition), "MQAR evaluation variant differs from its fixed definition.")
        if row["status"] != "complete":
            if row["status"] == "unavailable_training_failed":
                require(trained[condition["training_id"]]["status"] != "complete", "A successful trained model is missing its final evaluation.")
            else:
                require(row["status"] in {"failed_cuda_oom", "failed_nonfinite_evaluation"} and bool(row.get("error")),
                        "An unsuccessful MQAR evaluation lacks an explicit recorded execution failure.")
            require(row["examples"] == 0 and row["cells"] == [], "An incomplete MQAR evaluation was reported as a complete test cell.")
            continue
        require(row["examples"] == 9216 and len(row["cells"]) == 9, "A completed MQAR evaluation lacks its full nine-cell final set.")
        require({(cell["test_K"], cell["regime"]) for cell in row["cells"]} == {(k, regime) for k in [4, 8, 16] for regime in ["near", "far", "mixed"]}, "MQAR final cells are duplicated or missing.")
        for cell in row["cells"]:
            require(cell["examples"] == 1024 and cell["queries"] == 1024 * cell["test_K"] and close(cell["accuracy"], cell["correct"] / cell["queries"]), "MQAR final query denominator is wrong.")
            if deep:
                records = [record for shard in sorted((root / f"predictions/mqar/{condition['id']}").glob(f"K{cell['test_K']}_{cell['regime']}_*.jsonl.gz")) for record in read_gzip_records(shard)]
                expected_ids = {key for key, example in source_test.items() if example["K"] == cell["test_K"] and example["regime"] == cell["regime"]}
                require(len(records) == 1024 and {record["id"] for record in records} == expected_ids, "MQAR final prediction shards do not cover their supplied IDs exactly.")
                for record in records:
                    original = source_test[record["id"]]
                    require(all(record[key] == original[key] for key in ["K", "regime", "query_positions", "source_positions", "distances", "answer_ids"]), "A saved MQAR prediction changed its gold labels or source distances.")
                recalculated = aggregate_recall(records)
                require(recalculated["correct"] == cell["correct"] and close(recalculated["mean_query_nll"], cell["mean_query_nll"]), "MQAR aggregate does not match saved query predictions.")
        if deep and index % 9 == 0:
            print(f"Verified final MQAR predictions: {index}/54 variants.", flush=True)
    require(len(read_json(root / "metrics/mqar_inference/all_conditions.json")) == 20, "MQAR inference sweep is incomplete.")
    tokenized = read_gzip_records(root / "splits/retrieval_tokenized.jsonl.gz")
    staff = {case["id"]: case for case in load_retrieval_cases(paths["resources"])}
    require(len(tokenized) == 180 and {case["id"] for case in tokenized} == set(staff), "KV tokenized prompt set differs from the staff set.")
    geometry = read_json(root / "metrics/kv/model_geometry.json")
    require(geometry["model"] == configuration["kv"]["model"] and geometry["revision"] == configuration["kv"]["revision"], "KV checkpoint/tokenizer revision differs.")
    records = []
    for case in tokenized:
        length = case["prefix_length"]
        require(case["code"] == staff[case["id"]]["code"] and len(case["token_ids"]) == case["actual_prompt_tokens"], "KV gold code or encoded prompt length changed.")
        channel_count = 2 * geometry["layers"] * geometry["kv_heads"] * geometry["head_dimension"]
        budget = 32 + channel_count * ((length // 64) * 36 + 2 * (length % 64))
        reference_row = read_json(root / f"predictions/kv/{case['id']}/fp16.json")
        for method in configuration["kv"]["methods"]:
            row = read_json(root / f"predictions/kv/{case['id']}/{method}.json")
            require((row["id"], row["method"], row["prompt_sha256"]) == (case["id"], method, case["prompt_sha256"]), "A saved KV prediction changed its method or prompt.")
            if row["status"] == "complete":
                if reference_row["status"] == "complete":
                    require(row["fp16_reference_correct"] == reference_row["correct"]
                            and row["fp16_reference_answer_token_ids"] == reference_row["response"]["token_ids"], "KV methods do not share the same FP16 reference continuation.")
                require(row["reference_int4_budget_bytes"] == budget, "KV reference budget omits metadata, residuals or actual context lengths.")
                if method != "fp16":
                    require(row["actual_serialized_prefix_bytes"] <= budget, "A cache policy exceeds its matched INT4 bytes.")
                if method in {"int4_channel", "int4_token"}:
                    require(row["actual_serialized_prefix_bytes"] == budget, "Uniform INT4 layouts do not use equal actual bytes.")
                require(row["correct"] == (extract_code(row["response"]["text"]) == case["code"]), "KV accuracy differs from the fixed deterministic extractor.")
                require(row["response"]["new_tokens"] == len(row["response"]["token_ids"]) <= configuration["kv"]["max_new_tokens"], "KV generation cap/token denominator differs.")
                for field in ["reference_answer_nll", "gold_code_nll"]:
                    require(row[field]["tokens"] > 0 and close(row[field]["mean_nll"], row[field]["nll_sum"] / row[field]["tokens"]), "KV answer NLL has an invalid denominator.")
                require(math.isfinite(row["post_prefill_logit_kl"]) and row["post_prefill_logit_kl"] >= 0, "KV logit divergence is invalid.")
                positions = row["selected_fp16_positions"]
                require(positions == sorted(set(positions)) and all(0 <= position < length for position in positions), "KV token indices are invalid.")
                if method in {"sink_recent", "question_aware", "hybrid"}:
                    mandatory = set(range(8)) | set(range(length - 64, length))
                    require(mandatory.issubset(positions), "A cache policy lost mandatory sink/recent safeguards.")
            records.append(row)
    intervals = read_json(root / "metrics/kv/paired_comparisons.json")
    require(len(intervals) == 9 and {(row["method"], row["reference"]) for row in intervals} == set(KV_PAIRS), "Principal KV paired comparisons are incomplete.")
    for interval in intervals:
        require(interval["paired_prompts"] == 180 and interval["resamples"] >= 2000, "KV confidence intervals do not cover the full paired prompt set.")
        if deep:
            expected = paired_accuracy_interval([row for row in records if row["method"] == interval["method"]],
                [row for row in records if row["method"] == interval["reference"]], interval["resamples"], configuration["seed"])
            require(all(close(interval[key], expected[key]) for key in ["difference", "ci_low", "ci_high"]), "KV paired interval does not match saved prompt outcomes.")
    systems = read_json(root / "metrics/kv_systems/all_conditions.json")
    require(len(systems) == 24 and {(row["method"], row["batch"], row["context_length"]) for row in systems} == {(method, batch, length) for method in configuration["kv"]["methods"] for batch in [1, 8] for length in [2048, 4096]}, "Packed-cache systems matrix is incomplete.")
    for row in systems:
        if row["status"] == "complete":
            require(row["serialized_bytes"] == row["calculated_bytes"], "Actual packed cache size differs from its formula.")
            require(all(len(row[part]["samples_ms"]) >= 10 and row[part]["warmup_repetitions"] >= 1 for part in ["encode", "unpack", "decode_step"]), "Packed-cache timing repetitions are incomplete.")
    style = read_json(root / "figures/style_manifest.json")
    require(len(style["figures"]) == 8 and len(list((root / "figures").rglob("*.png"))) == 8, "Task 2 requires exactly eight canonical evidence PNGs.")
    for relative, record in style["figures"].items():
        require(record["font_sizes_pt"] == [10.] and record["font_families"] == ["Times New Roman"]
                and sha256_file(root / relative) == record["sha256"], "A main figure is corrupt or is not Times New Roman 10 pt throughout.")
    return manifest


class ReplaySession:
    """Expose the same notebook interface using completed cloud evidence only."""

    def __init__(self, configuration, paths):
        self.configuration, self.paths = configuration, paths
        self.manifest = validate_replay(configuration, paths)
        print(f"Replay only | computed on {self.manifest['computed_on']} | run {self.manifest['run_id']}")

    def prepare_data(self):
        return read_json(self.paths["results"] / "splits/data_manifest.json")

    def run_attention(self, method):
        return read_json(self.paths["results"] / f"metrics/attention/{method}_conditions.json")

    def train_group(self, methods):
        return [read_json(self.paths["results"] / f"metrics/mqar_training/{condition['id']}.json")
                for condition in training_conditions(self.configuration) if condition["method"] in methods]

    def freeze_intervention(self):
        return read_json(self.paths["results"] / "configs/intervention_locked.json")

    def evaluate_mqar(self):
        return {"status": "saved final MQAR evaluation", "conditions": 54, "training_runs": 42}

    def run_mqar_inference(self):
        return read_json(self.paths["results"] / "metrics/mqar_inference/all_conditions.json")

    def validate_kv_reference(self):
        return read_json(self.paths["results"] / "metrics/kv/reference_checks.json")

    def check_cache_policies(self, family):
        record = read_json(self.paths["results"] / f"metrics/kv/policy_{family}.json")
        return {"status": record["status"], "family": family, "saved_policy_checks": len(record["conditions"])}

    def run_kv_quality(self):
        return {"status": "saved paired retrieval", "prompts": 180, "methods": 6, "records": 1080}

    def run_kv_systems(self):
        return read_json(self.paths["results"] / "metrics/kv_systems/all_conditions.json")

    def build_evidence(self):
        return {"status": self.manifest["status"], "source": self.manifest["computed_on"], "figures": 8,
                "font": "Times New Roman, 10 pt", "results": str(self.paths["results"])}

    def finish(self):
        result = validate_replay(self.configuration, self.paths)
        return {"status": result["status"], "source": result["computed_on"], "run_id": result["run_id"]}


def display_question(session, question, kind):
    """Show canonical saved evidence with its actual cloud provenance."""
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
                print(f"Showing 24 of {len(frame)} saved rows; complete data are in the CSV.")
                frame = frame.head(24)
            display(frame)
        else:
            display(read_json(root / relative))
