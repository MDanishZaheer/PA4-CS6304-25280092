# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Validate and display downloaded evidence without importing model libraries."""

import math

from Task_1.common.artifacts import read_json, read_jsonl, sha256_file
from Task_1.common.config import fingerprint, scientific_configuration, source_fingerprint
from Task_1.data.hellaswag import format_hellaswag
from Task_1.data.prepare import TASK_FILES, read_gzip_records, validate_task_data
from Task_1.training.protocol import experiment_conditions


def validate_replay(configuration, paths):
    """Check hashes, protocol identity, complete counts, and regenerated aggregates."""
    root = paths["results"]
    manifest_path = root / "manifests/run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No completed replay evidence at {manifest_path}. Run Task_1 on Colab, then copy its Drive "
            f"Task_1/results folder to {root.parent}, keeping the run folder inside it. "
            "Local replay never downloads a model or starts computation.")
    manifest = read_json(manifest_path)
    expected_id = fingerprint({"scientific_configuration": scientific_configuration(configuration),
                               "source_id": source_fingerprint(paths["task"]), "conditions": experiment_conditions()})
    if manifest.get("task") != "Task_1" or manifest.get("config_id") != expected_id:
        raise ValueError("Replay evidence does not match this task's configuration and implementation.")
    if manifest.get("status") not in {"complete", "complete_with_recorded_failures"}:
        raise ValueError("Replay requires a completed evidence manifest, with any failures explicitly recorded.")
    if manifest.get("conditions") != [condition["id"] for condition in experiment_conditions()]:
        raise ValueError("Replay manifest omits or reorders mandatory conditions.")
    for relative, expected in manifest["required_artifacts"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or corrupt replay artifact: {relative}")
    lock = read_json(root / "manifests/run_lock.json")
    if lock["data_id"] != manifest["data_id"] or not lock.get("allocation_lock_id"):
        raise ValueError("Replay data/allocation identities are inconsistent.")
    if fingerprint(read_json(root / "configs/allocations_locked.json")) != lock["allocation_lock_id"]:
        raise ValueError("Replay allocation policy is not the frozen pre-test policy.")
    selection = read_json(root / "splits/selection.json")
    if fingerprint(selection) != lock["data_id"]:
        raise ValueError("Saved data-selection metadata have changed.")
    data = {}
    for name, filename in TASK_FILES.items():
        path = root / "splits/materialized" / filename
        if sha256_file(path) != selection["materialized_hashes"][filename]:
            raise ValueError("Saved materialized windows/examples differ from the data lock.")
        data[name] = read_gzip_records(path)
    validate_task_data(data, configuration)
    formatted = [format_hellaswag(row["example"]) for row in data["hellaswag"]]
    for condition in experiment_conditions():
        name = condition["id"]
        summary = read_json(root / f"metrics/{name}/summary.json")
        if summary["condition"] != name or summary["config_id"] != manifest["config_id"] or summary["data_id"] != manifest["data_id"]:
            raise ValueError(f"Incompatible per-condition result: {name}")
        if summary["status"].startswith("failed_"):
            if name not in manifest.get("recorded_failures", {}):
                raise ValueError("A failed condition is missing from the failure register.")
            continue
        for suite in ["wikitext2", "c4"]:
            records = read_jsonl(root / f"predictions/{name}/{suite}.jsonl")
            if len({row["example_id"] for row in records}) != len(records):
                raise ValueError("Duplicate replay prediction-window IDs.")
            denominator = sum(row["scored_tokens"] for row in records)
            expected_windows = data[suite]
            if len(records) != len(expected_windows):
                raise ValueError("Saved window coverage differs from the authoritative materialization.")
            for record, window in zip(records, expected_windows):
                expected_window_id = f"{suite}:{window['source_row']}:{window['token_offset']}"
                if (record["example_id"] != expected_window_id or record["token_ids_sha256"] != fingerprint(window["token_ids"])
                        or record["score_target_offsets"] != window["score_target_offsets"]
                        or record["scored_tokens"] != window["scored_tokens"] or not math.isfinite(record["nll_sum"])):
                    raise ValueError("Saved scores have wrong contexts, offsets, counts, or non-finite NLL.")
            if denominator != 50000:
                raise ValueError(f"{name}/{suite} does not contain 50,000 scored tokens.")
            nll = math.fsum(row["nll_sum"] for row in records) / denominator
            if not math.isclose(nll, summary["quality"][suite]["mean_nll"], abs_tol=1e-10):
                raise ValueError("Saved pooled NLL differs from its per-window evidence.")
            ppl = summary["quality"][suite]["perplexity"]
            if (nll >= 709 and ppl is not None) or (nll < 709 and (ppl is None or not math.isclose(math.exp(nll), ppl, rel_tol=1e-10))):
                raise ValueError("Saved perplexity is inconsistent with pooled token NLL.")
        predictions = read_jsonl(root / f"predictions/{name}/hellaswag.jsonl")
        expected_count = condition["hellaswag_count"]
        if len(predictions) != expected_count or [row["order"] for row in predictions] != list(range(expected_count)):
            raise ValueError(f"Incomplete or reordered HellaSwag evidence for {name}.")
        for row in predictions:
            order = row["order"]
            prompt, endings, gold = formatted[order]
            if (row["example_id"] != f"hellaswag:{data['hellaswag'][order]['source_row_index']}"
                    or row["prompt"] != prompt or row["endings"] != endings or row["gold"] != gold
                    or row["character_lengths"] != [len(ending) for ending in endings]
                    or row["prompt_sha256"] != fingerprint({"prompt": prompt, "endings": endings})):
                raise ValueError("HellaSwag saved prompts, choices, labels or character lengths differ from the fixed examples.")
            if len(row["log_likelihoods"]) != 4 or len(row["character_lengths"]) != 4:
                raise ValueError("HellaSwag candidate evidence is incomplete.")
            normalized = [score / length for score, length in zip(row["log_likelihoods"], row["character_lengths"])]
            if not all(math.isfinite(score) for score in row["log_likelihoods"]):
                raise ValueError("Non-finite HellaSwag likelihood in completed evidence.")
            if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(normalized, row["character_normalized_scores"])):
                raise ValueError("HellaSwag normalization did not use ending character counts.")
            raw_prediction = max(range(4), key=lambda index: row["log_likelihoods"][index])
            normalized_prediction = max(range(4), key=lambda index: normalized[index])
            if (row["prediction"] != raw_prediction or row["prediction_normalized"] != normalized_prediction
                    or row["correct"] != int(raw_prediction == row["gold"]) or row["correct_normalized"] != int(normalized_prediction == row["gold"])):
                raise ValueError("HellaSwag decisions disagree with saved candidate scores.")
        for field, correctness in [("accuracy", "correct"), ("accuracy_normalized", "correct_normalized")]:
            accuracy = sum(row[correctness] for row in predictions) / expected_count
            if not math.isclose(accuracy, summary["quality"]["hellaswag"][field], abs_tol=1e-12):
                raise ValueError("Saved accuracy differs from its per-example evidence.")
        if summary["status"] == "complete":
            for length in ["512", "2048"]:
                measurement = summary["latency"]["shapes"][length]
                if len(measurement["prefill_ms"]) < 5 or len(measurement["decode_64_steps_ms"]) < 5:
                    raise ValueError("Incomplete warmed-up timing repetitions.")
                if not measurement["sdpa_observed_operators"]:
                    raise ValueError("Missing actual attention-backend evidence.")
    for question in ["rq1", "rq2", "rq3", "rq4"]:
        evidence = manifest["research_questions"].get(question)
        if not evidence or not any(evidence.values()):
            raise ValueError(f"Missing required research-question evidence: {question}")
        for relatives in evidence.values():
            for relative in relatives:
                if relative not in manifest["required_artifacts"]:
                    raise ValueError(f"Unregistered research-question artifact: {relative}")
    return manifest


class ReplaySession:
    """Read cloud outputs and expose the same short notebook stage interface."""

    def __init__(self, configuration, paths):
        self.configuration = configuration
        self.paths = paths
        self.manifest = validate_replay(configuration, paths)
        print(f"Replay only | original execution: {self.manifest['computed_on']} | run {self.manifest['run_id']}")

    def prepare_data(self):
        """Display saved data-selection metadata without downloading raw data."""
        return read_json(self.paths["results"] / "splits/selection.json")["summary"]

    def prepare_calibration_and_plan(self):
        """Show the previously frozen calibration-only allocation policy."""
        lock = read_json(self.paths["results"] / "manifests/run_lock.json")
        return {"status": "saved cloud calibration/selection complete", "allocation_lock_id": lock["allocation_lock_id"]}

    def run_group(self, family):
        """Read saved condition summaries; never import or load a model."""
        return [read_json(self.paths["results"] / f"metrics/{condition['id']}/summary.json")
                for condition in experiment_conditions() if condition["family"] == family]

    def build_evidence(self):
        """Use saved evidence without expensive rescoring or statistical recomputation."""
        return {"status": self.manifest["status"], "research_questions": list(self.manifest["research_questions"]),
                "source": self.manifest["computed_on"]}

    def finish(self):
        """Revalidate completion and provenance after notebook replay."""
        return {"status": validate_replay(self.configuration, self.paths)["status"],
                "source": self.manifest["computed_on"], "run_id": self.manifest["run_id"]}


def display_question(session, question, kind):
    """Display the exact saved tables, figures, metrics, or examples for one RQ."""
    from IPython.display import Image, display

    import pandas as pd

    root = session.paths["results"]
    manifest = read_json(root / "manifests/run.json")
    for relative in manifest["research_questions"][question].get(kind, []):
        print(relative)
        if kind == "tables":
            frame = pd.read_csv(root / relative)
            if len(frame) > 40:
                print(f"Showing the first 20 of {len(frame)} saved rows; the complete table is in the CSV.")
                display(frame.head(20))
            else:
                display(frame)
        elif kind == "figures":
            display(Image(filename=str(root / relative)))
        else:
            display(read_json(root / relative))
