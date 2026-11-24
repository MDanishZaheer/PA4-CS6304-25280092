# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Seven explicit assignment stages, verified Drive recovery and frozen scientific identities."""

import importlib.util
import subprocess
import sys
import time
from datetime import datetime, timezone

from Task_3.common.artifacts import ArtifactStore, read_json, read_jsonl
from Task_3.common.config import fingerprint, protocol_id, trained_variants
from Task_3.common.paths import resolve_paths


def open_session(configuration, project, mode):
    """Local notebooks replay saved evidence and never implicitly fall back to computation."""
    paths = resolve_paths(configuration, project)
    if mode == "replay":
        from Task_3.common.replay import ReplaySession
        return ReplaySession(configuration, paths)
    try:
        colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        colab = False
    if mode != "compute" or not colab:
        raise RuntimeError("Full Task 3 experiments run only in Colab compute mode; use replay locally.")
    return ComputeSession(configuration, paths)


class ComputeSession:
    """Expose readable notebook operations and retain all partial durable work."""

    def __init__(self, configuration, paths):
        from Task_3.cloud.colab import DriveSync
        from Task_3.common.runtime import runtime_metadata, seed_everything
        from Task_3.models.registry import ModelRegistry
        self.configuration, self.paths = configuration, paths
        sync = DriveSync(paths["results"], paths["drive_results"]) if configuration["execution"]["sync_to_drive"] else None
        if sync:
            sync.restore()
        self.store = ArtifactStore(paths, configuration, sync)
        seed_everything(configuration["seed"])
        runtime = runtime_metadata()
        identity = protocol_id(configuration, paths["task"])
        relative = "manifests/run_lock.json"
        if self.store.path(relative).is_file():
            self.lock = read_json(self.store.path(relative))
            if self.lock["config_id"] != identity:
                raise ValueError("Scientific code/configuration differs from the saved run; restore it or use a new run ID.")
            if any(self.lock["runtime"][key] != runtime[key] for key in ["gpu", "packages", "cuda", "compute_capability"]):
                raise ValueError("Resume with the original GPU and pinned software for matched timings.")
        else:
            self.lock = {"task": "Task_3", "run_id": configuration["execution"]["run_id"], "config_id": identity,
                "created_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime, "data_id": None,
                "protocol": "fixed supplied prompts, four independent final-quota adapters, no held-out hyperparameter selection"}
            self.store.save_json(relative, self.lock)
        self.store.save_json("configs/resolved_configuration.json", configuration)
        freeze = self.store.path("configs/environment_freeze.txt")
        if not freeze.is_file():
            freeze.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout, encoding="utf-8")
            self.store.backup(["configs/environment_freeze.txt"])
        self.registry = ModelRegistry(self.store)
        self.tokenizer = self.draft_tokenizer = None
        self.train = self.test = None
        print(f"Task 3 compute | {runtime['gpu']} | {self.lock['run_id']}", flush=True)

    def timed_stage(self, name, operation):
        """Keep all observed stage wall time, including partial attempts and backup overhead."""
        start, completed = time.perf_counter(), False
        try:
            result = operation()
            completed = True
            return result
        finally:
            ledger = read_jsonl(self.store.path("metrics/stage_wall_costs.jsonl"))
            ledger.append({"stage": name, "seconds": time.perf_counter() - start, "completed_attempt": completed,
                "ended_utc": datetime.now(timezone.utc).isoformat(), "includes": "model preparation, generation/scoring/training/merge and in-stage artifact/Drive I/O"})
            self.store.save_records("metrics/stage_wall_costs.jsonl", ledger)

    def prepare_data(self):
        from Task_3.data.prepare import prepare_data
        from Task_3.models.runtime import load_tokenizers
        self.tokenizer, self.draft_tokenizer = load_tokenizers(self.configuration, self.paths["data"] / "huggingface")
        self.train, self.test, data = prepare_data(self.store, self.tokenizer, self.draft_tokenizer)
        identity = fingerprint(data)
        if self.lock["data_id"] not in {None, identity}:
            raise ValueError("Prepared prompt/revision identity differs from the frozen run.")
        self.lock["data_id"] = identity
        self.store.save_json("manifests/run_lock.json", self.lock)
        return {"train_prompts": len(self.train), "test_prompts": len(self.test), "tokenizer_compatibility": "all 920 prompts checked"}

    def validate_sampler(self):
        from Task_3.evaluation.systems import validate_real_caches
        from Task_3.sampling.markov import validate_markov
        from Task_3.models.runtime import eos_ids
        if self.store.completed("sampler"):
            return read_json(self.store.path("metrics/sampler_markov.json"))["conditions"]
        if self.train is None:
            self.prepare_data()
        markov = validate_markov(self.configuration["sampling"]["markov_trials"])
        self.store.save_json("metrics/sampler_markov.json", markov)
        target, draft = self.registry.teacher(), self.registry.merged("original")
        if target.config.vocab_size != draft.config.vocab_size or eos_ids(target) != eos_ids(draft):
            raise ValueError("Model vocabulary or fixed EOS IDs differ.")
        cases = [self.train[0], self.train[1], self.train[500], self.train[501]]
        checks = validate_real_caches(target, draft, cases)
        self.store.save_json("metrics/cache_checks.json", checks)
        self.store.complete("sampler", ["metrics/sampler_markov.json", "metrics/cache_checks.json"], {"status": "passed"})
        return {"markov": markov["conditions"], "real_cache_checks": len(checks["conditions"])}

    def baseline(self):
        from Task_3.evaluation.benchmark import benchmark
        from Task_3.evaluation.systems import isolated_costs
        if not self.store.completed("sampler"):
            raise RuntimeError("Validate the exact sampler and real caches before timing.")
        if self.test is None:
            self.prepare_data()
        target = self.registry.teacher()
        isolated_costs(self.store, target, self.registry.merged("original"), "original")
        if self.store.completed("benchmark_baseline"):
            return {"paired_records": 480, "status": "saved pretrained baseline"}
        return self.timed_stage("baseline", lambda: benchmark(self.store, target, self.registry.merged, self.tokenizer, self.test, baseline=True))

    def train_method(self, method):
        from Task_3.models.lora import attach_lora
        from Task_3.models.runtime import load_model
        from Task_3.training.distill import prepare_teacher_pool, train_variant
        from Task_3.common.runtime import seed_everything
        if self.train is None:
            self.prepare_data()
        self.registry.clear_drafts()
        target, results = self.registry.teacher(), []
        for variant in trained_variants(self.configuration):
            if variant["method"] != method:
                continue
            name = variant["id"]
            if self.store.completed(f"train_{name}"):
                results.append(read_json(self.store.path(f"metrics/training/{name}.json")))
                continue
            if method == "off_policy" and not self.store.completed(f"teacher_pool_seed{variant['seed']}"):
                self.timed_stage(f"pool_seed{variant['seed']}", lambda: prepare_teacher_pool(self.store, target, self.train, variant["seed"]))
            def train_one():
                seed_everything(variant["seed"])
                student = attach_lora(load_model(self.configuration, "student", self.registry.cache), self.configuration["training"], variant["seed"])
                return train_variant(self.store, target, student, self.train, variant)
            results.append(self.timed_stage(f"train_{name}", train_one))
        return results

    def alignment(self):
        from Task_3.evaluation.banks import evaluate_alignment, generate_banks
        for variant in trained_variants(self.configuration):
            if not self.store.completed(f"train_{variant['id']}"):
                raise RuntimeError("Finish all four required training runs before the common-prefix comparison.")
        if self.test is None:
            self.prepare_data()
        checkpoints = {"teacher": self.configuration["models"]["teacher_revision"], "original": self.configuration["models"]["student_revision"]}
        checkpoints.update({row["id"]: read_json(self.store.path(f"metrics/training/{row['id']}.json"))["adapter_sha256"] for row in trained_variants(self.configuration)})
        target = self.registry.teacher()
        if not self.store.completed("banks"):
            self.timed_stage("prefix_banks", lambda: generate_banks(self.store, target, self.registry.unmerged, self.test, checkpoints))
        if not self.store.completed("alignment"):
            return self.timed_stage("alignment", lambda: evaluate_alignment(self.store, target, self.registry.unmerged, self.test, self.tokenizer))
        return {"status": "saved common-prefix comparisons", "banks": 7, "drafts": 5}

    def merged_execution(self):
        from Task_3.evaluation.benchmark import benchmark
        from Task_3.evaluation.systems import isolated_costs
        if self.test is None:
            self.prepare_data()
        for variant in trained_variants(self.configuration):
            name = variant["id"]
            if not self.store.completed(f"merge_{name}"):
                self.timed_stage(f"merge_{name}", lambda: self.registry.merge(name, [self.train[0], self.train[1], self.train[500], self.train[501]]))
        from Task_3.common.config import draft_ids
        target = self.registry.teacher()
        for name in draft_ids(self.configuration):
            isolated_costs(self.store, target, self.registry.merged(name), name)
        wrapper = "off_policy_seed6304"
        isolated_costs(self.store, target, self.registry.unmerged(wrapper), wrapper, wrapper=True)
        if not self.store.completed("benchmark_primary"):
            return self.timed_stage("primary_benchmark", lambda: benchmark(self.store, target, self.registry.merged, self.tokenizer, self.test))
        return {"status": "saved counterbalanced primary benchmarks", "paired_records": 2400}

    def workload_evaluation(self):
        from Task_3.evaluation.benchmark import benchmark
        from Task_3.evaluation.quality import evaluate_quality
        from Task_3.common.config import draft_ids
        if self.test is None:
            self.prepare_data()
        target = self.registry.teacher()
        if not self.store.completed("benchmark_fixed64"):
            self.timed_stage("fixed64_diagnostic", lambda: benchmark(self.store, target, self.registry.merged, self.tokenizer, self.test, fixed_length=True))
        if not self.store.completed("quality"):
            return self.timed_stage("standalone_quality", lambda: evaluate_quality(self.store,
                lambda name: target if name == "target" else self.registry.merged(name), self.tokenizer, self.test,
                ["target", *draft_ids(self.configuration)]))
        return {"status": "saved fixed64 and standalone quality", "quality_records": 1440, "fixed64_pairs": 480}

    def build_evidence(self):
        from Task_3.analysis.evidence import build_evidence
        return build_evidence(self.store)

    def finish(self):
        from Task_3.common.replay import validate_replay
        result = validate_replay(self.configuration, self.paths)
        return {"status": result["status"], "download": str(self.paths["drive_results"].parent), "computed_on": "Google Colab"}
