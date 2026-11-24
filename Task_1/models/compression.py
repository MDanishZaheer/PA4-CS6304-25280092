# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Restore each independent condition and apply only eligible weight changes."""

import hashlib
import random

import numpy as np
import torch
from torch import nn

from Task_1.methods.pruning import row_prune, structured_prune
from Task_1.methods.quantization import quantize_groups


class CompensatedLinear(nn.Module):
    """Execute dense FP16 reconstructed weights with an explicit inverse scale."""

    def __init__(self, linear, scale):
        super().__init__()
        self.linear = linear
        self.register_buffer("input_scale", scale.to(device=linear.weight.device, dtype=torch.float16))

    @property
    def weight(self):
        """Expose the underlying projection for inventory and diagnostics."""
        return self.linear.weight

    @property
    def bias(self):
        """Preserve the unchanged, uncompressed original bias."""
        return self.linear.bias

    def forward(self, inputs):
        """Apply the serialized channel compensation before dense multiplication."""
        return self.linear(inputs / self.input_scale)


def seed_everything(seed):
    """Seed the permitted generators and disable approximate TF32 arithmetic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def eligible_projections(model, configuration):
    """Enumerate exactly seven eligible projections in every Transformer block."""
    projections = {}
    for index, block in enumerate(model.model.layers):
        for name in configuration["model"]["eligible"]:
            parent = block.self_attn if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else block.mlp
            module = getattr(parent, name)
            if module.weight.ndim != 2:
                raise ValueError(f"Eligible projection is not a matrix: layer {index}, {name}")
            path = f"model.layers.{index}.{'self_attn' if parent is block.self_attn else 'mlp'}.{name}"
            projections[path] = {"module": module, "parent": parent, "attribute": name, "layer": index}
    expected = 7 * len(model.model.layers)
    if len(projections) != expected or len({id(item["module"].weight) for item in projections.values()}) != expected:
        raise ValueError("Eligible projection enumeration contains missing or shared matrices.")
    return projections


def parameter_inventory(model, configuration):
    """Count tied parameters once and keep excluded parameters in model budgets."""
    projections = eligible_projections(model, configuration)
    eligible_ids = {id(item["module"].weight) for item in projections.values()}
    unique = {id(parameter): parameter for parameter in model.parameters()}
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in unique.values())
    eligible_bytes = sum(parameter.numel() * parameter.element_size() for key, parameter in unique.items() if key in eligible_ids)
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(str(id(parameter)), []).append(name)
    return {"parameters": sum(parameter.numel() for parameter in unique.values()),
            "eligible_parameters": sum(item["module"].weight.numel() for item in projections.values()),
            "dense_fp16_bytes": total_bytes, "uncompressed_bytes": total_bytes - eligible_bytes,
            "tied_parameter_groups": [names for names in aliases.values() if len(names) > 1],
            "projections": {name: {"name": name, "shape": list(item["module"].weight.shape),
                                    "elements": item["module"].weight.numel(), "layer": item["layer"]}
                            for name, item in projections.items()}}


def excluded_parameter_digest(model, configuration):
    """Prove embeddings, norms, head, and all biases remain byte-identical."""
    eligible_ids = {id(item["module"].weight) for item in eligible_projections(model, configuration).values()}
    digest = hashlib.sha256()
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in eligible_ids or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        digest.update(name.replace(".linear.", ".").encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_model(configuration, revision, cache_directory):
    """Load the base checkpoint in FP16 with ordinary dense SDPA execution."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Full Task 1 compute requires a Colab CUDA GPU.")
    seed_everything(configuration["seed"])
    tokenizer = AutoTokenizer.from_pretrained(configuration["model"]["name"], revision=revision,
                                               cache_dir=cache_directory, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(configuration["model"]["name"], revision=revision,
                                                cache_dir=cache_directory, torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.to("cuda").eval().requires_grad_(False)
    return model, tokenizer


class CompressionState:
    """Keep a CPU original so no condition inherits previously compressed weights."""

    def __init__(self, model, configuration):
        self.model = model
        self.configuration = configuration
        self.originals = {name: entry["module"].weight.detach().cpu().clone()
                          for name, entry in eligible_projections(model, configuration).items()}
        self.excluded_digest = excluded_parameter_digest(model, configuration)

    @torch.inference_mode()
    def restore(self):
        """Remove compensation wrappers and restore every eligible original tensor."""
        for name, entry in eligible_projections(self.model, self.configuration).items():
            module = entry["module"]
            if isinstance(module, CompensatedLinear):
                module = module.linear
                setattr(entry["parent"], entry["attribute"], module)
            module.weight.copy_(self.originals[name].to(module.weight.device))
        if excluded_parameter_digest(self.model, self.configuration) != self.excluded_digest:
            raise RuntimeError("An excluded parameter changed during compression.")

    @torch.inference_mode()
    def apply_full_precision_rescaling(self, profiles, bits):
        """Apply the identical AWQ compensation/folding graph without quantization."""
        self.restore()
        entries = eligible_projections(self.model, self.configuration)
        scales = {name: profiles[name]["scales"][str(bits)] for name in entries}
        for name, entry in entries.items():
            module = entry["module"]
            weight = self.originals[name].to(module.weight.device).float().clone()
            if name.endswith("up_proj"):
                if module.bias is not None:
                    raise ValueError("The down/up folding policy requires the checkpoint's bias-free up projection.")
                down_name = name.replace("up_proj", "down_proj")
                weight /= scales[down_name].to(weight.device).float()[:, None]
            weight *= scales[name].to(weight.device).float()[None, :]
            module.weight.copy_(weight.half())
            if not name.endswith("down_proj"):
                setattr(entry["parent"], entry["attribute"], CompensatedLinear(module, scales[name]))

    @torch.inference_mode()
    def apply(self, condition, activation_statistics, profiles, allocation=None):
        """Apply independent pruning/PTQ/AWQ using a preselected frozen policy."""
        self.restore()
        family = condition["family"]
        entries = eligible_projections(self.model, self.configuration)
        details = {}
        if family == "fp16":
            return details
        if family in {"P1", "P2", "P3", "structured"}:
            for name, entry in entries.items():
                weight = entry["module"].weight
                norms = activation_statistics[name]["input_norm"] if family != "P1" else None
                if family == "structured":
                    compressed, mask = structured_prune(weight, norms)
                else:
                    count = allocation["allocations"][name]["zero_count"] if family == "P3" else round(condition["sparsity"] * weight.numel())
                    compressed, mask = row_prune(weight, count, norms)
                weight.copy_(compressed)
                details[name] = {"selected_zero_count": int(mask.sum()), "effective_zero_count": int((weight == 0).sum()),
                                 "sparsity": float((weight == 0).sum() / weight.numel())}
        else:
            awq = family in {"Q2", "Q3"}
            scales = {}
            assignments = {name: allocation["allocations"][name] if family == "Q3" else condition["bits"] for name in entries}
            if awq:
                scales = {name: profiles[name]["scales"][str(assignments[name])] for name in entries}
            # Independent q/k/v and gate/up scales cannot be folded into their
            # shared RMSNorm. Unconstrained o scales do not commute through GQA.
            # The down inverse DOES commute with the gate*up product: fold it
            # into up's output rows before up is quantized, including metadata.
            fold_factors = {}
            if awq:
                for name, scale in scales.items():
                    if name.endswith("down_proj"):
                        fold_factors[name.replace("down_proj", "up_proj")] = scale
            for name, entry in entries.items():
                module = entry["module"]
                source_weight = self.originals[name].to(device=module.weight.device, dtype=torch.float32).clone()
                if name in fold_factors:
                    if module.bias is not None:
                        raise ValueError("The down/up folding policy requires a bias-free up projection.")
                    source_weight /= fold_factors[name].to(source_weight.device).float()[:, None]
                scale = scales.get(name)
                if scale is not None:
                    source_weight *= scale.to(source_weight.device).float()[None, :]
                reconstructed, _ = quantize_groups(source_weight, assignments[name], self.configuration["quantization"]["group_size"])
                module.weight.copy_(reconstructed.half())
                folded = awq and name.endswith("down_proj")
                if awq and not folded:
                    setattr(entry["parent"], entry["attribute"], CompensatedLinear(module, scale))
                details[name] = {"bits": assignments[name], "explicit_compensation": awq and not folded,
                                 "inverse_folded_into_up": folded,
                                 "up_rows_rescaled_for_down": name in fold_factors}
        if excluded_parameter_digest(self.model, self.configuration) != self.excluded_digest:
            raise RuntimeError("Compression modified an excluded weight or bias.")
        return details
