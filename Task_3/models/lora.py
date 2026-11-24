# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Small transparent LoRA modules with FP32 trainable parameters and explicit merging."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional


class LoRALinear(nn.Module):
    """Add a rank-r update to an unchanged frozen linear projection."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base, self.rank, self.scale = base, rank, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.enabled = True

    def forward(self, inputs):
        original = self.base(inputs)
        if not self.enabled:
            return original
        update = functional.linear(functional.linear(inputs.float(), self.A), self.B)
        return original + (self.scale * update).to(original.dtype)


def attach_lora(model, settings, seed):
    """Initialize each independent method identically for its prescribed seed."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.random.fork_rng(devices=[next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []):
        torch.manual_seed(seed)
        for name, module in list(model.named_modules()):
            if name.rsplit(".", 1)[-1] in settings["lora_targets"] and isinstance(module, nn.Linear):
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent), child, LoRALinear(module, settings["lora_rank"], settings["lora_alpha"]))
    return model


def adapter_state(model):
    """Save only the named FP32 adapter tensors, leaving base weights on the Hub."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters() if name.endswith((".A", ".B"))}


def load_adapter(model, state, enabled=True):
    """Reject mismatched tensor names, shapes and precision when swapping adapters."""
    parameters = dict(model.named_parameters())
    if set(state) != {name for name in parameters if name.endswith((".A", ".B"))}:
        raise ValueError("Adapter keys do not match the fixed architecture.")
    with torch.no_grad():
        for name, value in state.items():
            if value.dtype != torch.float32 or value.shape != parameters[name].shape:
                raise ValueError("Adapter shape or FP32 precision differs.")
            parameters[name].copy_(value.to(parameters[name].device))
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def adapter_digest(model_or_state):
    """Bind rollouts and recovered work to exact adapter values."""
    state = adapter_state(model_or_state) if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def merge_into_fresh(fresh_model, state, settings):
    """Merge in FP32 and cast once into a separate ordinary FP16 backbone."""
    expected = set()
    with torch.no_grad():
        for name, module in fresh_model.named_modules():
            if name.rsplit(".", 1)[-1] in settings["lora_targets"] and isinstance(module, nn.Linear):
                first, second = name + ".A", name + ".B"
                expected.update([first, second])
                delta = state[second].to(module.weight.device) @ state[first].to(module.weight.device)
                module.weight.copy_((module.weight.float() + delta * settings["lora_alpha"] / settings["lora_rank"]).to(module.weight.dtype))
    if expected != set(state):
        raise ValueError("Merging did not consume exactly the prescribed adapter tensors.")
    return fresh_model.eval()
