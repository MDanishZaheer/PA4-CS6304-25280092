# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Append packed KV entries while separating absolute RoPE IDs from physical length."""

import copy
import struct

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from Task_2.cache.packing import GroupedTensor, PackedStream, tensor_bytes


METHOD_CODES = {"fp16": 0, "sink_recent": 1, "question_aware": 2,
                "int4_token": 3, "int4_channel": 4, "hybrid": 5}


def clone_dynamic_cache(cache):
    """Make an independent mutable cache without recomputing or changing cached values."""
    layers = []
    for original in cache.layers:
        layer = DynamicLayer()
        layer.update(original.keys.clone(), original.values.clone())
        layers.append(layer)
    return Cache(layers=layers)


class PackedLayer:
    """Expose a dynamic-cache layer interface while persisting actual packed arrays."""

    is_sliding = False
    is_compileable = False
    is_initialized = True

    def __init__(self, keys, values, plan, positions):
        self.method = plan["method"]
        self.absolute_length = keys.shape[-2]
        self.dtype, self.device = keys.dtype, keys.device
        self.positions = positions
        self.stream = None
        if self.method in {"fp16", "sink_recent", "question_aware"}:
            self.high_keys = keys.clone() if self.method == "fp16" else keys.index_select(-2, positions.long())
            self.high_values = values.clone() if self.method == "fp16" else values.index_select(-2, positions.long())
        else:
            if self.method == "hybrid":
                self.high_keys, self.high_values = keys.index_select(-2, positions.long()), values.index_select(-2, positions.long())
                mask = torch.ones(keys.shape[-2], device=keys.device, dtype=torch.bool)
                mask[positions.long()] = False
                low_keys, low_values = keys[..., mask, :], values[..., mask, :]
            else:
                self.high_keys, self.high_values = keys[..., :0, :].clone(), values[..., :0, :].clone()
                low_keys, low_values = keys, values
            self.stream = PackedStream(low_keys, low_values, plan["bits"], plan["layout"])

    def materialize(self):
        """Create transient dense FP16 tensors for the unchanged SDPA attention path."""
        if self.stream is None:
            return self.high_keys, self.high_values
        keys, values = self.stream.decode()
        if self.method != "hybrid":
            return keys, values
        shape = list(keys.shape)
        shape[-2] += self.high_keys.shape[-2]
        full_keys, full_values = torch.empty(shape, device=keys.device, dtype=keys.dtype), torch.empty(shape, device=values.device, dtype=values.dtype)
        low_mask = torch.ones(shape[-2], device=keys.device, dtype=torch.bool)
        low_mask[self.positions.long()] = False
        full_keys[..., self.positions.long(), :], full_values[..., self.positions.long(), :] = self.high_keys, self.high_values
        full_keys[..., low_mask, :], full_values[..., low_mask, :] = keys, values
        return full_keys, full_values

    @property
    def keys(self):
        return self.materialize()[0]

    @property
    def values(self):
        return self.materialize()[1]

    def update(self, key_states, value_states, cache_kwargs=None):
        """Append already-rotated keys using physical storage lengths for causal masks."""
        self.absolute_length += key_states.shape[-2]
        if self.stream is None:
            self.high_keys = torch.cat([self.high_keys, key_states], dim=-2)
            self.high_values = torch.cat([self.high_values, value_states], dim=-2)
        else:
            self.stream.append(key_states, value_states)
        return self.materialize()

    def get_seq_length(self):
        return self.high_keys.shape[-2] + (self.stream.length if self.stream is not None else 0)

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1

    def tensors(self):
        """Return exactly the serialized layer payload, excluding shared position indices."""
        return [self.high_keys, self.high_values] + (self.stream.tensors() if self.stream is not None else [])


class BudgetCache(Cache):
    """Use real low-bit storage and expose dequantized values only during SDPA calls."""

    def __init__(self, original_cache, plan):
        self.plan = copy.deepcopy(plan)
        sample = original_cache.layers[0].keys
        if sample.dtype != torch.float16:
            raise ValueError("The matched KV formats require the reference cache in FP16.")
        self.batch, self.heads, _, self.dimension = sample.shape
        self.positions = torch.tensor(plan["positions"], device=sample.device, dtype=torch.int32)
        layers = [PackedLayer(layer.keys, layer.values, plan, self.positions) for layer in original_cache.layers]
        super().__init__(layers=layers)
        if self.serialized_bytes() != plan["expected_bytes"]:
            raise ValueError("Actual KV payload differs from its header/metadata/index-inclusive budget.")

    @property
    def absolute_length(self):
        return self.layers[0].absolute_length

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.plan["method"] in {"int4_token", "int4_channel", "hybrid"} and key_states.shape[-2] != 1:
            raise ValueError("Packed cache appends must be causal one-token updates; use model_step for a multi-token suffix.")
        if layer_idx == 0 and self.plan["method"] in {"sink_recent", "question_aware"}:
            appended = torch.arange(self.absolute_length, self.absolute_length + key_states.shape[-2], device=key_states.device, dtype=torch.int32)
            self.positions = torch.cat([self.positions, appended])
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def clone(self):
        """Clone the packed payload for independent free-running and teacher-forced paths."""
        return copy.deepcopy(self)

    def serialized_bytes(self):
        """Count actual payload arrays plus the physical 32-byte header and int32 indices."""
        return 32 + tensor_bytes([self.positions]) + sum(tensor_bytes(layer.tensors()) for layer in self.layers)

    def serialize(self):
        """Emit the documented binary format to verify its real byte length."""
        header = struct.pack("<8I", 0x5041344B, 256 + METHOD_CODES[self.plan["method"]], len(self.layers),
                             self.batch, self.heads, self.dimension, self.absolute_length, self.positions.numel())
        tensors = [self.positions] + [tensor for layer in self.layers for tensor in layer.tensors()]
        payload = header + b"".join(tensor.detach().cpu().contiguous().numpy().tobytes() for tensor in tensors)
        if len(payload) != self.serialized_bytes():
            raise RuntimeError("The emitted cache binary does not match actual byte accounting.")
        return payload

    @classmethod
    def deserialize(cls, payload, device="cpu"):
        """Decode the binary format without access to the original FP16 cache."""
        if len(payload) < 32:
            raise ValueError("Truncated packed-cache header.")
        magic, format_code, count, batch, heads, dimension, absolute, selected = struct.unpack("<8I", payload[:32])
        reverse = {value + 256: key for key, value in METHOD_CODES.items()}
        if magic != 0x5041344B or format_code not in reverse or dimension % 64 or not count or not batch or not heads:
            raise ValueError("Invalid packed-cache format or geometry.")
        method = reverse[format_code]
        if selected > absolute or (method in {"fp16", "int4_token", "int4_channel"} and selected):
            raise ValueError("Invalid shared position-index count.")
        offset = 32

        def take(shape, dtype):
            """Read a precisely sized typed array and reject missing payload bytes."""
            nonlocal offset
            elements = 1
            for size in shape:
                elements *= size
            size = elements * torch.empty((), dtype=dtype).element_size()
            if offset + size > len(payload):
                raise ValueError("Truncated packed-cache payload.")
            if size:
                result = torch.frombuffer(bytearray(payload[offset:offset + size]), dtype=dtype).clone().reshape(shape).to(device)
            else:
                result = torch.empty(shape, dtype=dtype, device=device)
            offset += size
            return result

        positions = take((selected,), torch.int32)
        if selected and (positions.min() < 0 or positions.max() >= absolute or not torch.all(positions[1:] > positions[:-1])):
            raise ValueError("Packed-cache positions must be distinct, sorted and in bounds.")
        bits, layout = (2 if method == "hybrid" else 4), ("token" if method == "int4_token" else "channel")

        def grouped(tokens, axis):
            """Reconstruct GroupedTensor metadata and its actual byte payload."""
            result = GroupedTensor.__new__(GroupedTensor)
            result.bits, result.layout = bits, axis
            result.batch, result.heads, result.tokens, result.dimension = batch, heads, tokens, dimension
            metadata_shape = (batch, heads, dimension, tokens // 64) if axis == "channel" else (batch, heads, tokens, dimension // 64)
            result.payload = take(metadata_shape + (64 * bits // 8,), torch.uint8)
            result.scales, result.offsets = take(metadata_shape, torch.float16), take(metadata_shape, torch.float16)
            return result

        layers = []
        for _ in range(count):
            layer = PackedLayer.__new__(PackedLayer)
            layer.method, layer.absolute_length, layer.positions = method, absolute, positions
            layer.dtype, layer.device = torch.float16, torch.device(device)
            high = absolute if method == "fp16" else selected
            layer.high_keys = take((batch, heads, high, dimension), torch.float16)
            layer.high_values = take((batch, heads, high, dimension), torch.float16)
            layer.stream = None
            if method in {"int4_token", "int4_channel", "hybrid"}:
                low = absolute - high
                complete, residual = low // 64 * 64, low % 64
                stream = PackedStream.__new__(PackedStream)
                stream.bits, stream.key_layout = bits, layout
                stream.quantized_keys = grouped(complete, layout) if complete else None
                stream.quantized_values = grouped(complete, "token") if complete else None
                stream.residual_keys = take((batch, heads, residual, dimension), torch.float16)
                stream.residual_values = take((batch, heads, residual, dimension), torch.float16)
                layer.stream = stream
            layers.append(layer)
        if offset != len(payload):
            raise ValueError("Unexpected trailing packed-cache bytes.")
        result = cls.__new__(cls)
        result.plan = {"method": method, "positions": positions.cpu().tolist(), "bits": 16 if method in {"fp16", "sink_recent", "question_aware"} else bits,
                       "layout": "fp16" if method in {"fp16", "sink_recent", "question_aware"} else layout,
                       "expected_bytes": len(payload)}
        result.batch, result.heads, result.dimension, result.positions = batch, heads, dimension, positions
        Cache.__init__(result, layers=layers)
        if result.serialized_bytes() != len(payload):
            raise ValueError("Decoded cache does not account for every serialized byte.")
        return result


@torch.inference_mode()
def model_step(model, token_ids, cache=None, absolute_start=None, all_logits=False):
    """Use absolute rotary positions and a separate physical append range after eviction."""
    if isinstance(cache, BudgetCache) and cache.plan["method"] in {"int4_token", "int4_channel", "hybrid"} and token_ids.shape[-1] > 1:
        # Packing a whole teacher-forced suffix at once can use future tokens in
        # channel-group extrema. Append causally so earlier logits cannot leak.
        first_absolute = cache.absolute_length if absolute_start is None else absolute_start
        outputs = []
        for index in range(token_ids.shape[-1]):
            current, cache = model_step(model, token_ids[:, index:index + 1], cache,
                                        first_absolute + index, all_logits=True)
            if all_logits or index + 1 == token_ids.shape[-1]:
                outputs.append(current)
        return torch.cat(outputs, dim=1), cache
    physical = cache.get_seq_length() if cache is not None else 0
    absolute = absolute_start if absolute_start is not None else getattr(cache, "absolute_length", physical)
    physical_positions = torch.arange(physical, physical + token_ids.shape[-1], device=token_ids.device)
    absolute_positions = torch.arange(absolute, absolute + token_ids.shape[-1], device=token_ids.device)[None]
    output = model.model(input_ids=token_ids, past_key_values=cache, use_cache=True,
                         cache_position=physical_positions, position_ids=absolute_positions, return_dict=True)
    hidden = output.last_hidden_state if all_logits else output.last_hidden_state[:, -1:]
    logits = model.lm_head(hidden).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Cache evaluation produced non-finite logits.")
    return logits, output.past_key_values
