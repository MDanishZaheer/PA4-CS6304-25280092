# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Real INT2/INT4 KV payloads with FP16 scales, offsets and trailing-token buffers."""

import torch


def pack_integers(integers, bits):
    """Pack contiguous low-bit values into bytes in little-endian element order."""
    if bits not in {2, 4} or integers.shape[-1] != 64:
        raise ValueError("KV packing requires INT2/INT4 groups of exactly 64 elements.")
    per_byte = 8 // bits
    if torch.any(integers.to(torch.int16) >= 2 ** bits) or torch.any(integers < 0):
        raise ValueError("An integer is outside its KV quantization range.")
    grouped = integers.to(torch.int16).reshape(*integers.shape[:-1], 64 // per_byte, per_byte)
    shifts = torch.arange(per_byte, device=integers.device) * bits
    return (grouped << shifts).sum(dim=-1).to(torch.uint8)


def unpack_integers(payload, bits):
    """Invert the actual packed byte representation without using an unpacked cache."""
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=payload.device) * bits
    integers = (payload.to(torch.int16)[..., None] >> shifts) & (2 ** bits - 1)
    return integers.reshape(*payload.shape[:-1], 64).to(torch.uint8)


class GroupedTensor:
    """Store only integer bytes and FP16 scale/offset arrays for complete groups."""

    def __init__(self, tensor, bits, layout):
        if tensor.ndim != 4 or tensor.shape[-1] % 64 or tensor.shape[-2] % 64:
            raise ValueError("Grouped KV tensors require B,H,L,D and complete 64-token/feature groups.")
        if layout not in {"channel", "token"}:
            raise ValueError("Unknown KV grouping axis.")
        self.bits, self.layout = bits, layout
        self.batch, self.heads, self.tokens, self.dimension = tensor.shape
        grouped = tensor.transpose(-1, -2).contiguous() if layout == "channel" else tensor
        grouped = grouped.reshape(*grouped.shape[:-1], -1, 64).float()
        minimum, maximum = grouped.amin(dim=-1), grouped.amax(dim=-1)
        scale = (maximum - minimum) / (2 ** bits - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale).clamp(min=2 ** -24, max=65504)
        self.scales, self.offsets = scale.half(), minimum.half()
        if not torch.isfinite(self.scales).all() or not torch.isfinite(self.offsets).all():
            raise FloatingPointError("Non-finite KV scale or offset.")
        integers = torch.round((grouped - self.offsets.float()[..., None]) / self.scales.float()[..., None])
        self.payload = pack_integers(integers.clamp(0, 2 ** bits - 1).to(torch.uint8), bits)

    def decode(self):
        """Dequantize into FP16 for ordinary SDPA; report that allocation separately."""
        integers = unpack_integers(self.payload, self.bits)
        restored = (integers.float() * self.scales.float()[..., None] + self.offsets.float()[..., None]).half()
        if self.layout == "channel":
            return restored.reshape(self.batch, self.heads, self.dimension, self.tokens).transpose(-1, -2).contiguous()
        return restored.reshape(self.batch, self.heads, self.tokens, self.dimension)

    def append(self, other):
        """Append newly complete token groups without requantizing existing groups."""
        if (self.bits, self.layout, self.batch, self.heads, self.dimension) != (other.bits, other.layout, other.batch, other.heads, other.dimension):
            raise ValueError("Cannot append incompatible packed KV groups.")
        payload_axis, metadata_axis = (-2, -1) if self.layout == "channel" else (2, 2)
        self.payload = torch.cat([self.payload, other.payload], dim=payload_axis)
        self.scales = torch.cat([self.scales, other.scales], dim=metadata_axis)
        self.offsets = torch.cat([self.offsets, other.offsets], dim=metadata_axis)
        self.tokens += other.tokens

    def tensors(self):
        """Expose the exact serialized arrays, excluding Python object overhead."""
        return [self.payload, self.scales, self.offsets]


class PackedStream:
    """Compress complete 64-token prefixes and keep the remainder in FP16 for K/V."""

    def __init__(self, keys, values, bits, key_layout):
        self.bits, self.key_layout = bits, key_layout
        self.quantized_keys = self.quantized_values = None
        self.residual_keys, self.residual_values = keys[..., :0, :].clone(), values[..., :0, :].clone()
        self.append(keys, values)

    def append(self, keys, values):
        """Quantize new entries exactly when a full token group becomes available."""
        keys = torch.cat([self.residual_keys, keys], dim=-2)
        values = torch.cat([self.residual_values, values], dim=-2)
        complete = keys.shape[-2] // 64 * 64
        if complete:
            packed_keys = GroupedTensor(keys[..., :complete, :], self.bits, self.key_layout)
            packed_values = GroupedTensor(values[..., :complete, :], self.bits, "token")
            if self.quantized_keys is None:
                self.quantized_keys, self.quantized_values = packed_keys, packed_values
            else:
                self.quantized_keys.append(packed_keys)
                self.quantized_values.append(packed_values)
        self.residual_keys = keys[..., complete:, :].clone()
        self.residual_values = values[..., complete:, :].clone()

    @property
    def length(self):
        """Return the logical number of preserved positions, including the FP16 tail."""
        return (self.quantized_keys.tokens if self.quantized_keys is not None else 0) + self.residual_keys.shape[-2]

    def decode(self):
        """Recover ordered FP16 K/V tensors from packed payload plus residual buffers."""
        if self.quantized_keys is None:
            return self.residual_keys.clone(), self.residual_values.clone()
        return (torch.cat([self.quantized_keys.decode(), self.residual_keys], dim=-2),
                torch.cat([self.quantized_values.decode(), self.residual_values], dim=-2))

    def tensors(self):
        """List all physical arrays that must be serialized to reproduce this stream."""
        tensors = []
        if self.quantized_keys is not None:
            tensors.extend(self.quantized_keys.tensors())
            tensors.extend(self.quantized_values.tensors())
        return tensors + [self.residual_keys, self.residual_values]


def payload_bytes(layers, batch, heads, dimension, length, bits):
    """Apply the assignment's group/metadata/residual formula before header or indices."""
    if dimension % 64 or length < 0:
        raise ValueError("The cache geometry must support complete feature groups.")
    groups, residual = divmod(length, 64)
    return 2 * layers * batch * heads * dimension * (groups * (64 * bits // 8 + 4) + 2 * residual)


def tensor_bytes(tensors):
    """Sum actual typed payload arrays; do not equate allocated GPU bytes with storage."""
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
