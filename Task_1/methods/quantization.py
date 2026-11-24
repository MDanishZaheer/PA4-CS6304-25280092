# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Matched asymmetric group PTQ, AWQ-style scaling, and real bit packing."""

import math

import torch
import torch.nn.functional as functional


def quantize_groups(weight, bits, group_size=128):
    """Quantize rows using stored FP16 scales and uint8 integer zero-points."""
    if weight.ndim != 2 or bits not in {2, 3, 4, 8} or group_size < 1:
        raise ValueError("Expected a matrix, supported bit width, and positive group size.")
    rows, columns = weight.shape
    padded_columns = math.ceil(columns / group_size) * group_size
    padded = functional.pad(weight.float(), (0, padded_columns - columns))
    groups = padded.reshape(rows, -1, group_size)
    minimum = groups.amin(dim=-1).clamp(max=0)
    maximum = groups.amax(dim=-1).clamp(min=0)
    levels = 2 ** bits - 1
    scale = (maximum - minimum) / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # The numerical model must use exactly the scales its packed format stores.
    scale = scale.clamp(min=2 ** -24, max=torch.finfo(torch.float16).max).half()
    if not torch.isfinite(scale).all():
        raise ValueError("Non-finite group scale.")
    zero_point = torch.round(-minimum / scale.float()).clamp(0, levels).to(torch.uint8)
    integers = torch.round(groups / scale.float()[..., None])
    integers = (integers + zero_point.float()[..., None]).clamp(0, levels).to(torch.uint8)
    reconstructed = scale.float()[..., None] * (integers.float() - zero_point.float()[..., None])
    reconstructed = reconstructed.reshape(rows, padded_columns)[:, :columns].to(weight.dtype)
    return reconstructed, {"integers": integers.reshape(rows, padded_columns),
                           "scales": scale, "zero_points": zero_point,
                           "original_columns": columns, "bits": bits, "group_size": group_size}


def pack_integers(integers, bits):
    """Pack a row-aligned little-endian stream, including cross-byte INT3 values."""
    if integers.ndim != 2 or bits not in {2, 3, 4, 8}:
        raise ValueError("Integer packing expects a two-dimensional supported-bit tensor.")
    if torch.any(integers < 0) or torch.any(integers.to(torch.int16) >= 2 ** bits):
        raise ValueError("An integer cannot be represented at the requested precision.")
    rows, columns = integers.shape
    bytes_per_row = math.ceil(columns * bits / 8)
    output = torch.zeros((rows, bytes_per_row), dtype=torch.int64, device=integers.device)
    positions = torch.arange(columns, device=integers.device) * bits
    byte_indices, shifts = positions // 8, positions % 8
    values = integers.to(torch.int64)
    output.scatter_add_(1, byte_indices[None, :].expand(rows, -1), (values << shifts[None, :]) & 255)
    crosses = shifts + bits > 8
    if crosses.any():
        output.scatter_add_(1, (byte_indices[crosses] + 1)[None, :].expand(rows, -1),
                            values[:, crosses] >> (8 - shifts[crosses])[None, :])
    return output.to(torch.uint8)


def unpack_integers(packed, bits, columns):
    """Invert packing while excluding byte-alignment padding."""
    if packed.ndim != 2 or packed.shape[1] != math.ceil(columns * bits / 8):
        raise ValueError("Packed payload has an invalid row length.")
    rows = packed.shape[0]
    positions = torch.arange(columns, device=packed.device) * bits
    indices, shifts = positions // 8, positions % 8
    values = packed.to(torch.int64)[:, indices] >> shifts[None, :]
    crosses = shifts + bits > 8
    if crosses.any():
        values[:, crosses] |= packed.to(torch.int64)[:, indices[crosses] + 1] << (8 - shifts[crosses])[None, :]
    return (values & (2 ** bits - 1)).to(torch.uint8).reshape(rows, columns)


def reconstruct_packed(payload):
    """Decode actual packed integers with the exact saved group metadata."""
    columns = payload["scales"].shape[1] * payload["group_size"]
    integers = unpack_integers(payload["packed"], payload["bits"], columns)
    groups = integers.reshape(*payload["scales"].shape, payload["group_size"])
    weights = payload["scales"].float()[..., None] * (groups.float() - payload["zero_points"].float()[..., None])
    return weights.flatten(1)[:, :payload["original_columns"]].half()


def packed_projection_bytes(rows, columns, bits, configuration, compensation=True):
    """Count payload, metadata, explicit inverse scales, header and alignment."""
    settings = configuration["quantization"]
    groups_per_row = math.ceil(columns / settings["group_size"])
    padded_columns = groups_per_row * settings["group_size"]
    integer_bytes = rows * math.ceil(padded_columns * bits / 8)
    scale_bytes = rows * groups_per_row * 2
    zero_point_bytes = rows * groups_per_row
    compensation_bytes = columns * 2 if compensation else 0
    header_bytes = settings["projection_header_bytes"]
    unaligned = integer_bytes + scale_bytes + zero_point_bytes + compensation_bytes + header_bytes
    alignment = settings["tensor_alignment_bytes"]
    padding_bytes = (-unaligned) % alignment
    return {"integer_payload_bytes": integer_bytes, "scale_bytes": scale_bytes,
            "zero_point_bytes": zero_point_bytes, "compensation_bytes": compensation_bytes,
            "header_bytes": header_bytes, "alignment_padding_bytes": padding_bytes,
            "padded_columns": padded_columns, "packed_bytes": unaligned + padding_bytes}


def channel_scale(mean_absolute, exponent, limits=(0.05, 20.0)):
    """Construct activation-dependent channel scales on a fixed candidate grid."""
    scale = mean_absolute.float().clamp(min=1e-6).pow(exponent)
    scale /= torch.exp(torch.log(scale).mean())
    return scale.clamp(min=limits[0], max=limits[1]).half()


def relative_output_error(reference_weight, compressed_weight, samples, input_scale=None):
    """Calculate local reconstruction distortion on the same sampled inputs."""
    inputs = samples.to(device=reference_weight.device, dtype=torch.float32)
    reference = functional.linear(inputs, reference_weight.float())
    adjusted = inputs if input_scale is None else inputs / input_scale.to(inputs.device).float()
    predicted = functional.linear(adjusted, compressed_weight.float())
    squared_error = (reference - predicted).square().sum(dtype=torch.float64)
    squared_reference = reference.square().sum(dtype=torch.float64)
    ratio = squared_error / squared_reference.clamp(min=1e-20)
    return {"relative_mse": float(ratio), "squared_error": float(squared_error),
            "squared_reference": float(squared_reference), "sample_tokens": inputs.shape[0]}


def search_awq_scale(weight, samples, mean_absolute, bits, configuration):
    """Select a simplified AWQ channel scale using calibration output error only."""
    settings = configuration["quantization"]
    candidates = []
    best = None
    for exponent in settings["scale_candidates"]:
        scale = channel_scale(mean_absolute.to(weight.device), exponent, settings["scale_clip"])
        transformed = weight.float() * scale.float()[None, :]
        reconstructed, _ = quantize_groups(transformed, bits, settings["group_size"])
        reconstructed = reconstructed.half()
        error = relative_output_error(weight, reconstructed, samples, scale)
        candidates.append({"exponent": exponent, **error})
        key = (error["relative_mse"], exponent)
        if best is None or key < best[0]:
            best = (key, scale.cpu())
    return best[1], {"bits": bits, "selected_exponent": best[0][1],
                     "relative_mse": best[0][0], "candidates": candidates}


def rescaling_equivalence(weight, samples, scale):
    """Test the full-precision algebra before integer rounding or FP16 execution."""
    inputs = samples[:32].to(device=weight.device, dtype=torch.float32)
    scales = scale.to(weight.device).float()
    reference = functional.linear(inputs, weight.float())
    adjusted = functional.linear(inputs / scales, weight.float() * scales[None, :])
    maximum = float((reference - adjusted).abs().max())
    relative = float((reference - adjusted).norm() / reference.norm().clamp(min=1e-12))
    if relative > 2e-5:
        raise RuntimeError(f"Full-precision channel compensation is not equivalent: {relative}")
    return {"relative_l2": relative, "maximum_absolute": maximum, "dtype": "float32"}
