"""Vectorized PyTorch Winograd F(2x2, 3x3) convolution for NCHW tensors."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as functional

Precision = Literal["fp32", "fp16", "bf16", "int8", "stagewise_int8", "input_int8"]

_G_VALUES = (
    (1.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.0, 0.0, 1.0),
)
_BT_VALUES = (
    (1.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0),
    (0.0, -1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, -1.0),
)
_A_VALUES = (
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (0.0, -1.0),
)


@lru_cache(maxsize=24)
def _transform_matrices(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.tensor(_G_VALUES, device=device, dtype=dtype)
    bt = torch.tensor(_BT_VALUES, device=device, dtype=dtype)
    b = bt.transpose(0, 1).contiguous()
    a = torch.tensor(_A_VALUES, device=device, dtype=dtype)
    at = a.transpose(0, 1).contiguous()
    return g, bt, b, a, at


def _padding_pair(padding: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(padding, int):
        pair = (padding, padding)
    else:
        if len(padding) != 2:
            raise ValueError("padding must be an int or a (height, width) pair")
        pair = (int(padding[0]), int(padding[1]))
    if min(pair) < 0:
        raise ValueError("padding values must be non-negative")
    return pair


def _per_position_quantize(
    values: torch.Tensor, reduce_dims: tuple[int, ...], clip_percentile: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < clip_percentile <= 100.0:
        raise ValueError("clip_percentile must be in (0, 100]")
    magnitude = values.detach().abs().float()
    if clip_percentile == 100.0:
        bound = magnitude.amax(dim=reduce_dims, keepdim=True)
    else:
        kept_dims = tuple(i for i in range(values.ndim) if i not in reduce_dims)
        permutation = kept_dims + reduce_dims
        permuted = magnitude.permute(permutation)
        reduce_size = math.prod(values.shape[i] for i in reduce_dims)
        bound = torch.quantile(
            permuted.reshape(*[values.shape[i] for i in kept_dims], reduce_size),
            clip_percentile / 100.0,
            dim=-1,
        )
        for dim in sorted(reduce_dims):
            bound = bound.unsqueeze(dim)
    bound = bound.clamp_min(1e-12)
    scale = bound / 127.0
    quantized = torch.round(values.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _int8_channel_gemm(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    if left.device.type != "cuda":
        return torch.matmul(left.to(torch.int64), right.to(torch.int64)).to(torch.int32)
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("This PyTorch build does not provide CUDA int8 GEMM (torch._int_mm)")
    rows, reduction = left.shape
    _, columns = right.shape
    padded_rows = max(32, math.ceil(rows / 32) * 32)
    padded_reduction = max(32, math.ceil(reduction / 32) * 32)
    padded_columns = max(16, math.ceil(columns / 16) * 16)
    left_padded = torch.zeros(
        (padded_rows, padded_reduction), dtype=torch.int8, device=left.device
    )
    right_padded = torch.zeros(
        (padded_reduction, padded_columns), dtype=torch.int8, device=right.device
    )
    left_padded[:rows, :reduction] = left
    right_padded[:reduction, :columns] = right
    return torch._int_mm(left_padded, right_padded)[:rows, :columns]


def winograd_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    padding: int | tuple[int, int] = 0,
    precision: Precision = "fp32",
    clip_percentile: float = 100.0,
    return_stages: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Winograd F(2x2, 3x3) to NCHW inputs and OIHW filters.

    INT8 quantizes transformed filters and input tiles independently per
    Winograd coordinate, accumulates the channel reduction in int32, and
    dequantizes before the inverse transform. ``stagewise_int8`` additionally
    requantizes the transformed product. ``input_int8`` quantizes before the
    transforms as a comparison baseline; it is not an integer Winograd kernel.
    """
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("input and weight must have shapes (N,C,H,W) and (F,C,3,3)")
    if weight.shape[-2:] != (3, 3):
        raise ValueError("Winograd F(2x2,3x3) requires 3x3 filters")
    if input.shape[1] != weight.shape[1]:
        raise ValueError("input channels must match the filter input channels")
    if input.shape[0] == 0 or input.shape[1] == 0 or min(input.shape[-2:]) < 1:
        raise ValueError("input dimensions and batch size must be non-empty")
    if weight.shape[0] == 0:
        raise ValueError("at least one output filter is required")
    if input.device != weight.device:
        raise ValueError("input and weight must be on the same device")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != input.device):
        raise ValueError("bias must have shape (out_channels,) on the input device")
    if precision not in ("fp32", "fp16", "bf16", "int8", "stagewise_int8", "input_int8"):
        raise ValueError(f"unsupported precision: {precision}")

    pad_h, pad_w = _padding_pair(padding)
    x = input.to(dtype=torch.float32)
    filters = weight.to(dtype=torch.float32)
    if precision == "input_int8":
        xq, xs = _per_position_quantize(x, (0, 1, 2, 3), 100.0)
        wq, ws = _per_position_quantize(filters, (0, 1, 2, 3), 100.0)
        x = xq.float() * xs
        filters = wq.float() * ws
        compute_precision: Precision = "fp32"
    else:
        compute_precision = precision

    output_h = x.shape[2] + 2 * pad_h - 2
    output_w = x.shape[3] + 2 * pad_w - 2
    if output_h <= 0 or output_w <= 0:
        raise ValueError("padded input must be at least as large as a 3x3 filter")
    tiles_h = math.ceil(output_h / 2)
    tiles_w = math.ceil(output_w / 2)
    extra_h = 2 * tiles_h - output_h
    extra_w = 2 * tiles_w - output_w
    x = functional.pad(x, (pad_w, pad_w + extra_w, pad_h, pad_h + extra_h))
    tiles = x.unfold(2, 4, 2).unfold(3, 4, 2)
    n, channels, _, _, _, _ = tiles.shape
    tile_count = tiles_h * tiles_w

    if compute_precision == "fp16":
        compute_dtype = torch.float16
    elif compute_precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    g, bt, b, a, at = _transform_matrices(x.device, compute_dtype)
    transformed_input = torch.einsum(
        "ai,ncuvij,jb->ncuvab",
        bt,
        tiles.to(compute_dtype),
        b,
    )
    transformed_filter = torch.einsum(
        "ai,fcij,bj->fcab",
        g,
        filters.to(compute_dtype),
        g,
    )

    transformed_input = transformed_input.reshape(n, channels, tile_count, 16)
    transformed_filter = transformed_filter.reshape(weight.shape[0], channels, 16)

    if compute_precision in ("int8", "stagewise_int8"):
        u_q, u_scale = _per_position_quantize(
            transformed_filter, (0, 1), clip_percentile
        )
        v_q, v_scale = _per_position_quantize(
            transformed_input, (0, 1, 2), clip_percentile
        )
        u = u_q.permute(2, 0, 1).contiguous().to(torch.int32)
        v = v_q.permute(3, 1, 0, 2).reshape(16, channels, n * tile_count)
        product = torch.stack(
            [_int8_channel_gemm(u[k], v[k]) for k in range(16)]
        ).reshape(16, weight.shape[0], n, tile_count)
        scales = (u_scale.reshape(16) * v_scale.reshape(16)).view(16, 1, 1, 1)
        product = product.float() * scales
        if compute_precision == "stagewise_int8":
            product_k = product.permute(1, 2, 3, 0)
            product_q, product_scale = _per_position_quantize(
                product_k, (0, 1, 2), clip_percentile
            )
            product = (
                product_q.float() * product_scale
            ).permute(3, 0, 1, 2)
        product = product.permute(2, 1, 3, 0)
    else:
        u = transformed_filter.permute(2, 0, 1).contiguous()
        v = transformed_input.permute(3, 1, 0, 2).reshape(
            16, channels, n * tile_count
        )
        product = torch.bmm(u, v).reshape(16, weight.shape[0], n, tile_count)
        product = product.permute(2, 1, 3, 0)

    product = product.reshape(n, weight.shape[0], tiles_h, tiles_w, 4, 4)
    output_tiles = torch.einsum(
        "ai,nftwij,jb->nftwab",
        at,
        product,
        a,
    )
    output = (
        output_tiles.permute(0, 1, 2, 4, 3, 5)
        .reshape(n, weight.shape[0], tiles_h * 2, tiles_w * 2)
    )[:, :, :output_h, :output_w]
    output = output.float()
    if bias is not None:
        output = output + bias.float().view(1, -1, 1, 1)

    if return_stages:
        stages = {
            "input_transform": transformed_input,
            "filter_transform": transformed_filter,
            "hadamard": product,
            "output_transform": output_tiles,
        }
        return output, stages
    return output


def torch_conv2d_precision(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    padding: int | tuple[int, int] = 0,
    precision: Literal["fp32", "fp16", "bf16", "input_int8"] = "fp32",
) -> torch.Tensor:
    """Run a cuDNN/PyTorch convolution at a matching floating precision."""
    if precision == "input_int8":
        x = input.float()
        w = weight.float()
        xq, xs = _per_position_quantize(x, (0, 1, 2, 3), 100.0)
        wq, ws = _per_position_quantize(w, (0, 1, 2, 3), 100.0)
        return functional.conv2d(xq.float() * xs, wq.float() * ws, bias, padding=padding)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        precision
    )
    if dtype is None:
        raise ValueError(f"unsupported cuDNN precision: {precision}")
    result = functional.conv2d(
        input.to(dtype), weight.to(dtype), None if bias is None else bias.to(dtype),
        padding=padding,
    )
    return result.float()
