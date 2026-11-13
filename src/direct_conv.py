"""
direct_conv.py
==============
Direct 2-D convolution reference implementations in FP32, FP16, and
simulated INT8 precisions.

All routines accept **single-channel** 2-D arrays (H × W for the input,
Kh × Kw for the kernel) and return the "valid" convolution output
((H-Kh+1) × (W-Kw+1)).

For multi-channel or batched inputs, callers should loop over channels /
batch elements and accumulate, or use the convenience wrappers provided
at the bottom of this module.

Dependencies: numpy only (no PyTorch required).

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .precision_utils import (
    quantize_to_int8,
    dequantize_from_int8,
    simulate_fp16_rounding,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _validate_inputs(inp: np.ndarray, kernel: np.ndarray) -> None:
    """Raise informative errors if inputs are incompatible."""
    if inp.ndim != 2:
        raise ValueError(
            f"Input must be a 2-D array; got shape {inp.shape}."
        )
    if kernel.ndim != 2:
        raise ValueError(
            f"Kernel must be a 2-D array; got shape {kernel.shape}."
        )
    if kernel.shape[0] > inp.shape[0] or kernel.shape[1] > inp.shape[1]:
        raise ValueError(
            f"Kernel {kernel.shape} is larger than input {inp.shape}."
        )


def _output_shape(
    inp_shape: Tuple[int, int],
    ker_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Return (out_h, out_w) for a *valid* convolution."""
    return (
        inp_shape[0] - ker_shape[0] + 1,
        inp_shape[1] - ker_shape[1] + 1,
    )


# ---------------------------------------------------------------------------
# Single-channel sliding-window convolution  (the inner loop)
# ---------------------------------------------------------------------------

def _sliding_window_conv(
    inp: np.ndarray,
    kernel: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Perform a 'valid' 2-D cross-correlation using an explicit sliding window.

    Parameters
    ----------
    inp    : (H, W) float/int array
    kernel : (Kh, Kw) float/int array
    dtype  : numpy dtype used for the accumulator and output

    Returns
    -------
    out : (H-Kh+1, W-Kw+1) array of *dtype*
    """
    kh, kw = kernel.shape
    oh, ow = _output_shape(inp.shape, kernel.shape)
    out = np.zeros((oh, ow), dtype=dtype)
    # Flip kernel for true convolution (cross-correlation uses non-flipped)
    # Note: CNN "convolution" is actually cross-correlation — no flip.
    for i in range(oh):
        for j in range(ow):
            patch = inp[i : i + kh, j : j + kw]
            out[i, j] = np.sum(patch * kernel, dtype=dtype)
    return out


# ---------------------------------------------------------------------------
# Public API – FP32
# ---------------------------------------------------------------------------

def direct_conv_2d_fp32(
    inp: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Direct 2-D convolution (cross-correlation) in full float32 precision.

    Uses a vectorised NumPy sliding-window approach via stride tricks for
    speed while remaining dependency-free.

    Parameters
    ----------
    inp    : (H, W) array-like — single-channel input feature map.
    kernel : (Kh, Kw) array-like — convolutional filter (3×3 by default).

    Returns
    -------
    out : (H-Kh+1, W-Kw+1) float32 ndarray

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(8, 8).astype(np.float32)
    >>> k = np.ones((3, 3), dtype=np.float32) / 9
    >>> y = direct_conv_2d_fp32(x, k)
    >>> y.shape
    (6, 6)
    """
    inp = np.asarray(inp, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    _validate_inputs(inp, kernel)

    kh, kw = kernel.shape
    oh, ow = _output_shape(inp.shape, kernel.shape)

    # Build a view of shape (oh, ow, kh, kw) using stride tricks (fast path)
    s0, s1 = inp.strides
    shape = (oh, ow, kh, kw)
    strides = (s0, s1, s0, s1)
    patches = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
    out = np.einsum("ijkl,kl->ij", patches, kernel, dtype=np.float32)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API – FP16
# ---------------------------------------------------------------------------

def direct_conv_2d_fp16(
    inp: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Direct 2-D convolution in float16 precision.

    Both the input and kernel are downcast to float16 before computation.
    The result is returned as float32 so it can be compared against other
    precisions without further conversion by the caller.

    Parameters
    ----------
    inp    : (H, W) array-like
    kernel : (Kh, Kw) array-like

    Returns
    -------
    out : (H-Kh+1, W-Kw+1) float32 ndarray  (computed in fp16, upcast for return)

    Notes
    -----
    FP16 has ~3 decimal digits of precision and a dynamic range of roughly
    ±65504.  Accumulation of many FP16 products can introduce noticeable
    rounding errors especially for large filters or many channels.
    """
    inp = np.asarray(inp, dtype=np.float16)
    kernel = np.asarray(kernel, dtype=np.float16)
    _validate_inputs(inp, kernel)

    kh, kw = kernel.shape
    oh, ow = _output_shape(inp.shape, kernel.shape)

    s0, s1 = inp.strides
    shape = (oh, ow, kh, kw)
    strides = (s0, s1, s0, s1)
    patches = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
    # Accumulate in fp16
    out = np.einsum("ijkl,kl->ij", patches, kernel).astype(np.float16)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API – Simulated INT8
# ---------------------------------------------------------------------------

def direct_conv_2d_int8_simulated(
    inp: np.ndarray,
    kernel: np.ndarray,
    scale_input: float = 0.0,
    scale_kernel: float = 0.0,
) -> np.ndarray:
    """
    Simulate INT8 convolution via post-training static quantisation.

    The pipeline is:
      1. Determine per-tensor scales (127 / max|x|) if not provided.
      2. Quantise input  → int8  in [-127, 127].
      3. Quantise kernel → int8  in [-127, 127].
      4. Promote both to int32, perform multiply-accumulate to avoid overflow.
      5. Dequantise the int32 output back to float32.

    Parameters
    ----------
    inp          : (H, W) float array-like
    kernel       : (Kh, Kw) float array-like
    scale_input  : float, optional
        Quantisation scale for the input (127 / max|input|).
        If 0.0 or not provided, computed automatically.
    scale_kernel : float, optional
        Quantisation scale for the kernel (127 / max|kernel|).
        If 0.0 or not provided, computed automatically.

    Returns
    -------
    out : (H-Kh+1, W-Kw+1) float32 ndarray

    Notes
    -----
    This is a *simulation* on a CPU using integer arithmetic; it does NOT
    run on dedicated INT8 hardware.  The purpose is to study numerical
    accuracy degradation introduced by INT8 quantisation.
    """
    inp = np.asarray(inp, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    _validate_inputs(inp, kernel)

    # Auto-compute scales if not supplied
    if scale_input == 0.0:
        max_inp = np.max(np.abs(inp))
        scale_input = (127.0 / max_inp) if max_inp != 0.0 else 1.0

    if scale_kernel == 0.0:
        max_ker = np.max(np.abs(kernel))
        scale_kernel = (127.0 / max_ker) if max_ker != 0.0 else 1.0

    # Quantise
    inp_q = quantize_to_int8(inp, scale_input)   # int8
    ker_q = quantize_to_int8(kernel, scale_kernel)  # int8

    # Promote to int32 for safe multiply-accumulate
    inp_i32 = inp_q.astype(np.int32)
    ker_i32 = ker_q.astype(np.int32)

    kh, kw = kernel.shape
    oh, ow = _output_shape(inp.shape, kernel.shape)

    # Sliding window in int32
    s0, s1 = inp_i32.strides
    shape = (oh, ow, kh, kw)
    strides = (s0, s1, s0, s1)
    patches_i32 = np.lib.stride_tricks.as_strided(
        inp_i32, shape=shape, strides=strides
    )
    out_i32 = np.einsum("ijkl,kl->ij", patches_i32, ker_i32)  # int64 accumulator

    # Dequantise: combined scale = 1/(scale_input * scale_kernel)
    combined_scale = 1.0 / (scale_input * scale_kernel)
    out_f32 = out_i32.astype(np.float32) * combined_scale
    return out_f32


# ---------------------------------------------------------------------------
# Multi-channel convenience wrappers
# ---------------------------------------------------------------------------

def direct_conv_2d_multichannel(
    inp: np.ndarray,
    kernel: np.ndarray,
    precision: str = "fp32",
    **kwargs,
) -> np.ndarray:
    """
    Multi-channel 2-D convolution dispatcher.

    Loops over input channels (C) and output filters (F) and accumulates
    per-channel results.

    Parameters
    ----------
    inp       : (C, H, W) or (H, W) array — input feature map.
    kernel    : (F, C, Kh, Kw) or (Kh, Kw) array — filter bank.
    precision : one of 'fp32', 'fp16', 'int8'
    **kwargs  : passed to the underlying single-channel function.

    Returns
    -------
    out : (F, H-Kh+1, W-Kw+1) or (H-Kh+1, W-Kw+1) float32 ndarray
    """
    dispatch = {
        "fp32": direct_conv_2d_fp32,
        "fp16": direct_conv_2d_fp16,
        "int8": direct_conv_2d_int8_simulated,
    }
    if precision not in dispatch:
        raise ValueError(f"precision must be one of {list(dispatch.keys())}")
    fn = dispatch[precision]

    # Single-channel / single-filter case
    if inp.ndim == 2 and kernel.ndim == 2:
        return fn(inp, kernel, **kwargs)

    # Normalise shapes
    if inp.ndim == 2:
        inp = inp[np.newaxis]          # (1, H, W)
    if kernel.ndim == 2:
        kernel = kernel[np.newaxis, np.newaxis]   # (1, 1, Kh, Kw)

    C_in = inp.shape[0]
    F = kernel.shape[0]
    # shape of a single output map
    sample_out = fn(inp[0], kernel[0, 0], **kwargs)
    oh, ow = sample_out.shape

    out = np.zeros((F, oh, ow), dtype=np.float32)
    for f in range(F):
        for c in range(C_in):
            out[f] += fn(inp[c], kernel[f, c], **kwargs)
    return out
