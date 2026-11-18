"""
winograd_conv.py
================
Winograd F(2×2, 3×3) convolution — tile-level and full-image implementations
in FP32, FP16, and simulated INT8 precisions.

Mathematical background
-----------------------
Winograd's minimal filtering algorithm F(m×m, r×r) computes an m×m output
from an (m+r-1)×(m+r-1) input tile using only m² + (r-1)m + (r-1) = 16
multiplications instead of m² * r² = 36 for F(2×2, 3×3).

The forward transform:
    U = G · g · Gᵀ          (filter  transform, once per filter)
    V = Bᵀ · d · B          (data    transform, once per tile)
    M = U ⊙ V               (element-wise product — the only 16 muls)
    Y = Aᵀ · M · A          (inverse transform — 2×2 output)

Transform matrices for F(2×2, 3×3):
    G = [[ 1,    0,    0  ],
         [ 1/2,  1/2,  1/2],
         [ 1/2, -1/2,  1/2],
         [ 0,    0,    1  ]]

    B = [[ 1,  0, -1,  0],
         [ 0,  1,  1,  0],
         [ 0, -1,  1,  0],
         [ 0,  1,  0, -1]]

    A = [[1,  0],
         [1,  1],
         [1, -1],
         [0, -1]]

References
----------
[1] Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks",
    CVPR 2016.
[2] Mori et al., "Wino Vidi Vici: Conquering Numerical Instability of
    8-bit Winograd Convolution", WACV 2024.

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple

from .precision_utils import (
    quantize_to_int8,
    quantize_with_clip,
    dequantize_from_int8,
    simulate_fp16_rounding,
)

# ---------------------------------------------------------------------------
# F(2×2, 3×3) transform matrices — defined in float64 for precision
# ---------------------------------------------------------------------------

G = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.5,  0.5,  0.5],
        [0.5, -0.5,  0.5],
        [0.0,  0.0,  1.0],
    ],
    dtype=np.float64,
)

Gt = G.T  # 3×4

B = np.array(
    [
        [1.0,  0.0, -1.0,  0.0],
        [0.0,  1.0,  1.0,  0.0],
        [0.0, -1.0,  1.0,  0.0],
        [0.0,  1.0,  0.0, -1.0],
    ],
    dtype=np.float64,
)

Bt = B.T  # 4×4 (same as B when you work it out, but let's be explicit)

A = np.array(
    [
        [1.0,  0.0],
        [1.0,  1.0],
        [1.0, -1.0],
        [0.0, -1.0],
    ],
    dtype=np.float64,
)

At = A.T  # 2×4


# ---------------------------------------------------------------------------
# Tile-level Winograd — single (4×4 input, 3×3 filter) → 2×2 output
# ---------------------------------------------------------------------------

def winograd_conv_tile_fp32(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Single-tile Winograd F(2×2, 3×3) convolution in float32.

    Parameters
    ----------
    d : (4, 4) float array — input tile
    g : (3, 3) float array — convolutional filter

    Returns
    -------
    Y : (2, 2) float32 ndarray — output tile

    Raises
    ------
    ValueError : if d is not 4×4 or g is not 3×3.

    Examples
    --------
    >>> import numpy as np
    >>> d = np.random.randn(4, 4).astype(np.float32)
    >>> g = np.random.randn(3, 3).astype(np.float32)
    >>> Y = winograd_conv_tile_fp32(d, g)
    >>> Y.shape
    (2, 2)
    """
    d = np.asarray(d, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if d.shape != (4, 4):
        raise ValueError(f"Input tile must be 4×4; got {d.shape}.")
    if g.shape != (3, 3):
        raise ValueError(f"Filter must be 3×3; got {g.shape}.")

    G_f32 = G.astype(np.float32)
    Gt_f32 = Gt.astype(np.float32)
    B_f32 = B.astype(np.float32)
    Bt_f32 = Bt.astype(np.float32)
    A_f32 = A.astype(np.float32)
    At_f32 = At.astype(np.float32)

    U = G_f32 @ g @ Gt_f32          # 4×4 filter transform
    V = B_f32 @ d @ Bt_f32          # 4×4 data   transform  (Bᵀ·d·B)
    M = U * V                       # 4×4 element-wise product — 16 muls
    Y = At_f32 @ M @ A_f32          # 2×2 output (Aᵀ·M·A)
    return Y


def winograd_conv_tile_fp64(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    High-precision float64 version — used internally for correctness checks.
    """
    d = np.asarray(d, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    U = G @ g @ Gt
    V = B @ d @ Bt        # Note: Bᵀ applied as matrix B (B already rows of Bᵀ)
    M = U * V
    Y = At @ M @ A
    return Y


# ---------------------------------------------------------------------------
# Internal: apply the correct Winograd data transform
# ---------------------------------------------------------------------------

def _winograd_data_transform(d: np.ndarray) -> np.ndarray:
    """Return Bᵀ · d · B for a (4,4) tile (float64 ops)."""
    return B @ d @ Bt


def _winograd_filter_transform(g: np.ndarray) -> np.ndarray:
    """Return G · g · Gᵀ for a (3,3) filter (float64 ops)."""
    return G @ g @ Gt


def _winograd_inverse_transform(M: np.ndarray) -> np.ndarray:
    """Return Aᵀ · M · A for a (4,4) matrix (float64 ops)."""
    return At @ M @ A


# ---------------------------------------------------------------------------
# Full-image Winograd — tiling logic
# ---------------------------------------------------------------------------

def _tile_input(inp: np.ndarray, tile_size: int = 4, stride: int = 2) -> Tuple[np.ndarray, int, int]:
    """
    Pad and tile a 2-D input into overlapping 4×4 tiles with stride 2.

    For an H×W input with 3×3 kernel:
      • Output size (valid) = (H-2) × (W-2)
      • Tiles needed: ceil((H-2)/2) × ceil((W-2)/2)
      • Pad input to (2 + n_tiles_h*2) × (2 + n_tiles_w*2)

    Returns
    -------
    padded  : padded 2-D array
    n_th    : number of tiles in the vertical direction
    n_tw    : number of tiles in the horizontal direction
    """
    H, W = inp.shape
    out_h = H - 2  # valid output height for 3×3 kernel
    out_w = W - 2

    n_th = math.ceil(out_h / stride)   # number of 2×2 output patches vertically
    n_tw = math.ceil(out_w / stride)

    # Required padded size
    pad_h = n_th * stride + 2   # = n_th*2 + 2 (the extra 2 for the kernel overlap)
    pad_w = n_tw * stride + 2

    # Pad with zeros on the bottom/right if necessary
    padded = np.zeros((pad_h, pad_w), dtype=inp.dtype)
    padded[:H, :W] = inp
    return padded, n_th, n_tw


def winograd_conv_2d_fp32(
    inp: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Full-image Winograd F(2×2, 3×3) convolution in float32.

    The image is tiled into overlapping 4×4 patches with stride 2.
    Each tile produces a 2×2 output patch.  The patches are assembled
    into the final output of size (H-2) × (W-2).

    Parameters
    ----------
    inp    : (H, W) float array-like — single-channel input (H,W ≥ 4)
    kernel : (3, 3) float array-like — convolutional filter

    Returns
    -------
    out : (H-2, W-2) float32 ndarray

    Notes
    -----
    For images where H or W is not of the form 2+2k, the input is
    zero-padded on the right/bottom to fit an integer number of tiles.
    The output is then cropped to the correct (H-2)×(W-2) size.
    """
    inp = np.asarray(inp, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    if kernel.shape != (3, 3):
        raise ValueError(f"Winograd F(2×2,3×3) requires a 3×3 kernel; got {kernel.shape}.")
    if inp.ndim != 2:
        raise ValueError(f"Input must be 2-D; got shape {inp.shape}.")
    if inp.shape[0] < 4 or inp.shape[1] < 4:
        raise ValueError(f"Input must be at least 4×4 for Winograd tiling; got {inp.shape}.")

    H, W = inp.shape
    out_h = H - 2
    out_w = W - 2

    # Pre-compute filter transform (done once)
    G_f32  = G.astype(np.float32)
    Gt_f32 = Gt.astype(np.float32)
    B_f32  = B.astype(np.float32)
    Bt_f32 = Bt.astype(np.float32)
    A_f32  = A.astype(np.float32)
    At_f32 = At.astype(np.float32)

    U = G_f32 @ kernel @ Gt_f32   # 4×4

    # Tile the input
    padded, n_th, n_tw = _tile_input(inp.astype(np.float32))
    out_full = np.zeros((n_th * 2, n_tw * 2), dtype=np.float32)

    for ti in range(n_th):
        for tj in range(n_tw):
            row = ti * 2
            col = tj * 2
            d = padded[row : row + 4, col : col + 4]
            # Data transform
            V = B_f32 @ d @ Bt_f32
            # Element-wise product
            M = U * V
            # Inverse transform → 2×2 output patch
            Y = At_f32 @ M @ A_f32
            out_full[ti * 2 : ti * 2 + 2, tj * 2 : tj * 2 + 2] = Y

    # Crop to valid output size
    return out_full[:out_h, :out_w]


# ---------------------------------------------------------------------------
# FP16 version
# ---------------------------------------------------------------------------

def winograd_conv_2d_fp16(
    inp: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Full-image Winograd F(2×2, 3×3) convolution in float16.

    Both transforms and the element-wise product are performed in fp16.
    The result is returned as float32.

    Parameters
    ----------
    inp    : (H, W) array-like
    kernel : (3, 3) array-like

    Returns
    -------
    out : (H-2, W-2) float32 ndarray (computed in fp16)

    Notes
    -----
    FP16 Winograd can suffer from numerical instability because the
    transform matrices introduce intermediate values that amplify
    rounding errors.  This is the central theme studied in Paper [2].
    """
    inp = np.asarray(inp, dtype=np.float16)
    kernel = np.asarray(kernel, dtype=np.float16)

    if kernel.shape != (3, 3):
        raise ValueError(f"Winograd F(2×2,3×3) requires a 3×3 kernel; got {kernel.shape}.")
    if inp.shape[0] < 4 or inp.shape[1] < 4:
        raise ValueError(f"Input must be at least 4×4; got {inp.shape}.")

    H, W = inp.shape
    out_h = H - 2
    out_w = W - 2

    G_f16  = G.astype(np.float16)
    Gt_f16 = Gt.astype(np.float16)
    B_f16  = B.astype(np.float16)
    Bt_f16 = Bt.astype(np.float16)
    A_f16  = A.astype(np.float16)
    At_f16 = At.astype(np.float16)

    U = G_f16 @ kernel @ Gt_f16

    padded, n_th, n_tw = _tile_input(inp)
    out_full = np.zeros((n_th * 2, n_tw * 2), dtype=np.float16)

    for ti in range(n_th):
        for tj in range(n_tw):
            row, col = ti * 2, tj * 2
            d = padded[row : row + 4, col : col + 4]
            V = B_f16 @ d @ Bt_f16
            M = U * V
            Y = At_f16 @ M @ A_f16
            out_full[ti * 2 : ti * 2 + 2, tj * 2 : tj * 2 + 2] = Y

    return out_full[:out_h, :out_w].astype(np.float32)


# ---------------------------------------------------------------------------
# Simulated INT8 version
# ---------------------------------------------------------------------------

def winograd_conv_2d_int8_simulated(
    inp: np.ndarray,
    kernel: np.ndarray,
    scale_input: float = 0.0,
    scale_kernel: float = 0.0,
) -> np.ndarray:
    """
    Full-image Winograd F(2×2, 3×3) convolution with simulated INT8.

    Strategy:
      1. Quantise input and kernel to int8.
      2. Dequantise back to float32 (this is the "fake-quantisation" step
         that models the rounding introduced by INT8 storage).
      3. Run the standard FP32 Winograd pipeline on the dequantised values.

    This models the accuracy loss from weight/activation quantisation
    without attempting to simulate integer-domain Winograd transforms
    (which would require re-quantising intermediate values).

    Parameters
    ----------
    inp          : (H, W) float array-like
    kernel       : (3, 3) float array-like
    scale_input  : float — quantisation scale for input  (auto if 0)
    scale_kernel : float — quantisation scale for kernel (auto if 0)

    Returns
    -------
    out : (H-2, W-2) float32 ndarray
    """
    inp = np.asarray(inp, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    if scale_input == 0.0:
        max_inp = np.max(np.abs(inp))
        scale_input = (127.0 / max_inp) if max_inp != 0.0 else 1.0

    if scale_kernel == 0.0:
        max_ker = np.max(np.abs(kernel))
        scale_kernel = (127.0 / max_ker) if max_ker != 0.0 else 1.0

    # Fake-quantise: round-trip through int8
    inp_dq = dequantize_from_int8(quantize_to_int8(inp, scale_input), scale_input)
    ker_dq = dequantize_from_int8(quantize_to_int8(kernel, scale_kernel), scale_kernel)

    # Run FP32 Winograd on the quantised (dequantised) values
    return winograd_conv_2d_fp32(inp_dq, ker_dq)


def winograd_conv_2d_int8_stagewise(
    inp: np.ndarray,
    kernel: np.ndarray,
    clip_factor: float = 1.0,
) -> np.ndarray:
    """
    Simulate stage-wise transform-domain INT8 requantization.

    Pipeline used here:
      1. Transform the kernel and each 4×4 tile with G and B.
      2. Clip each transformed tensor using a max-derived, per-tile bound.
      3. Quantise the transformed kernel and data with INT8 rounding.
      4. Dequantise back to float32.
      5. Multiply element-wise in the transformed domain.
      6. Clip the element-wise product before its own quantisation round-trip.
      7. Apply the inverse transform to recover the output patch.

    This heuristic is inspired by transform-domain stability work but does
    not reproduce learned clipping, calibration, or quantization-aware
    training from Wino Vidi Vici.
    """
    inp = np.asarray(inp, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if clip_factor <= 0.0:
        raise ValueError("clip_factor must be positive")

    if kernel.shape != (3, 3):
        raise ValueError(f"Winograd F(2×2,3×3) requires a 3×3 kernel; got {kernel.shape}.")
    if inp.ndim != 2:
        raise ValueError(f"Input must be 2-D; got shape {inp.shape}.")
    if inp.shape[0] < 4 or inp.shape[1] < 4:
        raise ValueError(f"Input must be at least 4×4 for Winograd tiling; got {inp.shape}.")

    H, W = inp.shape
    out_h = H - 2
    out_w = W - 2

    G_f32 = G.astype(np.float32)
    Gt_f32 = Gt.astype(np.float32)
    B_f32 = B.astype(np.float32)
    Bt_f32 = Bt.astype(np.float32)
    A_f32 = A.astype(np.float32)
    At_f32 = At.astype(np.float32)

    kernel_transform = G_f32 @ kernel @ Gt_f32
    kernel_clip = max(np.max(np.abs(kernel_transform)), 1e-6) * clip_factor

    padded, n_th, n_tw = _tile_input(inp.astype(np.float32))
    out_full = np.zeros((n_th * 2, n_tw * 2), dtype=np.float32)

    for ti in range(n_th):
        for tj in range(n_tw):
            row, col = ti * 2, tj * 2
            d = padded[row : row + 4, col : col + 4]
            data_transform = B_f32 @ d @ Bt_f32
            data_clip = max(np.max(np.abs(data_transform)), 1e-6) * clip_factor

            # Stage 1: quantise the transformed filter and data with clipping.
            U_q, scale_u, _ = quantize_with_clip(kernel_transform, kernel_clip)
            V_q, scale_v, _ = quantize_with_clip(data_transform, data_clip)
            U_dq = dequantize_from_int8(U_q, scale_u).astype(np.float32)
            V_dq = dequantize_from_int8(V_q, scale_v).astype(np.float32)

            M = U_dq * V_dq
            m_clip = max(np.max(np.abs(M)), 1e-6) * clip_factor
            M_q, scale_m, _ = quantize_with_clip(M, m_clip)
            M_dq = dequantize_from_int8(M_q, scale_m).astype(np.float32)

            Y = At_f32 @ M_dq @ A_f32
            out_full[ti * 2 : ti * 2 + 2, tj * 2 : tj * 2 + 2] = Y

    return out_full[:out_h, :out_w].astype(np.float32)


# ---------------------------------------------------------------------------
# Correctness verification helper
# ---------------------------------------------------------------------------

def verify_winograd_vs_direct(
    inp: np.ndarray,
    kernel: np.ndarray,
    tol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """
    Verify that Winograd FP32 and direct FP32 produce the same result.

    Parameters
    ----------
    inp     : (H, W) input array (H,W ≥ 4)
    kernel  : (3, 3) filter
    tol     : maximum allowed absolute difference (default 1e-4)
    verbose : if True, print the max absolute error

    Returns
    -------
    bool — True if max error ≤ tol, False otherwise
    """
    from .direct_conv import direct_conv_2d_fp32
    from .precision_utils import compute_max_abs_error

    ref = direct_conv_2d_fp32(inp, kernel)
    win = winograd_conv_2d_fp32(inp, kernel)

    # Both produce (H-2)×(W-2); may differ by 1 pixel if tiling pads
    min_h = min(ref.shape[0], win.shape[0])
    min_w = min(ref.shape[1], win.shape[1])
    err = compute_max_abs_error(ref[:min_h, :min_w], win[:min_h, :min_w])

    if verbose:
        print(f"[verify] Max absolute error (Winograd FP32 vs Direct FP32): {err:.2e}")
        status = "✓ PASS" if err <= tol else "✗ FAIL"
        print(f"         Tolerance: {tol:.2e}  →  {status}")

    return err <= tol
