"""
precision_utils.py
==================
Helper utilities for quantisation, dequantisation, and numerical error
metrics used throughout the Winograd Reduced-Precision study.

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantize_to_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Quantise a float32 array to the INT8 range [-127, 127].

    The forward mapping is:
        q = clip( round(arr * scale), -127, 127 )

    Parameters
    ----------
    arr   : float32 ndarray of arbitrary shape
    scale : float — multiply factor (typically 127 / max|arr|)

    Returns
    -------
    q : int8 ndarray, same shape as *arr*

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.5, -0.5, 1.0])
    >>> quantize_to_int8(a, scale=127.0)
    array([ 63, -63, 127], dtype=int8)
    """
    arr = np.asarray(arr, dtype=np.float32)
    q = np.round(arr * scale).astype(np.int32)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


def dequantize_from_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Dequantise an INT8 array back to float32.

    The reverse mapping is:
        x_approx = arr / scale

    Parameters
    ----------
    arr   : int8 ndarray (or any integer dtype)
    scale : float — the same scale used during quantisation

    Returns
    -------
    x_approx : float32 ndarray, same shape as *arr*

    Notes
    -----
    Due to rounding in the forward pass, ``dequantize(quantize(x, s), s)``
    is only an approximation of ``x``.
    """
    arr = np.asarray(arr, dtype=np.float32)
    return (arr / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# FP16 rounding simulation
# ---------------------------------------------------------------------------

def simulate_fp16_rounding(arr: np.ndarray) -> np.ndarray:
    """
    Simulate float16 rounding by casting to fp16 and back to fp32.

    This introduces the same mantissa truncation / rounding that would
    occur on hardware that stores values in fp16.

    Parameters
    ----------
    arr : float32 ndarray

    Returns
    -------
    arr_rounded : float32 ndarray with fp16 numerical precision

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0 / 3.0], dtype=np.float32)
    >>> simulate_fp16_rounding(a)
    array([0.3333], dtype=float32)
    """
    arr = np.asarray(arr, dtype=np.float32)
    return arr.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_max_abs_error(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute the maximum absolute element-wise error between two arrays.

    max_error = max | ref - test |

    Parameters
    ----------
    ref  : reference (ground-truth) ndarray
    test : test (approximate) ndarray, must be broadcast-compatible with *ref*

    Returns
    -------
    float — maximum absolute error (scalar)

    Raises
    ------
    ValueError : if the shapes are incompatible.
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref.shape}, test={test.shape}."
        )
    return float(np.max(np.abs(ref - test)))


def compute_mse(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between two arrays.

    MSE = mean( (ref - test)^2 )

    Parameters
    ----------
    ref  : reference ndarray
    test : test ndarray, must be same shape as *ref*

    Returns
    -------
    float — MSE (scalar)
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref.shape}, test={test.shape}."
        )
    return float(np.mean((ref - test) ** 2))


def compute_snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    SNR_dB = 10 * log10( power(ref) / power(ref - test) )

    A higher SNR means the approximation is closer to the reference.
    Returns +inf if the error is exactly zero.

    Parameters
    ----------
    ref  : reference ndarray
    test : test ndarray, must be same shape as *ref*

    Returns
    -------
    float — SNR in dB
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref.shape}, test={test.shape}."
        )
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean((ref - test) ** 2)
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


# ---------------------------------------------------------------------------
# Additional utility
# ---------------------------------------------------------------------------

def compute_relative_error(ref: np.ndarray, test: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute mean relative absolute error.

    rel_error = mean( |ref - test| / (|ref| + eps) )

    Parameters
    ----------
    ref  : reference ndarray
    test : test ndarray
    eps  : small constant to prevent division by zero

    Returns
    -------
    float — mean relative error (scalar)
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref.shape}, test={test.shape}."
        )
    return float(np.mean(np.abs(ref - test) / (np.abs(ref) + eps)))
