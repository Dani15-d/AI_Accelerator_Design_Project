"""
benchmark.py
============
Benchmarking utilities for the Winograd Reduced-Precision study.

Provides:
  - ``time_function``        — generic timing harness with warmup
  - ``benchmark_convolution`` — benchmark all method/precision combos
                                 for a given input shape

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import time
import numpy as np
from typing import Callable, Dict, List, Any, Tuple

from .direct_conv import (
    direct_conv_2d_fp32,
    direct_conv_2d_fp16,
    direct_conv_2d_int8_simulated,
)
from .winograd_conv import (
    winograd_conv_2d_fp32,
    winograd_conv_2d_fp16,
    winograd_conv_2d_int8_simulated,
)


# ---------------------------------------------------------------------------
# Generic timing harness
# ---------------------------------------------------------------------------

def time_function(
    fn: Callable,
    *args,
    n_runs: int = 100,
    warmup: int = 5,
    **kwargs,
) -> Dict[str, float]:
    """
    Time a callable with warmup, returning statistics in milliseconds.

    Parameters
    ----------
    fn      : callable — the function to time
    *args   : positional arguments forwarded to *fn*
    n_runs  : int — number of timed iterations after warmup (default 100)
    warmup  : int — number of warmup calls (not timed, default 5)
    **kwargs: keyword arguments forwarded to *fn*

    Returns
    -------
    dict with keys:
        'mean_ms'  : float — arithmetic mean of run times in ms
        'std_ms'   : float — standard deviation in ms
        'min_ms'   : float — minimum run time in ms
        'max_ms'   : float — maximum run time in ms
        'median_ms': float — median run time in ms

    Notes
    -----
    Uses ``time.perf_counter`` for sub-millisecond resolution.
    On Windows, ``time.perf_counter`` typically has ~100 ns resolution.
    """
    # Warmup phase
    for _ in range(warmup):
        fn(*args, **kwargs)

    # Timed phase
    times_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1_000.0)

    times_arr = np.array(times_ms)
    return {
        "mean_ms":   float(np.mean(times_arr)),
        "std_ms":    float(np.std(times_arr)),
        "min_ms":    float(np.min(times_arr)),
        "max_ms":    float(np.max(times_arr)),
        "median_ms": float(np.median(times_arr)),
    }


# ---------------------------------------------------------------------------
# Convolution benchmark
# ---------------------------------------------------------------------------

def benchmark_convolution(
    input_shape: Tuple[int, int],
    kernel_size: int = 3,
    n_runs: int = 50,
    warmup: int = 5,
    precisions: List[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmark all direct and Winograd convolution methods for a given shape.

    Parameters
    ----------
    input_shape  : (H, W) tuple — spatial dimensions of the input feature map
    kernel_size  : int — filter size (must be 3 for Winograd; default 3)
    n_runs       : int — number of timing iterations (default 50)
    warmup       : int — warmup calls before timing (default 5)
    precisions   : list of str — subset of ['fp32','fp16','int8']
                   (default: all three)
    seed         : int — random seed for reproducible inputs

    Returns
    -------
    dict mapping '<method>_<precision>' → timing stats dict
    (see ``time_function`` for the structure of each stats dict)

    The dict also contains:
        'input_shape' : (H, W)
        'kernel_size' : int

    Raises
    ------
    ValueError : if kernel_size != 3 (Winograd is F(2×2, 3×3) only).
    """
    if kernel_size != 3:
        raise ValueError(
            f"Winograd F(2×2,3×3) only supports 3×3 kernels; got {kernel_size}."
        )
    if precisions is None:
        precisions = ["fp32", "fp16", "int8"]

    rng = np.random.default_rng(seed)
    H, W = input_shape
    inp = rng.standard_normal((H, W)).astype(np.float32)
    ker = rng.standard_normal((kernel_size, kernel_size)).astype(np.float32)

    results: Dict[str, Any] = {
        "input_shape": input_shape,
        "kernel_size": kernel_size,
    }

    # --- Direct convolution ---
    method_map_direct = {
        "fp32": direct_conv_2d_fp32,
        "fp16": direct_conv_2d_fp16,
        "int8": direct_conv_2d_int8_simulated,
    }
    for prec in precisions:
        if prec not in method_map_direct:
            continue
        fn = method_map_direct[prec]
        key = f"direct_{prec}"
        try:
            stats = time_function(fn, inp, ker, n_runs=n_runs, warmup=warmup)
            results[key] = stats
        except Exception as exc:  # noqa: BLE001
            results[key] = {"error": str(exc)}

    # --- Winograd convolution ---
    # Winograd requires at least 4×4 input
    if H >= 4 and W >= 4:
        method_map_wino = {
            "fp32": winograd_conv_2d_fp32,
            "fp16": winograd_conv_2d_fp16,
            "int8": winograd_conv_2d_int8_simulated,
        }
        for prec in precisions:
            if prec not in method_map_wino:
                continue
            fn = method_map_wino[prec]
            key = f"winograd_{prec}"
            try:
                stats = time_function(fn, inp, ker, n_runs=n_runs, warmup=warmup)
                results[key] = stats
            except Exception as exc:  # noqa: BLE001
                results[key] = {"error": str(exc)}
    else:
        for prec in precisions:
            results[f"winograd_{prec}"] = {
                "error": f"Input {input_shape} too small for Winograd (need ≥ 4×4)."
            }

    return results


def print_benchmark_table(results: Dict[str, Any]) -> None:
    """
    Pretty-print a benchmark result dict produced by ``benchmark_convolution``.

    Parameters
    ----------
    results : dict returned by ``benchmark_convolution``
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("[Warning] 'tabulate' not installed; printing raw dict.")
        print(results)
        return

    rows = []
    for key, val in results.items():
        if key in ("input_shape", "kernel_size"):
            continue
        if isinstance(val, dict) and "mean_ms" in val:
            rows.append(
                [
                    key,
                    f"{val['mean_ms']:.4f}",
                    f"{val['std_ms']:.4f}",
                    f"{val['min_ms']:.4f}",
                    f"{val['max_ms']:.4f}",
                ]
            )
        elif isinstance(val, dict) and "error" in val:
            rows.append([key, "ERROR", val["error"], "", ""])

    H, W = results.get("input_shape", ("?", "?"))
    print(f"\n=== Benchmark: input {H}×{W}, kernel 3×3 ===")
    print(
        tabulate(
            rows,
            headers=["Method+Precision", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)"],
            tablefmt="github",
        )
    )
