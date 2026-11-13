"""
experiment.py
=============
Main experiment runner for the Winograd Reduced-Precision study.

Workflow
--------
1. For each input size in [8, 16, 32, 64, 128]:
   a. Generate a random input and 3×3 kernel (fixed seed for reproducibility).
   b. Compute the FP32 direct convolution as the numerical ground truth.
   c. Run all six method/precision combinations:
        direct_fp32, direct_fp16, direct_int8,
        winograd_fp32, winograd_fp16, winograd_int8
   d. Compute numerical errors vs. the ground truth.
   e. Measure runtime.
2. Collect all results in a list of dicts → pandas DataFrame (or plain CSV).
3. Save to results/results.csv.
4. Print a formatted summary table.

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import os
import csv
import time
import numpy as np
from typing import List, Dict, Any

from tqdm import tqdm

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
from .precision_utils import (
    compute_max_abs_error,
    compute_mse,
    compute_snr_db,
)
from .benchmark import time_function

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZES  = [8, 16, 32, 64, 128]
KERNEL_SIZE  = 3
N_RUNS       = 30     # timing iterations per combination
WARMUP_RUNS  = 5
RANDOM_SEED  = 42
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

CSV_FIELDNAMES = [
    "input_size",
    "method",
    "precision",
    "max_abs_error",
    "mse",
    "snr_db",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "median_ms",
]


# ---------------------------------------------------------------------------
# Helper: safe error computation (handles shape mismatches from tiling)
# ---------------------------------------------------------------------------

def _crop_to_min(a: np.ndarray, b: np.ndarray):
    """Crop both arrays to their overlapping valid region."""
    min_h = min(a.shape[0], b.shape[0])
    min_w = min(a.shape[1], b.shape[1])
    return a[:min_h, :min_w], b[:min_h, :min_w]


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    input_sizes: List[int] = None,
    n_runs: int = N_RUNS,
    seed: int = RANDOM_SEED,
    save_csv: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the full numerical and performance study.

    For each input size and method/precision combination:
      • Compute the output.
      • Measure numerical errors vs. the FP32 direct baseline.
      • Measure wall-clock runtime.

    Parameters
    ----------
    input_sizes : list of int — spatial sizes (H=W) to sweep (default INPUT_SIZES)
    n_runs      : int — timing iterations per combination (default 30)
    seed        : int — random seed (default 42)
    save_csv    : bool — write results to results/results.csv (default True)
    verbose     : bool — print progress and summary table (default True)

    Returns
    -------
    list of dicts, one per (size, method, precision) combination.
    Each dict has the keys listed in CSV_FIELDNAMES.
    """
    if input_sizes is None:
        input_sizes = INPUT_SIZES

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    # Pre-generate random inputs and kernels for each size (reproducible)
    test_cases = {
        sz: (
            rng.standard_normal((sz, sz)).astype(np.float32),
            rng.standard_normal((KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32),
        )
        for sz in input_sizes
    }

    # Method registry: (method_name, precision, callable)
    methods = [
        ("direct",   "fp32", direct_conv_2d_fp32),
        ("direct",   "fp16", direct_conv_2d_fp16),
        ("direct",   "int8", direct_conv_2d_int8_simulated),
        ("winograd", "fp32", winograd_conv_2d_fp32),
        ("winograd", "fp16", winograd_conv_2d_fp16),
        ("winograd", "int8", winograd_conv_2d_int8_simulated),
    ]

    total = len(input_sizes) * len(methods)
    pbar = tqdm(total=total, desc="Running experiments", unit="combo") if verbose else None

    for sz in input_sizes:
        inp, ker = test_cases[sz]

        # Ground truth — FP32 direct convolution
        ref = direct_conv_2d_fp32(inp, ker)

        for method_name, precision, fn in methods:
            if pbar:
                pbar.set_postfix(size=sz, method=f"{method_name}_{precision}")

            row: Dict[str, Any] = {
                "input_size": sz,
                "method":     method_name,
                "precision":  precision,
            }

            # --- Numerical errors ---
            try:
                out = fn(inp, ker)
                ref_c, out_c = _crop_to_min(ref, out)
                row["max_abs_error"] = compute_max_abs_error(ref_c, out_c)
                row["mse"]           = compute_mse(ref_c, out_c)
                row["snr_db"]        = compute_snr_db(ref_c, out_c)
            except Exception as e:  # noqa: BLE001
                row["max_abs_error"] = float("nan")
                row["mse"]           = float("nan")
                row["snr_db"]        = float("nan")
                if verbose:
                    print(f"[Error] {method_name}_{precision} @ {sz}: {e}")

            # --- Runtime ---
            try:
                stats = time_function(fn, inp, ker, n_runs=n_runs, warmup=WARMUP_RUNS)
                row.update(stats)
            except Exception as e:  # noqa: BLE001
                for k in ("mean_ms", "std_ms", "min_ms", "max_ms", "median_ms"):
                    row[k] = float("nan")
                if verbose:
                    print(f"[Timing Error] {method_name}_{precision} @ {sz}: {e}")

            all_rows.append(row)
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # --- Save CSV ---
    if save_csv:
        _save_csv(all_rows, RESULTS_FILE)
        if verbose:
            print(f"\n[✓] Results saved to: {os.path.abspath(RESULTS_FILE)}")

    # --- Print summary table ---
    if verbose:
        _print_summary(all_rows)

    return all_rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of result dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            # Ensure all fields exist (fill missing with nan)
            clean = {k: row.get(k, float("nan")) for k in CSV_FIELDNAMES}
            writer.writerow(clean)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table using tabulate (or fallback)."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("[Warning] 'tabulate' not installed; skipping pretty table.")
        for r in rows:
            print(r)
        return

    headers = ["Size", "Method", "Precision", "MaxAbsErr", "MSE", "SNR(dB)", "Mean(ms)"]
    table = []
    for r in rows:
        table.append([
            r.get("input_size", "?"),
            r.get("method", "?"),
            r.get("precision", "?"),
            f"{r.get('max_abs_error', float('nan')):.4e}",
            f"{r.get('mse', float('nan')):.4e}",
            f"{r.get('snr_db', float('nan')):.2f}",
            f"{r.get('mean_ms', float('nan')):.4f}",
        ])

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("=" * 70)
