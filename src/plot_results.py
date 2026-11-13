"""
plot_results.py
===============
Plotting module for the Winograd Reduced-Precision study.

Reads ``results/results.csv`` and generates four publication-quality
plots, saved to the same directory:

  1. error_vs_method.png          — bar chart: max absolute error per combo
  2. mse_vs_method.png            — bar chart: MSE per combo
  3. runtime_vs_inputsize.png     — line chart: mean runtime vs input size
  4. error_vs_runtime_tradeoff.png — scatter: max error vs mean runtime

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from __future__ import annotations

import os
import csv
import math
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

# Colour palette for the six method/precision combos
COMBO_COLORS = {
    "direct_fp32":    "#2196F3",   # blue
    "direct_fp16":    "#64B5F6",   # light blue
    "direct_int8":    "#0D47A1",   # dark blue
    "winograd_fp32":  "#F44336",   # red
    "winograd_fp16":  "#EF9A9A",   # light red
    "winograd_int8":  "#B71C1C",   # dark red
}

COMBO_MARKERS = {
    "direct_fp32":    "o",
    "direct_fp16":    "s",
    "direct_int8":    "^",
    "winograd_fp32":  "D",
    "winograd_fp16":  "P",
    "winograd_int8":  "X",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    """
    Load results from a CSV file produced by experiment.py.

    Parameters
    ----------
    csv_path : str — path to the CSV file

    Returns
    -------
    list of dicts with numeric fields converted to float
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            "Run 'python main.py' first to generate it."
        )

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _make_combo_key(row: Dict[str, Any]) -> str:
    return f"{row['method']}_{row['precision']}"


def _group_by_combo(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        key = _make_combo_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _group_by_size(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict]]:
    groups: Dict[int, List[Dict]] = {}
    for row in rows:
        sz = int(row["input_size"])
        groups.setdefault(sz, []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Plot 1: Max absolute error per method/precision combo (bar chart)
# ---------------------------------------------------------------------------

def plot_error_vs_method(
    rows: List[Dict[str, Any]],
    out_path: Optional[str] = None,
    aggregation_size: int = 64,
) -> str:
    """
    Bar chart of maximum absolute error per method/precision combination.

    Uses data from a single representative input size (default 64×64).

    Parameters
    ----------
    rows           : list of result dicts
    out_path       : output file path (auto-computed if None)
    aggregation_size : which input size to show (default 64)

    Returns
    -------
    str — absolute path to the saved figure
    """
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "error_vs_method.png")

    # Filter to the chosen size
    subset = [r for r in rows if int(r["input_size"]) == aggregation_size]
    if not subset:
        # Fallback: use the largest available size
        sizes = sorted({int(r["input_size"]) for r in rows})
        aggregation_size = sizes[-1]
        subset = [r for r in rows if int(r["input_size"]) == aggregation_size]

    combos = [_make_combo_key(r) for r in subset]
    errors = [r["max_abs_error"] for r in subset]

    colors = [COMBO_COLORS.get(c, "#888888") for c in combos]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(combos))
    bars = ax.bar(x, errors, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Max Absolute Error (log scale)", fontsize=11)
    ax.set_xlabel("Method + Precision", fontsize=11)
    ax.set_title(
        f"Max Absolute Error vs. Method/Precision  (input size {aggregation_size}×{aggregation_size})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars
    for bar, err in zip(bars, errors):
        if math.isfinite(err) and err > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.5,
                f"{err:.2e}",
                ha="center", va="bottom", fontsize=8, rotation=0,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: MSE per method/precision combo (bar chart)
# ---------------------------------------------------------------------------

def plot_mse_vs_method(
    rows: List[Dict[str, Any]],
    out_path: Optional[str] = None,
    aggregation_size: int = 64,
) -> str:
    """
    Bar chart of Mean Squared Error per method/precision combination.

    Parameters
    ----------
    rows           : list of result dicts
    out_path       : output file path (auto-computed if None)
    aggregation_size : which input size to show (default 64)

    Returns
    -------
    str — absolute path to the saved figure
    """
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "mse_vs_method.png")

    subset = [r for r in rows if int(r["input_size"]) == aggregation_size]
    if not subset:
        sizes = sorted({int(r["input_size"]) for r in rows})
        aggregation_size = sizes[-1]
        subset = [r for r in rows if int(r["input_size"]) == aggregation_size]

    combos = [_make_combo_key(r) for r in subset]
    mses   = [r["mse"] for r in subset]
    colors = [COMBO_COLORS.get(c, "#888888") for c in combos]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(combos))
    bars = ax.bar(x, mses, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Squared Error (log scale)", fontsize=11)
    ax.set_xlabel("Method + Precision", fontsize=11)
    ax.set_title(
        f"MSE vs. Method/Precision  (input size {aggregation_size}×{aggregation_size})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, mse in zip(bars, mses):
        if math.isfinite(mse) and mse > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.5,
                f"{mse:.2e}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 3: Runtime vs input size (line chart)
# ---------------------------------------------------------------------------

def plot_runtime_vs_inputsize(
    rows: List[Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """
    Line chart of mean runtime (ms) vs input spatial size for each combo.

    Parameters
    ----------
    rows     : list of result dicts
    out_path : output file path (auto-computed if None)

    Returns
    -------
    str — absolute path to the saved figure
    """
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "runtime_vs_inputsize.png")

    groups = _group_by_combo(rows)
    sizes_all = sorted({int(r["input_size"]) for r in rows})

    fig, ax = plt.subplots(figsize=(10, 6))

    for combo, combo_rows in sorted(groups.items()):
        size_to_time = {}
        for r in combo_rows:
            sz = int(r["input_size"])
            t  = r.get("mean_ms", float("nan"))
            if math.isfinite(t):
                size_to_time[sz] = t

        xs = sorted(size_to_time.keys())
        ys = [size_to_time[s] for s in xs]
        if not xs:
            continue

        color  = COMBO_COLORS.get(combo, "#888888")
        marker = COMBO_MARKERS.get(combo, "o")
        ax.plot(xs, ys, marker=marker, color=color, label=combo,
                linewidth=1.8, markersize=7)

    ax.set_xlabel("Input Size (H=W)", fontsize=12)
    ax.set_ylabel("Mean Runtime (ms)", fontsize=12)
    ax.set_title("Runtime vs. Input Size — All Methods/Precisions",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(sizes_all)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes_all], fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 4: Error vs Runtime trade-off (scatter)
# ---------------------------------------------------------------------------

def plot_error_vs_runtime_tradeoff(
    rows: List[Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """
    Scatter plot of max absolute error vs mean runtime — the key trade-off.

    Each point represents one (method, precision, size) combination.
    Points are coloured by combo and shaped by input size.

    Parameters
    ----------
    rows     : list of result dicts
    out_path : output file path (auto-computed if None)

    Returns
    -------
    str — absolute path to the saved figure
    """
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "error_vs_runtime_tradeoff.png")

    fig, ax = plt.subplots(figsize=(10, 7))

    sizes_all = sorted({int(r["input_size"]) for r in rows})
    size_to_marker = {s: m for s, m in zip(sizes_all, ["o", "s", "^", "D", "P", "X"])}

    plotted_combos: set = set()
    plotted_sizes:  set = set()

    for row in rows:
        combo = _make_combo_key(row)
        sz    = int(row["input_size"])
        err   = row.get("max_abs_error", float("nan"))
        t     = row.get("mean_ms", float("nan"))

        if not math.isfinite(err) or not math.isfinite(t) or err <= 0:
            continue

        color  = COMBO_COLORS.get(combo, "#888888")
        marker = size_to_marker.get(sz, "o")
        label_combo = combo if combo not in plotted_combos else None
        label_size  = f"size {sz}×{sz}" if sz not in plotted_sizes else None

        ax.scatter(
            t, err,
            color=color, marker=marker,
            s=90, edgecolors="black", linewidths=0.5,
            label=label_combo, zorder=3,
        )
        plotted_combos.add(combo)
        plotted_sizes.add(sz)

        # Annotate with size
        ax.annotate(
            f"{sz}",
            (t, err),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Runtime (ms) — log scale", fontsize=12)
    ax.set_ylabel("Max Absolute Error — log scale", fontsize=12)
    ax.set_title(
        "Error vs. Runtime Trade-off\n(lower-left = better)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Add an arrow annotation to guide the eye
    ax.annotate(
        "← faster\n↓ more accurate",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=9, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    rows: Optional[List[Dict[str, Any]]] = None,
    csv_path: str = RESULTS_FILE,
) -> Dict[str, str]:
    """
    Generate all four plots.

    Parameters
    ----------
    rows     : pre-loaded result list (if None, reads from *csv_path*)
    csv_path : path to results CSV

    Returns
    -------
    dict mapping plot name → absolute file path
    """
    if rows is None:
        rows = load_results(csv_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    return {
        "error_vs_method":          plot_error_vs_method(rows),
        "mse_vs_method":            plot_mse_vs_method(rows),
        "runtime_vs_inputsize":     plot_runtime_vs_inputsize(rows),
        "error_vs_runtime_tradeoff": plot_error_vs_runtime_tradeoff(rows),
    }
