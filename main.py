"""Entry point for the vectorized Winograd precision and crossover study."""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as functional

from src.torch_experiment import (
    ERROR_BUDGET,
    RESULTS_FILE,
    generate_all_torch_plots,
    generate_project_report,
    run_torch_experiments,
)
from src.torch_winograd import winograd_conv2d


def _read_rows() -> list[dict[str, object]]:
    with open(RESULTS_FILE, newline="", encoding="utf-8") as file:
        rows: list[dict[str, object]] = list(csv.DictReader(file))
    for row in rows:
        for key in (
            "batch", "in_channels", "out_channels", "height", "width",
            "max_abs_error", "mse", "relative_error", "normalized_rmse",
            "snr_db", "error_budget", "mean_ms", "std_ms", "median_ms",
            "min_ms", "max_ms", "speedup_vs_precision_baseline",
            "speedup_vs_fp32_cudnn",
        ):
            if row.get(key) not in (None, ""):
                row[key] = float(str(row[key]))
        if row.get("worthwhile_at_budget") not in (None, ""):
            row["worthwhile_at_budget"] = str(row["worthwhile_at_budget"]).lower() == "true"
    return rows


def _verify() -> None:
    generator = torch.Generator().manual_seed(7)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device_name in devices:
        device = torch.device(device_name)
        inputs = torch.randn((2, 3, 9, 11), generator=generator).to(device)
        weights = torch.randn((5, 3, 3, 3), generator=generator).to(device)
        actual = winograd_conv2d(inputs, weights, padding=1)
        expected = functional.conv2d(inputs.double(), weights.double(), padding=1).float()
        error = (actual - expected).abs().max().item()
        print(f"{device_name}: FP32 vs FP64-oracle max error = {error:.3e}")
        if error > 1e-4:
            raise RuntimeError(f"Winograd correctness check failed on {device_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run a CPU/GPU FP64-oracle check")
    parser.add_argument("--plot-only", action="store_true", help="rebuild plots and report from results.csv")
    parser.add_argument("--runs", type=int, default=100, help="timed CUDA-event samples per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="independent timing repeats")
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-budget", type=float, default=ERROR_BUDGET)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--no-numpy", action="store_true",
        help="skip the separate single-channel NumPy direct baseline",
    )
    args = parser.parse_args()

    if args.verify:
        _verify()
        return

    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            raise FileNotFoundError(f"Results CSV does not exist: {RESULTS_FILE}")
        rows = _read_rows()
    else:
        device = None if args.device == "auto" else args.device
        rows = run_torch_experiments(
            n_runs=args.runs,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
            error_budget=args.error_budget,
            device_name=device,
            include_numpy=not args.no_numpy,
        )

    plot_paths = generate_all_torch_plots(rows)
    report = generate_project_report(rows)
    eligible = [
        row for row in rows
        if row.get("method") == "winograd"
        and (row.get("worthwhile_at_budget") is True
             or str(row.get("worthwhile_at_budget", "")).lower() == "true")
    ]
    print(f"Saved {RESULTS_FILE}")
    for path in plot_paths.values():
        print(f"Saved {path}")
    print(f"Saved {report}")
    print(f"Winograd configurations meeting the {args.error_budget:.3g} max-error budget and speedup rule: {len(eligible)}")
    if eligible:
        for row in eligible:
            print(
                f"  {row['layer']} N={row['batch']} {row['precision']}: "
                f"{float(row['speedup_vs_precision_baseline']):.2f}x, "
                f"max error {float(row['max_abs_error']):.3g}"
            )


if __name__ == "__main__":
    main()
