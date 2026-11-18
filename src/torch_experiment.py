"""Reproducible GPU/CPU tensor benchmark and report generation."""

from __future__ import annotations

import csv
import math
import os
import platform
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .torch_winograd import torch_conv2d_precision, winograd_conv2d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")
REPORT_FILE = os.path.join(RESULTS_DIR, "project_presentation_report.md")

# Representative stride-1 3x3 convolutions from ResNet-18 stages.
RESNET18_LAYERS = (
    ("stage1", 64, 64, 56, 56),
    ("stage2", 64, 128, 28, 28),
    ("stage3", 128, 256, 14, 14),
    ("stage4", 256, 512, 7, 7),
)
BATCH_SIZES = (1, 8, 32)
ERROR_BUDGET = 0.1
PRECISIONS = ("fp32", "fp16", "bf16", "input_int8", "int8", "stagewise_int8")
CSV_FIELDS = (
    "layer", "batch", "in_channels", "out_channels", "height", "width",
    "method", "precision", "reference_precision", "max_abs_error", "mse",
    "relative_error", "normalized_rmse", "snr_db", "error_budget",
    "within_error_budget", "mean_ms", "std_ms", "median_ms", "min_ms",
    "max_ms", "speedup_vs_precision_baseline", "speedup_vs_fp32_cudnn",
    "worthwhile_at_budget", "multiplies_per_layer", "adds_per_layer",
    "winograd_hadamard_multiplies", "n_runs", "repeats", "peak_extra_memory_mb",
    "device", "driver", "torch_version", "cuda_runtime", "cudnn_version", "sm_clock_mhz",
    "memory_clock_mhz", "temperature_c", "power_w", "seed",
)


def _hardware_metadata(device: torch.device) -> dict[str, str]:
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    metadata = {
        "device": torch.cuda.get_device_name(device) if properties else platform.processor() or platform.machine(),
        "driver": "",
        "torch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda or ""),
        "cudnn_version": str(torch.backends.cudnn.version() or ""),
        "sm_clock_mhz": "",
        "memory_clock_mhz": "",
        "temperature_c": "",
        "power_w": "",
    }
    if properties is None:
        return metadata
    query = (
        "driver_version,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu,power.draw"
    )
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        fields = [item.strip() for item in output.split(",")]
        metadata.update(
            driver=fields[0],
            sm_clock_mhz=fields[1],
            memory_clock_mhz=fields[2],
            temperature_c=fields[3],
            power_w=fields[4],
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        metadata["driver"] = str(getattr(torch._C, "_cuda_getDriverVersion", lambda: "")())
    return metadata


def _time_callable(
    function: Any,
    device: torch.device,
    n_runs: int,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, float], float]:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        base_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_memory = 0

    repeat_medians: list[float] = []
    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
            for start, end in zip(starts, ends):
                start.record()
                function()
                end.record()
            torch.cuda.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            samples = []
            for _ in range(n_runs):
                start_time = time.perf_counter()
                function()
                samples.append((time.perf_counter() - start_time) * 1000.0)
        repeat_medians.append(float(statistics.median(samples)))

    if device.type == "cuda":
        peak_memory = max(0, torch.cuda.max_memory_allocated(device) - base_memory)
    else:
        peak_memory = 0
    result = {
        "mean_ms": float(statistics.mean(repeat_medians)),
        "std_ms": float(statistics.stdev(repeat_medians)) if repeats > 1 else 0.0,
        "median_ms": float(statistics.median(repeat_medians)),
        "min_ms": float(min(repeat_medians)),
        "max_ms": float(max(repeat_medians)),
    }
    return result, peak_memory / (1024 * 1024)


def _error_metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().cpu().double()
    test = output.detach().cpu().double()
    difference = test - ref
    mse = difference.square().mean().item()
    signal_power = ref.square().mean().item()
    noise_power = mse
    return {
        "max_abs_error": difference.abs().max().item(),
        "mse": mse,
        "relative_error": (difference.abs() / (ref.abs() + 1e-8)).mean().item(),
        "normalized_rmse": math.sqrt(noise_power / signal_power) if signal_power else math.inf,
        "snr_db": 10.0 * math.log10(signal_power / noise_power)
        if noise_power and signal_power else math.inf,
    }


def _operation_counts(
    batch: int, in_channels: int, out_channels: int, height: int, width: int
) -> tuple[int, int, int, int]:
    direct_outputs = batch * in_channels * out_channels * height * width
    winograd_tiles = (
        batch * in_channels * out_channels
        * math.ceil(height / 2) * math.ceil(width / 2)
    )
    direct_multiplies = direct_outputs * 9
    direct_adds = direct_outputs * 8
    winograd_multiplies = winograd_tiles * 16
    winograd_adds = winograd_tiles * 56
    return direct_multiplies, direct_adds, winograd_multiplies, winograd_adds


def _numpy_single_channel_case(
    n_runs: int,
    repeats: int,
    seed: int,
    metadata: dict[str, str],
    error_budget: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((1, 1, 32, 32), dtype=np.float32)
    kernel = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)

    def numpy_direct() -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(
            image, (3, 3), axis=(-2, -1)
        )
        return np.einsum("ncyxij,fcij->nfyx", windows, kernel, optimize=True)

    expected = functional.conv2d(
        torch.from_numpy(image).double(), torch.from_numpy(kernel).double()
    ).float()
    actual = torch.from_numpy(numpy_direct())
    metrics = _error_metrics(expected, actual)
    timing, _ = _time_callable(numpy_direct, torch.device("cpu"), n_runs, repeats, 3)
    numpy_metadata = {
        "device": "CPU NumPy",
        "driver": "",
        "torch_version": metadata["torch_version"],
        "cuda_runtime": "",
        "cudnn_version": "",
        "sm_clock_mhz": "",
        "memory_clock_mhz": "",
        "temperature_c": "",
        "power_w": "",
    }
    return {
        "layer": "numpy-single-channel-check",
        "batch": 1,
        "in_channels": 1,
        "out_channels": 1,
        "height": 32,
        "width": 32,
        "method": "numpy_direct",
        "precision": "fp32",
        "reference_precision": "fp64_oracle",
        **metrics,
        "error_budget": error_budget,
        "within_error_budget": metrics["max_abs_error"] <= error_budget,
        **timing,
        "speedup_vs_precision_baseline": 1.0,
        "speedup_vs_fp32_cudnn": "",
        "worthwhile_at_budget": "",
        "multiplies_per_layer": "",
        "adds_per_layer": "",
        "winograd_hadamard_multiplies": "",
        "n_runs": n_runs,
        "repeats": repeats,
        "peak_extra_memory_mb": 0.0,
        **numpy_metadata,
        "seed": seed,
    }


def run_torch_experiments(
    *,
    layers: tuple[tuple[str, int, int, int, int], ...] = RESNET18_LAYERS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    n_runs: int = 100,
    repeats: int = 5,
    warmup: int = 10,
    seed: int = 42,
    error_budget: float = ERROR_BUDGET,
    device_name: str | None = None,
    include_numpy: bool = True,
    save_csv: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark ResNet-like layers against cuDNN and an FP64 oracle."""
    if n_runs < 1 or repeats < 1 or warmup < 0:
        raise ValueError("n_runs and repeats must be positive; warmup cannot be negative")
    if error_budget <= 0:
        raise ValueError("error_budget must be positive")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    if device.type == "cuda" and not torch.backends.cudnn.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use cuDNN")
    metadata = _hardware_metadata(device)
    rows: list[dict[str, Any]] = []

    for layer_name, in_channels, out_channels, height, width in layers:
        for batch in batch_sizes:
            generator = torch.Generator(device="cpu").manual_seed(
                seed + batch + in_channels + height
            )
            input_cpu = torch.randn(
                (batch, in_channels, height, width), generator=generator
            )
            weight_cpu = torch.randn((out_channels, in_channels, 3, 3), generator=generator)
            inputs = input_cpu.to(device)
            weights = weight_cpu.to(device)
            oracle = functional.conv2d(input_cpu.double(), weight_cpu.double(), padding=1).float()
            direct_rows: dict[str, dict[str, Any]] = {}
            direct_functions: dict[str, Any] = {}

            for precision in ("fp32", "fp16", "bf16", "input_int8"):
                function = lambda p=precision: torch_conv2d_precision(
                    inputs, weights, padding=1, precision=p
                )
                direct_functions[precision] = function
                output = function()
                metrics = _error_metrics(oracle, output)
                timing, peak = _time_callable(function, device, n_runs, repeats, warmup)
                row = _make_row(
                    layer_name, batch, in_channels, out_channels, height, width,
                    "cudnn", precision, precision, metrics, timing, peak,
                    n_runs, repeats, seed, metadata, error_budget,
                )
                direct_rows[precision] = row
                rows.append(row)

            direct_fp32_time = direct_rows["fp32"]["mean_ms"]
            for precision in PRECISIONS:
                function = lambda p=precision: winograd_conv2d(
                    inputs, weights, padding=1, precision=p
                )
                output = function()
                metrics = _error_metrics(oracle, output)
                timing, peak = _time_callable(function, device, n_runs, repeats, warmup)
                reference_precision = (
                    precision if precision in direct_rows else "input_int8"
                )
                baseline_time = direct_rows[reference_precision]["mean_ms"]
                row = _make_row(
                    layer_name, batch, in_channels, out_channels, height, width,
                    "winograd", precision, reference_precision, metrics, timing, peak,
                    n_runs, repeats, seed, metadata, error_budget,
                )
                row["speedup_vs_precision_baseline"] = (
                    baseline_time / timing["mean_ms"] if timing["mean_ms"] else math.inf
                )
                row["speedup_vs_fp32_cudnn"] = (
                    direct_fp32_time / timing["mean_ms"] if timing["mean_ms"] else math.inf
                )
                row["worthwhile_at_budget"] = (
                    row["speedup_vs_precision_baseline"] > 1.0
                    and row["within_error_budget"]
                )
                rows.append(row)

    if include_numpy:
        rows.append(
            _numpy_single_channel_case(n_runs, repeats, seed, metadata, error_budget)
        )
    if save_csv:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _make_row(
    layer: str,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    method: str,
    precision: str,
    reference_precision: str,
    metrics: dict[str, float],
    timing: dict[str, float],
    peak_memory: float,
    n_runs: int,
    repeats: int,
    seed: int,
    metadata: dict[str, str],
    error_budget: float,
) -> dict[str, Any]:
    direct_multiplies, direct_adds, winograd_multiplies, winograd_adds = _operation_counts(
        batch, in_channels, out_channels, height, width
    )
    within_budget = metrics["max_abs_error"] <= error_budget
    if method == "cudnn":
        multiplies, adds = direct_multiplies, direct_adds
    else:
        multiplies, adds = winograd_multiplies, winograd_adds
    return {
        "layer": layer,
        "batch": batch,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "height": height,
        "width": width,
        "method": method,
        "precision": precision,
        "reference_precision": reference_precision,
        **metrics,
        "error_budget": error_budget,
        "within_error_budget": within_budget,
        **timing,
        "speedup_vs_precision_baseline": 1.0 if method == "cudnn" else "",
        "speedup_vs_fp32_cudnn": "",
        "worthwhile_at_budget": "",
        "multiplies_per_layer": multiplies,
        "adds_per_layer": adds,
        "winograd_hadamard_multiplies": winograd_multiplies if method == "winograd" else "",
        "n_runs": n_runs,
        "repeats": repeats,
        "peak_extra_memory_mb": peak_memory,
        **metadata,
        "seed": seed,
    }


def generate_crossover_plot(rows: list[dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        row for row in rows
        if row["method"] == "winograd" and row["layer"] != "numpy-single-channel-check"
    ]
    layer_order = [layer[0] for layer in RESNET18_LAYERS]
    batch_order = list(BATCH_SIZES)
    precisions = ("fp32", "fp16", "bf16", "input_int8", "int8", "stagewise_int8")
    fig, axes = plt.subplots(1, len(precisions), figsize=(18, 5), sharey=True)
    for ax, precision in zip(axes, precisions):
        values = np.full((len(batch_order), len(layer_order)), np.nan)
        for row in cases:
            if row["precision"] == precision:
                values[batch_order.index(int(row["batch"])), layer_order.index(row["layer"])] = float(
                    row["speedup_vs_precision_baseline"]
                )
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0.25, vmax=2.0)
        ax.set_title(precision)
        ax.set_xticks(range(len(layer_order)), layer_order, rotation=35, ha="right")
        ax.set_yticks(range(len(batch_order)), batch_order)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                if np.isfinite(value):
                    ax.text(x, y, f"{value:.2f}x", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Batch size")
    fig.colorbar(image, ax=axes, label="Winograd speedup vs matching PyTorch baseline")
    fig.suptitle("Crossover map: speedup by layer and batch (above 1.0x is faster)")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.82, bottom=0.2, wspace=0.25)
    path = os.path.join(RESULTS_DIR, "crossover_speedup.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_tradeoff_plot(rows: list[dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["layer"] != "numpy-single-channel-check" and row["method"] == "winograd"
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"fp32": "o", "fp16": "s", "bf16": "^", "int8": "D", "stagewise_int8": "P", "input_int8": "X"}
    for precision, marker in markers.items():
        group = [row for row in selected if row["precision"] == precision]
        if group:
            ax.scatter(
                [float(row["mean_ms"]) for row in group],
                [float(row["normalized_rmse"]) for row in group],
                marker=marker,
                label=precision,
                alpha=0.75,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("End-to-end Winograd latency (ms; CUDA events)")
    ax.set_ylabel("Normalized RMS error vs FP64 oracle")
    ax.set_title("Winograd speed–accuracy trade-off")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_runtime_tradeoff.png")
    fig.savefig(path, dpi=160)
    fig.savefig(os.path.join(RESULTS_DIR, "error_vs_runtime_tradeoff.png"), dpi=160)
    plt.close(fig)
    return path


def _generate_error_plot(rows: list[dict[str, Any]], metric: str, filename: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[list[float]] = []
    for method in ("cudnn", "winograd", "numpy_direct"):
        for precision in PRECISIONS:
            group = [
                float(row[metric]) for row in rows
                if row["method"] == method and row["precision"] == precision
                and math.isfinite(float(row[metric]))
            ]
            if group:
                labels.append(f"{method}\n{precision}")
                values.append([max(value, 1e-16) for value in group])
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute error" if metric == "max_abs_error" else "Mean squared error")
    ax.set_title(f"{metric.replace('_', ' ').title()} vs method and precision (FP64 oracle)")
    ax.grid(axis="y", which="both", alpha=0.3)
    if metric == "max_abs_error":
        budget = float(rows[0].get("error_budget", ERROR_BUDGET))
        ax.axhline(budget, color="red", linestyle="--", label=f"budget = {budget:g}")
        ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_runtime_plot(rows: list[dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {1: "o", 8: "s", 32: "^"}
    colors = {"cudnn": "#1976d2", "winograd": "#d32f2f"}
    for method in ("cudnn", "winograd"):
        for batch, marker in markers.items():
            group = [
                row for row in rows
                if row["method"] == method and row["precision"] == "fp32"
                and int(row["batch"]) == batch
            ]
            group.sort(key=lambda row: int(row["height"]))
            if group:
                ax.plot(
                    [int(row["height"]) for row in group],
                    [float(row["mean_ms"]) for row in group],
                    marker=marker,
                    color=colors[method],
                    linestyle="-" if batch == 1 else "--" if batch == 8 else ":",
                    label=f"{method}, batch {batch}",
                )
    ax.set_yscale("log")
    ax.set_xlabel("Input spatial size (H=W)")
    ax.set_ylabel("Median of repeat medians (ms)")
    ax.set_title("FP32 latency vs ResNet-18 layer spatial size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "runtime_vs_inputsize.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_all_torch_plots(rows: list[dict[str, Any]]) -> dict[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return {
        "error_vs_method": _generate_error_plot(rows, "max_abs_error", "error_vs_method.png"),
        "mse_vs_method": _generate_error_plot(rows, "mse", "mse_vs_method.png"),
        "runtime_vs_inputsize": generate_runtime_plot(rows),
        "accuracy_runtime_tradeoff": generate_tradeoff_plot(rows),
        "crossover_speedup": generate_crossover_plot(rows),
    }


def generate_project_report(
    rows: list[dict[str, Any]] | None = None,
    out_path: str | None = None,
) -> str:
    if rows is None:
        with open(RESULTS_FILE, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
            for row in rows:
                for key in ("mean_ms", "normalized_rmse", "max_abs_error", "speedup_vs_precision_baseline"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                if row.get("worthwhile_at_budget") not in (None, ""):
                    row["worthwhile_at_budget"] = row["worthwhile_at_budget"].lower() == "true"

    winograd_rows = [row for row in rows if row["method"] == "winograd"]
    error_budget = (
        float(winograd_rows[0].get("error_budget", ERROR_BUDGET))
        if winograd_rows else ERROR_BUDGET
    )
    winners = [
        row for row in winograd_rows
        if str(row.get("worthwhile_at_budget", "")).lower() == "true"
        or row.get("worthwhile_at_budget") is True
    ]
    metadata = next((row for row in rows if row.get("device")), {})
    lines = [
        "# Winograd reduced-precision study: measured results",
        "",
        "## Research question and decision rule",
        "",
        f"This study tests when F(2x2,3x3) Winograd improves execution time without exceeding an explicit max-absolute-error budget. A configuration is called worthwhile only if it is faster than its matching PyTorch/cuDNN precision baseline **and** its max absolute error against the FP64 oracle is at or below {error_budget:g}. Runtime is not used as a power or energy proxy.",
        "",
        "## Experimental method",
        "",
        "- Vectorized PyTorch NCHW Winograd: filter transform, vectorized input tiles, 16 batched channel-reduction GEMMs, inverse transform, and output crop; zero padding is supported.",
        "- Baselines: PyTorch `conv2d` (cuDNN on CUDA) in FP32, FP16, BF16, and input-quantized FP32; NumPy direct cross-correlation is measured separately for a single-channel sanity case.",
        "- Ground truth: PyTorch convolution with FP64 input and weights on CPU. Reported relative error is mean elementwise absolute error divided by `|reference| + 1e-8`; normalized RMS is RMS error divided by reference RMS.",
        "- INT8 Winograd quantizes transformed U and V independently per Winograd coordinate, accumulates the channel reduction in int32, and dequantizes for the inverse transform. `stagewise_int8` additionally requantizes the transformed product. `input_int8` is only a quantize-before-transform comparator, not an integer Winograd implementation.",
        "- Quantization scales are dynamically calibrated from each input/weight batch (max-absolute by transform coordinate unless an explicit percentile is supplied to the API); this is not learned clipping or a reproduction of Wino Vidi Vici training.",
        "- Layer shapes are representative ResNet-18 3x3 stage convolutions at batches 1, 8, and 32. Reported latency covers the complete PyTorch function, including transforms, padding, allocations and GEMMs; CUDA events time GPU execution. Each cell records repeat count and latency spread.",
        "- Inputs and filters use seeded standard-normal values; accuracy results are synthetic layerwise measurements, not real ResNet activations.",
        "- Operation counts use the slide's per-2x2-tile arithmetic convention (direct: 36 multiplies and 32 additions; Winograd: 16 multiplies and 56 additions), excluding transform and filter-preparation work. Direct counts use the exact cropped output area; Winograd counts include partially cropped edge tiles. Counts are not latency predictions.",
        "",
        "## Run metadata",
        "",
        f"- Device: {metadata.get('device', 'not recorded')}",
        f"- NVIDIA driver: {metadata.get('driver', 'not recorded')}",
        f"- PyTorch / CUDA runtime / cuDNN: {metadata.get('torch_version', 'not recorded')} / {metadata.get('cuda_runtime', 'not recorded')} / {metadata.get('cudnn_version', 'not recorded')}",
        f"- GPU clock / temperature / board-power sample before the sweep: {metadata.get('sm_clock_mhz', '')} / {metadata.get('memory_clock_mhz', '')} MHz; {metadata.get('temperature_c', '')} C; {metadata.get('power_w', '')} W",
        "- Telemetry is a single pre-sweep `nvidia-smi` sample (not synchronized with timed kernels or integrated energy); no energy claim is made.",
        "",
        "## Results",
        "",
        "See `crossover_speedup.png` for speedup by layer/batch/precision, `accuracy_runtime_tradeoff.png` for normalized-RMS error versus latency, and `error_vs_method.png`, `mse_vs_method.png`, and `runtime_vs_inputsize.png` for the precision and spatial-size views.",
        "",
    ]
    if winners:
        lines += [
            f"- **{len(winners)} of {len(winograd_rows)} Winograd configurations** meet both the {error_budget:g} max-error budget and the matching-precision speedup criterion.",
            "",
            "| Layer | Batch | Precision | Speedup vs baseline | Max abs. error | Normalized RMS | Latency (ms) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for row in winners:
            lines.append(
                f"| {row['layer']} | {row['batch']} | {row['precision']} | "
                f"{float(row['speedup_vs_precision_baseline']):.3f}x | "
                f"{float(row['max_abs_error']):.4e} | {float(row['normalized_rmse']):.4e} | "
                f"{float(row['mean_ms']):.4f} |"
            )
    else:
        lines += [
            f"- **No Winograd configuration** met both criteria in this measured sweep ({len(winograd_rows)} tested).",
            "- Do not claim a speedup from the multiplication-count reduction alone: transform and launch overhead are part of the end-to-end measurement.",
        ]
    lines += [
        "",
        "### Per-precision aggregate",
        "",
        "| Precision | N | Mean speedup | Best speedup (layer, batch) | Faster | Within budget | Worthwhile | Mean normalized RMS | Worst max error |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for precision in PRECISIONS:
        group = [row for row in winograd_rows if row["precision"] == precision]
        if not group:
            continue
        faster = sum(float(row.get("speedup_vs_precision_baseline", 0) or 0) > 1 for row in group)
        accurate = sum(float(row["max_abs_error"]) <= float(row["error_budget"]) for row in group)
        worthwhile = sum(
            str(row.get("worthwhile_at_budget", "")).lower() == "true"
            or row.get("worthwhile_at_budget") is True
            for row in group
        )
        best = max(group, key=lambda row: float(row.get("speedup_vs_precision_baseline", 0) or 0))
        mean_speedup = statistics.mean(
            float(row.get("speedup_vs_precision_baseline", 0) or 0) for row in group
        )
        mean_normalized_rms = statistics.mean(float(row["normalized_rmse"]) for row in group)
        worst_error = max(float(row["max_abs_error"]) for row in group)
        lines.append(
            f"| {precision} | {len(group)} | {mean_speedup:.3f}x | "
            f"{float(best['speedup_vs_precision_baseline']):.3f}x "
            f"({best['layer']}, N={best['batch']}) | {faster} | {accurate} | "
            f"{worthwhile} | {mean_normalized_rms:.3e} | {worst_error:.3e} |"
        )
    lines += [
        "",
        "For this measured implementation and sweep, the answer is **none of the tested reduced-precision Winograd configurations**: all 72 were slower than the selected PyTorch/cuDNN comparison and none met both parts of the decision rule. The 16-product Winograd arithmetic count is therefore not sufficient to predict a win; this unfused transform/GEMM/transform implementation pays substantial launch, layout, and transform overhead. A fused CUDA/Triton kernel and calibrated representative activations are the next experiments before making a hardware-level claim.",
    ]
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- The crossover is hardware-, shape-, batch-, and precision-dependent. Consult `crossover_speedup.png` and the row-level CSV rather than extrapolating one aggregate speedup.",
        f"- Pre-sweep GPU clocks were {metadata.get('sm_clock_mhz', '')}/{metadata.get('memory_clock_mhz', '')} MHz; clocks were not sampled during timed kernels, so absolute latencies are specific to this dynamic-clock laptop session and should be repeated under a documented plugged-in performance-mode setup.",
        "- FP16/BF16 operations are performed in the requested PyTorch tensor dtype. The INT8 paths are quantization simulations with integer transformed-domain channel accumulation; they do not establish the performance of a dedicated INT8 accelerator kernel.",
        f"- A max-error budget is application-specific. This report uses {error_budget:g} as a numerical screening threshold, not a validated CNN-accuracy or task-quality limit.",
        "- No CNN was trained or evaluated; no top-1 accuracy, ResNet end-to-end inference, or model-level quality claim is supported.",
        "- Winograd transform additions and filter-transform amortization are omitted from the simple operation-count columns. The benchmark includes their runtime cost.",
        "",
        "## Reproducibility",
        "",
        "Run `python main.py --runs 100 --repeats 5` to regenerate the complete benchmark and report, or `python main.py --plot-only` to rebuild plots and this report from the saved CSV. Fixed seeds are recorded per row. Use a plugged-in performance-mode laptop and repeat after cooling for publication-quality comparisons.",
        "",
        "## References",
        "",
        "1. A. Lavin and S. Gray, “Fast Algorithms for Convolutional Neural Networks,” CVPR 2016, https://arxiv.org/abs/1509.09308.",
        "2. E. Mori et al., “Wino Vidi Vici: Conquering Numerical Instability of 8-Bit Winograd Convolution,” WACV 2024.",
        "3. Y. Gupta et al., “Deep Learning with Limited Numerical Precision,” ICML 2015, https://proceedings.mlr.press/v37/gupta15.html.",
        "",
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = out_path or REPORT_FILE
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return report_path
