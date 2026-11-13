"""
__init__.py
===========
Package initialiser for the Winograd Reduced-Precision Study source package.

Exposes the most commonly used public symbols at the package level for
convenient imports, e.g.:

    from src import direct_conv_2d_fp32, winograd_conv_2d_fp32

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

from .direct_conv import (
    direct_conv_2d_fp32,
    direct_conv_2d_fp16,
    direct_conv_2d_int8_simulated,
    direct_conv_2d_multichannel,
)

from .winograd_conv import (
    G, Gt, B, Bt, A, At,
    winograd_conv_tile_fp32,
    winograd_conv_2d_fp32,
    winograd_conv_2d_fp16,
    winograd_conv_2d_int8_simulated,
    verify_winograd_vs_direct,
)

from .precision_utils import (
    quantize_to_int8,
    dequantize_from_int8,
    simulate_fp16_rounding,
    compute_max_abs_error,
    compute_mse,
    compute_snr_db,
    compute_relative_error,
)

from .benchmark import time_function, benchmark_convolution, print_benchmark_table
from .experiment import run_experiments
from .plot_results import generate_all_plots, load_results

__all__ = [
    # direct_conv
    "direct_conv_2d_fp32",
    "direct_conv_2d_fp16",
    "direct_conv_2d_int8_simulated",
    "direct_conv_2d_multichannel",
    # winograd_conv
    "G", "Gt", "B", "Bt", "A", "At",
    "winograd_conv_tile_fp32",
    "winograd_conv_2d_fp32",
    "winograd_conv_2d_fp16",
    "winograd_conv_2d_int8_simulated",
    "verify_winograd_vs_direct",
    # precision_utils
    "quantize_to_int8",
    "dequantize_from_int8",
    "simulate_fp16_rounding",
    "compute_max_abs_error",
    "compute_mse",
    "compute_snr_db",
    "compute_relative_error",
    # benchmark
    "time_function",
    "benchmark_convolution",
    "print_benchmark_table",
    # experiment
    "run_experiments",
    # plot_results
    "generate_all_plots",
    "load_results",
]
