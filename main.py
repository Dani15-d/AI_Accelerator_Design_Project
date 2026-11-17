"""
main.py
=======
Entry point for the Winograd CNN Convolution – Reduced Precision Study.

Usage
-----
    python main.py              # run full experiment + generate plots
    python main.py --verify     # only run correctness check
    python main.py --plot-only  # regenerate plots from existing CSV

Authors : Jaswanth Palla (CS23B034), Danish (CS23B047)
Institute: IIT Tirupati
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment import run_experiments
from src.plot_results import generate_all_plots, generate_project_report
from src.winograd_conv import verify_winograd_vs_direct
import numpy as np


def run_correctness_check():
    """Quick sanity check: Winograd FP32 must match Direct FP32 within 1e-4."""
    print("\n--- Correctness Check ---")
    rng = np.random.default_rng(0)
    for sz in [8, 16, 32]:
        inp = rng.standard_normal((sz, sz)).astype(np.float32)
        ker = rng.standard_normal((3, 3)).astype(np.float32)
        ok = verify_winograd_vs_direct(inp, ker, tol=1e-4, verbose=True)
        print(f"  Input {sz}×{sz}: {'PASS ✓' if ok else 'FAIL ✗'}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    print("=" * 60)
    print("  Winograd CNN Convolution: Reduced Precision Study")
    print("  IIT Tirupati — CS23B034 & CS23B047")
    print("=" * 60)

    if "--verify" in args:
        run_correctness_check()
        sys.exit(0)

    if "--plot-only" in args:
        print("\n[Re-generating plots from existing CSV …]")
        paths = generate_all_plots()
        report = generate_project_report()
        print("\n[✓] Plots generated:")
        for name, path in paths.items():
            print(f"    {name}: {path}")
        print(f"    project_report: {report}")
        sys.exit(0)

    # Default: full pipeline
    run_correctness_check()

    print("\n[1/2] Running experiments …")
    results = run_experiments(verbose=True)

    print("\n[2/2] Generating plots and presentation report …")
    plot_paths = generate_all_plots(rows=results)
    report_path = generate_project_report(rows=results)

    print("\n" + "=" * 60)
    print("  Done!  Check the results/ directory for:")
    print("    • results.csv")
    for name, path in plot_paths.items():
        print(f"    • {os.path.basename(path)}")
    print(f"    • {os.path.basename(report_path)}")
    print("=" * 60)
    print("\nTo explore interactively:")
    print("  jupyter notebook notebooks/analysis.ipynb")
