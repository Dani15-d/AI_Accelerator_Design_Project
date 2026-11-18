# Winograd reduced-precision convolution study

This project measures the end-to-end speed and numerical error of Winograd
F(2x2, 3x3) against PyTorch `conv2d` on representative ResNet-18 layer shapes.
The decision rule is explicit: a Winograd run is worthwhile only when it is
faster than its matching PyTorch precision baseline and its maximum absolute
error against the FP64 oracle is no greater than the configured error budget
(default `0.1`).

## What is implemented

- Vectorized PyTorch NCHW convolution with arbitrary batch/channel counts,
  multiple output filters, valid or symmetric zero padding, odd output sizes,
  and bias.
- FP32, FP16, and BF16 transforms and channel GEMMs.
- Transform-domain INT8: transformed filters and input tiles are quantized
  per Winograd coordinate; CUDA uses INT8 matrix multiplication with int32
  accumulation. `stagewise_int8` also requantizes the transformed product.
- `input_int8` is a comparison path that quantizes before transforms; it is
  not an INT8 Winograd implementation. All quantization is simulated and does
  not establish performance of a dedicated accelerator kernel.
- cuDNN/PyTorch precision baselines, a separate NumPy single-channel direct
  baseline, FP64 oracle comparisons, max/mean/relative/normalized-RMS errors,
  nominal operation counts, peak extra CUDA allocation, repeat statistics,
  and device/driver/runtime/clock metadata.
- A shape/batch/precision crossover plot, an accuracy–runtime plot, and a
  generated report with no runtime-as-energy proxy.
- Additional max-error, MSE, and FP32 spatial-latency plots are regenerated
  from the same benchmark CSV.

The stage-wise INT8 option is a requantization experiment inspired by
transform-domain stability work; it does not reproduce Wino Vidi Vici's
learned clipping or quantization-aware training. No CNN has been trained or
evaluated, so this repository makes no model-accuracy claim.
The older single-channel NumPy API remains available for comparison, but its
runner writes `results/legacy_numpy_results.csv` and is not the primary study.

## Environment and installation

Python 3.10–3.12 is supported. For a CUDA 12.1 PyTorch wheel on Windows or
Linux, install the pinned CUDA build first, then the remaining pinned
requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU-only execution, install the CPU wheel instead:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Check CUDA detection with `python -c "import torch; print(torch.cuda.is_available())"`.
The pinned dependencies are in [requirements.txt](requirements.txt).

## Run

Run the full benchmark (default: 100 timed samples per repeat, 5 repeats, 10
warmups) on CUDA when available:

```powershell
python main.py
```

Useful options:

```powershell
python main.py --verify
python main.py --runs 100 --repeats 5 --warmup 10 --error-budget 0.1
python main.py --device cpu --runs 10 --repeats 3
python main.py --plot-only
python -m pytest -q
```

The default sweep covers ResNet-18 stage 3x3 shapes
`(Cin,Cout,H,W) = (64,64,56,56), (64,128,28,28), (128,256,14,14),
(256,512,7,7)` at batch sizes 1, 8, and 32. The random inputs and filters are
seeded standard-normal tensors, not captured ResNet activations. FP64 CPU
convolution is the numerical oracle. CUDA-event latency includes transforms,
padding, GEMMs, allocation and output transform, but excludes host-side input
generation and oracle computation.

## Results

`results/results.csv` is the row-level benchmark data. Each row records tensor
shape, method/precision, error metrics, timing distribution, operation counts,
error budget decision, memory use, seed, GPU name, driver, PyTorch/CUDA version,
and sampled clocks/temperature/board power where `nvidia-smi` is available.
`results/project_presentation_report.md` and the plots are generated from that
CSV. Board power is a metadata sample only—not an energy measurement. Runtime
is not treated as power. Clock, temperature, and board-power metadata is
sampled before the sweep, not synchronized with individual timed kernels.

The nominal count convention is per 2x2 output tile and input/output channel:
direct uses 36 multiplies and 32 additions; Winograd uses 16 Hadamard
multiplies and 56 additions. This excludes transform/filter-preparation work;
the measured end-to-end latency includes transform overhead.

## Packaging

The repository ignores `.venv`, `.git` is repository metadata, and Python
cache files are ignored. Distribute tracked source and results only; do not
include a local environment or `.git` directory in a submission archive.

## References

1. A. Lavin and S. Gray, “Fast Algorithms for Convolutional Neural Networks,”
   CVPR 2016. https://arxiv.org/abs/1509.09308
2. E. Mori et al., “Wino Vidi Vici: Conquering Numerical Instability of 8-Bit
   Winograd Convolution,” WACV 2024.
3. Y. Gupta et al., “Deep Learning with Limited Numerical Precision,”
   ICML 2015. https://proceedings.mlr.press/v37/gupta15.html
