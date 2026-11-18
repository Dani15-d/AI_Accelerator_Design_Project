# Winograd reduced-precision study: measured results

## Research question and decision rule

This study tests when F(2x2,3x3) Winograd improves execution time without exceeding an explicit max-absolute-error budget. A configuration is called worthwhile only if it is faster than its matching PyTorch/cuDNN precision baseline **and** its max absolute error against the FP64 oracle is at or below 0.1. Runtime is not used as a power or energy proxy.

## Experimental method

- Vectorized PyTorch NCHW Winograd: filter transform, vectorized input tiles, 16 batched channel-reduction GEMMs, inverse transform, and output crop; zero padding is supported.
- Baselines: PyTorch `conv2d` (cuDNN on CUDA) in FP32, FP16, BF16, and input-quantized FP32; NumPy direct cross-correlation is measured separately for a single-channel sanity case.
- Ground truth: PyTorch convolution with FP64 input and weights on CPU. Reported relative error is mean elementwise absolute error divided by `|reference| + 1e-8`; normalized RMS is RMS error divided by reference RMS.
- INT8 Winograd quantizes transformed U and V independently per Winograd coordinate, accumulates the channel reduction in int32, and dequantizes for the inverse transform. `stagewise_int8` additionally requantizes the transformed product. `input_int8` is only a quantize-before-transform comparator, not an integer Winograd implementation.
- Quantization scales are dynamically calibrated from each input/weight batch (max-absolute by transform coordinate unless an explicit percentile is supplied to the API); this is not learned clipping or a reproduction of Wino Vidi Vici training.
- Layer shapes are representative ResNet-18 3x3 stage convolutions at batches 1, 8, and 32. Reported latency covers the complete PyTorch function, including transforms, padding, allocations and GEMMs; CUDA events time GPU execution. Each cell records repeat count and latency spread.
- Inputs and filters use seeded standard-normal values; accuracy results are synthetic layerwise measurements, not real ResNet activations.
- Operation counts use the slide's per-2x2-tile arithmetic convention (direct: 36 multiplies and 32 additions; Winograd: 16 multiplies and 56 additions), excluding transform and filter-preparation work. Direct counts use the exact cropped output area; Winograd counts include partially cropped edge tiles. Counts are not latency predictions.

## Run metadata

- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA driver: 591.66
- PyTorch / CUDA runtime / cuDNN: 2.5.1+cu121 / 12.1 / 90100
- GPU clock / temperature / board-power sample before the sweep: 210 / 405 MHz; 47 C; 3.87 W
- Telemetry is a single pre-sweep `nvidia-smi` sample (not synchronized with timed kernels or integrated energy); no energy claim is made.

## Results

See `crossover_speedup.png` for speedup by layer/batch/precision, `accuracy_runtime_tradeoff.png` for normalized-RMS error versus latency, and `error_vs_method.png`, `mse_vs_method.png`, and `runtime_vs_inputsize.png` for the precision and spatial-size views.

- **No Winograd configuration** met both criteria in this measured sweep (72 tested).
- Do not claim a speedup from the multiplication-count reduction alone: transform and launch overhead are part of the end-to-end measurement.

### Per-precision aggregate

| Precision | N | Mean speedup | Best speedup (layer, batch) | Faster | Within budget | Worthwhile | Mean normalized RMS | Worst max error |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| fp32 | 12 | 0.165x | 0.335x (stage3, N=32.0) | 0 | 12 | 0 | 3.445e-07 | 1.640e-04 |
| fp16 | 12 | 0.130x | 0.192x (stage2, N=1.0) | 0 | 1 | 0 | 8.199e-04 | 1.969e-01 |
| bf16 | 12 | 0.118x | 0.149x (stage2, N=1.0) | 0 | 0 | 0 | 6.559e-03 | 1.570e+00 |
| input_int8 | 12 | 0.253x | 0.316x (stage4, N=32.0) | 0 | 0 | 0 | 1.525e-02 | 3.617e+00 |
| int8 | 12 | 0.118x | 0.139x (stage3, N=32.0) | 0 | 0 | 0 | 2.354e-02 | 6.460e+00 |
| stagewise_int8 | 12 | 0.102x | 0.125x (stage3, N=1.0) | 0 | 0 | 0 | 3.008e-02 | 6.590e+00 |

For this measured implementation and sweep, the answer is **none of the tested reduced-precision Winograd configurations**: all 72 were slower than the selected PyTorch/cuDNN comparison and none met both parts of the decision rule. The 16-product Winograd arithmetic count is therefore not sufficient to predict a win; this unfused transform/GEMM/transform implementation pays substantial launch, layout, and transform overhead. A fused CUDA/Triton kernel and calibrated representative activations are the next experiments before making a hardware-level claim.

## Interpretation and limitations

- The crossover is hardware-, shape-, batch-, and precision-dependent. Consult `crossover_speedup.png` and the row-level CSV rather than extrapolating one aggregate speedup.
- Pre-sweep GPU clocks were 210/405 MHz; clocks were not sampled during timed kernels, so absolute latencies are specific to this dynamic-clock laptop session and should be repeated under a documented plugged-in performance-mode setup.
- FP16/BF16 operations are performed in the requested PyTorch tensor dtype. The INT8 paths are quantization simulations with integer transformed-domain channel accumulation; they do not establish the performance of a dedicated INT8 accelerator kernel.
- A max-error budget is application-specific. This report uses 0.1 as a numerical screening threshold, not a validated CNN-accuracy or task-quality limit.
- No CNN was trained or evaluated; no top-1 accuracy, ResNet end-to-end inference, or model-level quality claim is supported.
- Winograd transform additions and filter-transform amortization are omitted from the simple operation-count columns. The benchmark includes their runtime cost.

## Reproducibility

Run `python main.py --runs 100 --repeats 5` to regenerate the complete benchmark and report, or `python main.py --plot-only` to rebuild plots and this report from the saved CSV. Fixed seeds are recorded per row. Use a plugged-in performance-mode laptop and repeat after cooling for publication-quality comparisons.

## References

1. A. Lavin and S. Gray, “Fast Algorithms for Convolutional Neural Networks,” CVPR 2016, https://arxiv.org/abs/1509.09308.
2. E. Mori et al., “Wino Vidi Vici: Conquering Numerical Instability of 8-Bit Winograd Convolution,” WACV 2024.
3. Y. Gupta et al., “Deep Learning with Limited Numerical Precision,” ICML 2015, https://proceedings.mlr.press/v37/gupta15.html.
