# Winograd CNN Convolution with Reduced Precision

> **IIT Tirupati — Department of Computer Science**  
> **Team:** Jaswanth Palla (CS23B034) · Danish (CS23B047)

---

## Research Question

> *When does reduced-precision Winograd convolution improve execution time without unacceptable numerical or model-accuracy loss?*

---

## Background

Convolutional Neural Networks (CNNs) are dominated by 3×3 convolutions. At inference time, these account for the bulk of multiply-accumulate (MAC) operations.  **Winograd's minimal filtering algorithm F(2×2, 3×3)** reduces the number of multiplications from **36 to 16** (55.6% fewer) by pre-transforming the input and filter into a domain where their element-wise product replaces the full convolution.

However, reducing arithmetic precision (FP16, INT8) introduces rounding errors that interact non-trivially with the Winograd transforms.  This project provides a **reproducible numerical benchmark** to study that interaction.

---

## Mathematical Formulation

### Direct Convolution (reference)
For a 3×3 filter `g` and input tile `d`:

```
output[i,j] = ΣΣ d[i+m, j+n] · g[m, n]   for m,n ∈ {0,1,2}
```

Each of the 4 output values requires 9 multiplications → **36 total**.

### Winograd F(2×2, 3×3)

$$Y = A^\top \bigl[(G\,g\,G^\top) \odot (B^\top\,d\,B)\bigr] A$$

where ⊙ is element-wise multiplication.

#### Transform Matrices

**G** (filter transform, 4×3):
```
G = [[ 1,    0,    0  ],
     [ 1/2,  1/2,  1/2],
     [ 1/2, -1/2,  1/2],
     [ 0,    0,    1  ]]
```

**B** (data transform, 4×4):
```
B = [[ 1,  0, -1,  0],
     [ 0,  1,  1,  0],
     [ 0, -1,  1,  0],
     [ 0,  1,  0, -1]]
```

**A** (inverse transform, 4×2):
```
A = [[1,  0],
     [1,  1],
     [1, -1],
     [0, -1]]
```

#### Multiplication Count
| Method    | Multiplications | Reduction |
|-----------|----------------|-----------|
| Direct    | 36             | —         |
| Winograd  | 16             | 55.6%     |

#### Three Failure Modes (studied in this project)
1. **Transform overhead** — the matrix multiplications (additions only, no muls) have a wall-clock cost that can exceed savings for small inputs.
2. **Numerical sensitivity** — FP16/INT8 quantisation errors are amplified by the transform matrices.
3. **Hardware dependence** — actual speedup depends on whether hardware has efficient GEMM/element-wise primitives for the transformed domain.

### Paper-inspired stability mode: Wino Vidi Vici
This project now includes a Winograd-INT8 path that explicitly models the key idea from the Wino Vidi Vici paper: clip the transformed intermediate values before each INT8 quantisation stage so the 8-bit representation does not overflow in the transformed domain. The naive INT8 Winograd baseline uses a one-shot quantisation of the input and kernel, while the new `winograd_int8_vvid` variant adds stage-wise clipping and re-quantisation of the transformed filter, activation tile, and element-wise product.

---

## Results Interpretation Guide

After running `python main.py`, the `results/` directory will contain:

| File | Description |
|------|-------------|
| `results.csv` | Raw data: errors + timings for each (size, method, precision) combo, including the paper-style `winograd_int8_vvid` benchmark |
| `error_vs_method.png` | Bar chart of max absolute error (log scale) at a fixed input size |
| `mse_vs_method.png` | Bar chart of MSE (log scale) at a fixed input size |
| `runtime_vs_inputsize.png` | Line chart of mean runtime vs spatial size |
| `error_vs_runtime_tradeoff.png` | **Key plot**: error vs runtime scatter — lower-left = better |

### Reading the Trade-off Plot

- Points in the **lower-left** corner are most desirable (fast + accurate).
- **FP32 methods** cluster near the bottom (low error) but may be slower.
- **INT8 methods** cluster at the right of the y-axis (highest error).
- The goal is to find configurations where Winograd offers a **runtime advantage without crossing your error tolerance**.

---

## Setup Instructions

### Prerequisites
- Python 3.9 or later
- pip

### Installation

```bash
# Clone or unzip the project
cd "AI Accelarator project"

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> PyTorch is **optional** and commented out in `requirements.txt`. All code runs on pure NumPy.

---

## Usage

### Run the full study

```bash
python main.py
```

This will:
1. Run a correctness check (Winograd FP32 vs Direct FP32).
2. Benchmark the direct and Winograd baselines at input sizes 8, 16, 32, 64, 128.
3. Include both the naive INT8 Winograd baseline and the clipping-aware `int8_vvid` variant inspired by Wino Vidi Vici.
4. Save `results/results.csv`.
5. Generate and save all 4 plots to `results/`.

### Verify correctness only

```bash
python main.py --verify
```

### Regenerate plots from an existing CSV

```bash
python main.py --plot-only
```

### Interactive exploration

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## File Structure

```
AI Accelarator project/
│
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/
│   ├── __init__.py           # Package init; re-exports public API
│   ├── direct_conv.py        # Direct convolution (FP32, FP16, INT8-sim)
│   ├── winograd_conv.py      # Winograd F(2×2,3×3) (FP32, FP16, INT8-sim)
│   ├── precision_utils.py    # Quantisation helpers + error metrics
│   ├── benchmark.py          # Timing harness + convolution benchmarker
│   ├── experiment.py         # Full experiment runner → CSV
│   └── plot_results.py       # 4-plot generator from CSV
│
├── notebooks/
│   └── analysis.ipynb        # Jupyter walkthrough (6 cells)
│
└── results/                  # Created automatically
    ├── results.csv
    ├── error_vs_method.png
    ├── mse_vs_method.png
    ├── runtime_vs_inputsize.png
    └── error_vs_runtime_tradeoff.png
```

---

## References

1. **Lavin, A. & Gray, S.** (2016). *Fast Algorithms for Convolutional Neural Networks*. CVPR 2016. [[arXiv](https://arxiv.org/abs/1509.09308)]

2. **Mori, E. et al.** (2024). *Wino Vidi Vici: Conquering Numerical Instability of 8-Bit Winograd Convolution*. WACV 2024.

---

## Implementation Notes

- **Tiling**: For an H×W input with a 3×3 kernel, the valid output is (H−2)×(W−2). The input is tiled into overlapping 4×4 patches with stride 2; each produces a 2×2 output patch.
- **Edge handling**: Inputs are zero-padded on the right/bottom to fit an integer number of tiles; the output is then cropped to the exact valid size.
- **INT8 simulation**: Quantisation uses per-tensor symmetric INT8 (range [−127, 127]). Multiply-accumulate is done in INT32 to prevent overflow.
- **Multi-channel**: Loop over channels; accumulate per-channel contributions.

---

*Last updated: November 2026*
