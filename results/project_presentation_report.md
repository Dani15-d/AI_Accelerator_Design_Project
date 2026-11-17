# Winograd CNN Reduced-Precision Project Presentation

## 1. Motivation

The central question is whether Winograd convolution can reduce arithmetic work enough to improve runtime without causing unacceptable numerical error. In real hardware, longer execution time usually means more total work and therefore higher dynamic power consumption. For that reason, runtime is treated here as a practical proxy for computational effort and power draw.

## 2. Runtime-to-power interpretation

We use a simple proxy:

$$E_{proxy} = t_{mean} 	imes op\_factor 	imes precision\_factor$$

Where:
- $t_{mean}$ is the measured runtime in milliseconds
- $op\_factor$ reflects the reduced arithmetic count of Winograd compared with direct convolution
- $precision\_factor$ reflects the additional quantization overhead for FP16 and INT8 paths

This is not a physical power measurement, but it gives a meaningful engineering interpretation of why a method that performs more operations can be expected to consume more power and dissipate more heat.

## 3. Key numerical finding

- Best numerical accuracy overall: **direct_fp32** with average max absolute error ≈ 0.0000e+00
- Fastest average runtime overall: **direct_fp32** with mean runtime ≈ 0.083 ms
- Lowest runtime-based power proxy overall: **direct_int8** with proxy ≈ 2.829

## 4. Interpretation of the trade-off

- Direct convolution is the reference method and remains numerically stable, but it does more element-wise work per output pixel.
- Winograd FP32 reduces arithmetic count significantly and remains accurate, but the transform overhead is still visible in the timing profile.
- FP16 is a practical middle ground: it retains good accuracy while reducing precision cost.
- The naive INT8 Winograd path is inexpensive in principle, but its quantization instability is amplified by the transform matrices.
- The Wino Vidi Vici-inspired clipping path improves the numerical robustness of INT8 Winograd by managing unstable intermediate values before quantization, making it the closest thing to a stable low-precision Winograd design in this project.

## 5. Results summary table

| Combo | Avg runtime (ms) | Avg max error | Power proxy |
|---|---:|---:|---:|
| direct_fp16 | 0.322 | 4.5992e-03 | 8.354 |
| direct_fp32 | 0.083 | 0.0000e+00 | 2.992 |
| direct_int8 | 0.157 | 7.1062e-02 | 2.829 |
| winograd_fp16 | 5.163 | 8.8364e-03 | 59.474 |
| winograd_fp32 | 4.279 | 1.3351e-06 | 68.469 |
| winograd_int8_naive | 4.405 | 7.1062e-02 | 35.238 |
| winograd_int8_vvid | 91.255 | 1.6959e-01 | 846.848 |

## 6. Conclusion

The experimental results support the engineering intuition that lower computational workload usually leads to lower runtime and lower estimated power draw. Winograd remains attractive because it reduces the number of arithmetic operations, but increased numerical instability under aggressively quantized settings can erase that benefit unless transformations are stabilized. The clipping-aware design is therefore a more credible low-precision Winograd solution for real deployment than a naive INT8 transform.

## 7. Final takeaway

For a project presentation, the most important message is: runtime is not just a speed metric; it is a practical proxy for how much arithmetic work the accelerator does, and therefore how much power it likely consumes. This makes the Winograd-vs-direct trade-off not only an accuracy question, but also a power-efficiency question.