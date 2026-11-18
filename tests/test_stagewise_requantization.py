import numpy as np

from src.direct_conv import direct_conv_2d_fp32
from src.precision_utils import compute_max_abs_error
from src.winograd_conv import winograd_conv_2d_int8_stagewise


def test_stagewise_requantization_is_finite_and_reasonably_bounded():
    rng = np.random.default_rng(7)
    inp = rng.standard_normal((32, 32)).astype(np.float32)
    inp = inp * 3.0
    kernel = np.array(
        [
            [1.0, 0.5, -0.25],
            [-0.5, 0.2, 0.75],
            [0.25, -0.75, 1.0],
        ],
        dtype=np.float32,
    )

    ref = direct_conv_2d_fp32(inp, kernel)
    output = winograd_conv_2d_int8_stagewise(inp, kernel)
    max_error = compute_max_abs_error(ref, output)

    assert output.shape == ref.shape
    assert np.all(np.isfinite(output))
    assert max_error < 1.0
