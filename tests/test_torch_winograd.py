import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from scipy.signal import correlate2d

from src.torch_experiment import _operation_counts
from src.torch_winograd import winograd_conv2d


@pytest.mark.parametrize(
    ("shape", "padding"),
    [
        ((2, 3, 9, 11), 0),
        ((2, 4, 8, 13), 1),
        ((1, 2, 7, 10), (2, 1)),
    ],
)
def test_fp32_matches_fp64_oracle_for_batches_and_padding(shape, padding):
    generator = torch.Generator().manual_seed(21)
    _, channels, _, _ = shape
    inputs = torch.randn(shape, generator=generator)
    weights = torch.randn((5, channels, 3, 3), generator=generator)
    actual = winograd_conv2d(inputs, weights, padding=padding)
    expected = functional.conv2d(
        inputs.double(), weights.double(), padding=padding
    ).float()
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_single_channel_matches_scipy_cross_correlation():
    rng = np.random.default_rng(4)
    image = rng.standard_normal((9, 11)).astype(np.float32)
    kernel = rng.standard_normal((3, 3)).astype(np.float32)
    actual = winograd_conv2d(
        torch.from_numpy(image)[None, None],
        torch.from_numpy(kernel)[None, None],
    )[0, 0].numpy()
    expected = correlate2d(image, kernel, mode="valid")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("precision", "max_error_tolerance", "relative_rms_tolerance"),
    [
        ("fp16", 0.08, 0.002),
        ("bf16", 0.5, 0.01),
        ("int8", 0.7, 0.03),
        ("stagewise_int8", 0.8, 0.03),
        ("input_int8", 0.5, 0.02),
    ],
)
def test_reduced_precision_is_finite_and_within_error_budget(
    precision, max_error_tolerance, relative_rms_tolerance
):
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randn((2, 8, 15, 13), generator=generator)
    weights = torch.randn((6, 8, 3, 3), generator=generator)
    actual = winograd_conv2d(inputs, weights, padding=1, precision=precision)
    oracle = functional.conv2d(inputs.double(), weights.double(), padding=1).float()
    error = actual - oracle
    assert torch.isfinite(actual).all()
    assert error.abs().max().item() < max_error_tolerance
    relative_rms = error.square().mean().sqrt() / oracle.square().mean().sqrt()
    assert relative_rms.item() < relative_rms_tolerance


def test_int8_path_quantizes_transformed_values_and_exposes_stages():
    generator = torch.Generator().manual_seed(9)
    inputs = torch.randn((1, 4, 9, 9), generator=generator)
    weights = torch.randn((3, 4, 3, 3), generator=generator)
    output, stages = winograd_conv2d(
        inputs, weights, padding=1, precision="int8", return_stages=True
    )
    assert output.shape == (1, 3, 9, 9)
    assert set(stages) == {
        "input_transform",
        "filter_transform",
        "hadamard",
        "output_transform",
    }
    assert stages["input_transform"].shape[-1] == 16
    assert stages["filter_transform"].shape[-1] == 16
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_int8_uses_integer_accumulation_and_matches_oracle():
    generator = torch.Generator(device="cuda").manual_seed(11)
    inputs = torch.randn((2, 4, 9, 11), generator=generator, device="cuda")
    weights = torch.randn((5, 4, 3, 3), generator=generator, device="cuda")
    actual = winograd_conv2d(inputs, weights, padding=1, precision="int8")
    oracle = functional.conv2d(inputs.cpu().double(), weights.cpu().double(), padding=1).float()
    assert torch.isfinite(actual).all()
    assert (actual.cpu() - oracle).abs().max().item() < 0.6


def test_percentile_calibration_and_validation():
    inputs = torch.randn((1, 4, 9, 9), generator=torch.Generator().manual_seed(8))
    weights = torch.randn((3, 4, 3, 3), generator=torch.Generator().manual_seed(9))
    output = winograd_conv2d(
        inputs, weights, padding=1, precision="int8", clip_percentile=99.0
    )
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="clip_percentile"):
        winograd_conv2d(inputs, weights, precision="int8", clip_percentile=0.0)


def test_operation_counts_use_cropped_direct_outputs_and_padded_tiles():
    direct_multiplies, direct_adds, winograd_multiplies, winograd_adds = _operation_counts(
        batch=1, in_channels=2, out_channels=3, height=7, width=5
    )
    assert (direct_multiplies, direct_adds) == (1890, 1680)
    assert (winograd_multiplies, winograd_adds) == (1152, 4032)
