import os
import sys

import numpy as np
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.preprocessing import handle_nans, crop_center, normalize_channels


def test_handle_nans_replaces_nan_with_zero_numpy():
    data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    result = handle_nans(data, replacement_val=0.0, method='zero')

    expected = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(result, expected)


def test_handle_nans_replaces_nan_with_mean_tensor():
    data = torch.tensor([[1.0, float('nan')], [3.0, 5.0]], dtype=torch.float32)
    result = handle_nans(data, method='mean')

    assert isinstance(result, torch.Tensor)
    expected_mean = torch.tensor(3.0, dtype=torch.float32)
    assert torch.allclose(result, torch.tensor([[1.0, expected_mean], [3.0, 5.0]]))


def test_handle_nans_replaces_nan_with_median_numpy():
    data = np.array([[2.0, np.nan], [5.0, 1.0]], dtype=np.float32)
    result = handle_nans(data, method='median')

    expected = np.array([[2.0, 2.0], [5.0, 1.0]], dtype=np.float32)
    assert np.array_equal(result, expected)


def test_handle_nans_unsupported_method_raises():
    data = np.array([[1.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match='Unsupported NaN replacement method'):
        handle_nans(data, method='invalid')


def test_crop_center_returns_center_section():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = crop_center(data, crop_h=2, crop_w=2)

    expected = np.array([[5.0, 6.0], [9.0, 10.0]], dtype=np.float32)
    assert result.shape == (2, 2)
    assert np.array_equal(result, expected)


def test_crop_center_pads_when_crop_larger_than_input():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = crop_center(data, crop_h=4, crop_w=4)

    assert result.shape == (4, 4)
    assert np.all(result[0, :] == result[1, :])
    assert np.all(result[-1, :] == result[-2, :])
    assert np.all(result[:, 0] == result[:, 1])
    assert np.all(result[:, -1] == result[:, -2])


def test_normalize_channels_minmax():
    data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    result = normalize_channels(data, method='minmax')

    expected = np.array([[0.0, 0.33333334], [0.6666667, 1.0]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_normalize_channels_zscore_tensor():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = normalize_channels(data, method='zscore')

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_normalize_channels_percentile_clips_and_scales():
    data = np.array([[0.0, 1.0], [100.0, 200.0]], dtype=np.float32)
    result = normalize_channels(data, method='percentile', clip_percentile=(0, 50))

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result[1, 1] == 1.0


def test_normalize_channels_none_returns_same_array():
    data = np.array([[5.0, 10.0]], dtype=np.float32)
    result = normalize_channels(data, method='none')

    assert np.array_equal(result, data)
