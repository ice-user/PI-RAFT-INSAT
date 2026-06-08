import numpy as np
import torch


def handle_nans(data_matrix, replacement_val=0.0, method="zero"):
    """
    Replace invalid values in a 2D array or tensor.

    Args:
        data_matrix (np.ndarray|torch.Tensor): Input data grid.
        replacement_val (float): Fallback value for missing entries.
        method (str): Replacement method: "zero", "mean", or "median".

    Returns:
        same type as data_matrix: sanitized output.
    """
    if isinstance(data_matrix, torch.Tensor):
        data = data_matrix.cpu().numpy()
        is_tensor = True
    else:
        data = np.array(data_matrix, dtype=np.float32)
        is_tensor = False

    if method == "zero":
        clean = np.nan_to_num(data, nan=replacement_val)
    elif method == "mean":
        mean_val = np.nanmean(data)
        clean = np.nan_to_num(data, nan=mean_val)
    elif method == "median":
        median_val = np.nanmedian(data)
        clean = np.nan_to_num(data, nan=median_val)
    else:
        raise ValueError(f"Unsupported NaN replacement method: {method}")

    return torch.from_numpy(clean) if is_tensor else clean


def crop_center(data_matrix, crop_h=512, crop_w=512):
    """
    Crops a central region from a 2D array.

    If the requested crop is larger than the source image, the function
    pads the image symmetrically with edge values.
    """
    h, w = data_matrix.shape
    if crop_h > h or crop_w > w:
        pad_h = max(0, (crop_h - h + 1) // 2)
        pad_w = max(0, (crop_w - w + 1) // 2)
        data_matrix = np.pad(
            data_matrix,
            pad_width=((pad_h, pad_h), (pad_w, pad_w)),
            mode="edge"
        )
        h, w = data_matrix.shape

    start_h = max(0, h // 2 - crop_h // 2)
    start_w = max(0, w // 2 - crop_w // 2)
    return data_matrix[start_h:start_h + crop_h, start_w:start_w + crop_w]


def normalize_channels(image_tensor, method="minmax", clip_percentile=(1, 99)):
    """
    Normalize a 2D image using a standard scaling method.

    Args:
        image_tensor (np.ndarray|torch.Tensor): 2D image grid.
        method (str): One of ["minmax", "zscore", "percentile", "none"].
        clip_percentile (tuple): Percentiles used for percentile normalization.

    Returns:
        np.ndarray or torch.Tensor: normalized image in the same type.
    """
    if isinstance(image_tensor, torch.Tensor):
        data = image_tensor.cpu().numpy().astype(np.float32)
        is_tensor = True
    else:
        data = np.array(image_tensor, dtype=np.float32)
        is_tensor = False

    if method == "none":
        normalized = data
    elif method == "minmax":
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        span = max_val - min_val if max_val > min_val else 1.0
        normalized = (data - min_val) / span
    elif method == "zscore":
        mu = np.nanmean(data)
        sigma = np.nanstd(data)
        normalized = (data - mu) / (sigma if sigma > 0 else 1.0)
    elif method == "percentile":
        low, high = np.nanpercentile(data, clip_percentile)
        clipped = np.clip(data, low, high)
        normalized = (clipped - low) / max(high - low, 1e-6)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return torch.from_numpy(normalized) if is_tensor else normalized
