# API Contracts and Tensor Semantics

This document defines the strict API contracts, expected tensor dimensions, data types (dtypes), invariants, and common usage mistakes for each module in the **PI-RAFT** repository. These rules are put in place to prevent **silent tensor drift [unintended changes in tensor dimensions or values as they pass through model layers]**.

---

## 1. Datasets Module (`datasets/`)

### GOES16ProxyDataset (`datasets/goes16_loader.py`)
*   **Purpose**: Fetches netCDF4 data from NOAA's public S3 bucket, crops a 512x512 matrix, handles NaNs, and yields temporal frame sequences.
*   **API Signature**:
    ```python
    GOES16ProxyDataset(product='ABI-L2-CMIPC', band='C09', year=2025, day_of_year=150, hour=14, sequence_length=4)
    ```
*   **Output shapes (from `__getitem__`)**:
    *   `frames`: `[T, 1, 512, 512]` (dtype: `torch.float32`)
    *   `mock_dem`: `[1, 512, 512]` (dtype: `torch.float32`)
*   **Invariants**:
    *   The output spatial dimensions ($H=512$, $W=512$) are hardcoded and must be identical across the frame stack and DEM.
    *   The pixel count values must be non-NaN (NaNs replaced with `0.0`).
*   **Common Mistakes**:
    *   Loading bands with different base spatial resolutions (e.g. VIS bands at 0.5km vs IR bands at 2km) without spatial grid resampling.
    *   Treating the dataset output as a 2-tuple of frames instead of a temporal stack.

### INSAT3DSProxyDataset (`datasets/insat3ds_loader.py`)
*   **Purpose**: Placeholder interface for INSAT-3DS HDF5 data loading, calibration, and geolocation.
*   **API Signature**:
    ```python
    INSAT3DSProxyDataset(data_dir=None, band='WV', year=2024, day_of_year=120, hour=14)
    ```
*   **Output shapes**:
    *   Currently raises `NotImplementedError` as it is a placeholder. Future outputs must match the shape contract: `[1, H, W]`.

---

## 2. Models Module (`models/`)

### PhysicsInformedRAFT (`models/pi_raft.py`)
*   **Purpose**: Computes optical flow and pressure vertical coordinates from a temporal image sequence.
*   **API Signature**:
    ```python
    model = PhysicsInformedRAFT(hidden_dim=128, corr_radius=3)
    flow, height = model(images, dem, iters=4)
    ```
*   **Input Shapes**:
    *   `images`: `[B, T, 1, H, W]` (dtype: `torch.float32`)
    *   `dem`: `[B, 1, H, W]` (dtype: `torch.float32`)
*   **Output Shapes**:
    *   `flow`: `[B, T-1, 2, H, W]` (dtype: `torch.float32`) - containing zonal `u` and meridional `v` velocities for each adjacent pair.
    *   `height`: `[B, T-1, 1, H, W]` (dtype: `torch.float32`) - containing per-pair cloud-top pressure in hPa.
*   **Invariants**:
    *   Input dimensions $H$ and $W$ must be multiples of 8 (due to the $1/8$ spatial downsampling tier).
    *   Output tensors are bilinearly upsampled to match input $H$ and $W$ exactly for each time-pair output.
*   **Common Mistakes**:
    *   Passing a pair of images instead of a temporal stack.
    *   Mismatch of hardware devices between inputs (e.g., placing `dem` on CPU while `images` are on GPU).

---

## 3. Losses Module (`losses/`)

### PhysicsInformedLoss (`losses/physics_loss.py`)
*   **Purpose**: Computes multi-constraint fluid dynamics losses using differentiable warping over a temporal image sequence.
*   **API Signature**:
    ```python
    criterion = PhysicsInformedLoss(alpha=0.5, beta=0.1, gamma=0.01, epsilon=0.001)
    total_loss, data_loss, physics_loss = criterion(flow_pred, height_pred, img_sequence, background_flow=None)
    ```
*   **Input Shapes**:
    *   `flow_pred`: `[B, T-1, 2, H, W]` (dtype: `torch.float32`)
    *   `height_pred`: `[B, T-1, 1, H, W]` (dtype: `torch.float32`)
    *   `img_sequence`: `[B, T, 1, H, W]` (dtype: `torch.float32`)
    *   `background_flow`: `[B, T-1, 2, H, W]` (dtype: `torch.float32`, optional)
*   **Output Shapes**:
    *   `total_loss`: `[]` or `[1]` (scalar, dtype: `torch.float32`)
    *   `data_loss`: `[]` or `[1]` (scalar, dtype: `torch.float32`)
    *   `physics_loss`: `[]` or `[1]` (scalar, dtype: `torch.float32`)
*   **Invariants**:
    *   The bilinear warp maps the values of `img2` back to the frame coordinates of `img1`.
    *   All loss values are average scalar metrics aggregated over batch and spatial bounds (`torch.mean`).
*   **Common Mistakes**:
    *   Passing `flow_pred` with dimensions swapped (e.g. `[B, H, W, 2]`). Warping expects `[B, 2, H, W]`.

---

## 4. Export Module (`export/`)

### export_amv_netcdf (`export/netcdf_export.py`)
*   **Purpose**: Writes arrays to a Climate and Forecast (CF) compliant NetCDF4 container on disk.
*   **API Signature**:
    ```python
    export_amv_netcdf(u_wind_array, v_wind_array, pressure_array, output_path, metadata=None)
    ```
*   **Input Shapes**:
    *   `u_wind_array`: `[H, W]` (dtype: `np.float32`)
    *   `v_wind_array`: `[H, W]` (dtype: `np.float32`)
    *   `pressure_array`: `[H, W]` (dtype: `np.float32`)
*   **Invariants**:
    *   All arrays must have identical height and width dimensions.
    *   Output NetCDF variables are compressed using zlib compression.

---

## 5. Target 4-Frame Contracts (Future Design)

For the planned 4-frame temporal sequence tracking:
*   **Model Input Tensor**: `[B, T, C, H, W]` where $T=4$.
*   **Model Flow Output**: `[B, T-1, 2, H, W]`.
*   **Invariants**: The temporal cadence ($\Delta t$) must be kept uniform across the sequence to ensure correct velocity derivatives.
