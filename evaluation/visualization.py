import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr


def _prepare_flow_for_visualization(flow_pred):
    """Normalize model flow output to a single [2, H, W] NumPy array."""
    if isinstance(flow_pred, torch.Tensor):
        flow_np = flow_pred.cpu().numpy()
    else:
        flow_np = np.asarray(flow_pred)

    # Remove a leading batch dimension if present
    if flow_np.ndim == 5:
        flow_np = flow_np[0]

    # Remove singleton batch dimension for a standard flow field [1, 2, H, W]
    if flow_np.ndim == 4 and flow_np.shape[0] == 1 and flow_np.shape[1] == 2:
        flow_np = flow_np[0]

    # If the model returns a sequence of adjacent-pair flows [T-1, 2, H, W], average them
    if flow_np.ndim == 4 and flow_np.shape[1] == 2:
        flow_np = flow_np.mean(axis=0)

    if flow_np.ndim != 3 or flow_np.shape[0] != 2:
        raise ValueError(
            f"Expected flow field shape [2,H,W] or [T-1,2,H,W], got {flow_np.shape}"
        )

    return flow_np


def plot_untrained_baseline(img1, flow_pred, title="Cell 6: Untrained Baseline Flow Map (Random Initial State)", stride=16):
    """
    Plots the wind vector quiver field overlaid on the background imagery for untrained initial state.
    
    Args:
        img1 (torch.Tensor or np.ndarray): Background satellite image, shape [1, H, W] or [H, W]
        flow_pred (torch.Tensor or np.ndarray): Predicted wind displacement field [u, v], shape [2, H, W] or [T-1, 2, H, W]
        title (str): Title of the plot.
        stride (int): Spatial stride interval to make quiver needles legible.
    """
    # Convert PyTorch tensors to NumPy arrays
    if isinstance(img1, torch.Tensor):
        img1_np = img1.squeeze(0).cpu().numpy()
        # Handle case where shape is [C, H, W] vs [H, W]
        if img1_np.ndim == 3:
            img1_np = img1_np[0]
    else:
        img1_np = img1[0] if img1.ndim == 3 else img1
        
    flow_np = _prepare_flow_for_visualization(flow_pred)

    u_val = flow_np[0]
    v_val = flow_np[1]
    magnitude = np.sqrt(u_val**2 + v_val**2)
    H, W = img1_np.shape

    # Build coordinate grid mapping the spatial space
    yy, xx = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')

    # Apply spatial stride to make vector needles readable
    xx_sub = xx[::stride, ::stride]
    yy_sub = yy[::stride, ::stride]
    u_sub = u_val[::stride, ::stride]
    v_sub = v_val[::stride, ::stride]
    mag_sub = magnitude[::stride, ::stride]

    plt.figure(figsize=(10, 10))
    plt.imshow(img1_np, cmap='twilight', origin='upper')

    # Overlay velocity arrows
    quiver = plt.quiver(xx_sub, yy_sub, u_sub, v_sub, mag_sub,
                        cmap='jet', angles='xy', scale_units='xy', scale=1.0)

    plt.colorbar(quiver, label='Random Initialized Vector Velocity Magnitude')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()


def plot_inference_scaling_comparison(img1, flow_low, flow_high, standard_iters=4, production_iters=12, scale_factor=0.1, stride=16):
    """
    Produces a side-by-side visualization comparing AMV field resolutions at standard vs production iteration depths.
    
    Args:
        img1 (torch.Tensor or np.ndarray): Background satellite image, shape [1, H, W] or [H, W]
        flow_low (torch.Tensor or np.ndarray): Low resolution / low iteration flow, shape [2, H, W] or [T-1, 2, H, W]
        flow_high (torch.Tensor or np.ndarray): High resolution / high iteration flow, shape [2, H, W] or [T-1, 2, H, W]
        standard_iters (int): Number of standard iterations.
        production_iters (int): Number of production iterations.
        scale_factor (float): Vector arrow scaling factor.
        stride (int): Spatial stride for quiver plotting.
    """
    if isinstance(img1, torch.Tensor):
        img1_np = img1.squeeze(0).cpu().numpy()
        if img1_np.ndim == 3:
            img1_np = img1_np[0]
    else:
        img1_np = img1[0] if img1.ndim == 3 else img1

    flow_low_np = _prepare_flow_for_visualization(flow_low)
    flow_high_np = _prepare_flow_for_visualization(flow_high)

    u_low, v_low = flow_low_np[0], flow_low_np[1]
    mag_low = np.sqrt(u_low**2 + v_low**2)

    u_high, v_high = flow_high_np[0], flow_high_np[1]
    mag_high = np.sqrt(u_high**2 + v_high**2)

    H, W = img1_np.shape
    yy, xx = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')

    xx_sub = xx[::stride, ::stride]
    yy_sub = yy[::stride, ::stride]

    # Calculate global min/max for consistent colorbar scaling
    global_mag_min = min(mag_low.min(), mag_high.min())
    global_mag_max = max(mag_low.max(), mag_high.max())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left Panel: Low Iteration Baseline
    axes[0].imshow(img1_np, cmap='twilight', origin='upper')
    quiver_low = axes[0].quiver(
        xx_sub, yy_sub, u_low[::stride, ::stride], v_low[::stride, ::stride], mag_low[::stride, ::stride],
        cmap='jet', angles='xy', scale_units='xy', scale=scale_factor,
        norm=plt.Normalize(vmin=global_mag_min, vmax=global_mag_max)
    )
    axes[0].set_title(f"Standard Inference Depth (iters={standard_iters})\n[Coarse Mesoscale Approximations]", fontsize=12)
    axes[0].axis('off')

    # Right Panel: Scaled Inference Depth
    axes[1].imshow(img1_np, cmap='twilight', origin='upper')
    quiver_high = axes[1].quiver(
        xx_sub, yy_sub, u_high[::stride, ::stride], v_high[::stride, ::stride], mag_high[::stride, ::stride],
        cmap='jet', angles='xy', scale_units='xy', scale=scale_factor,
        norm=plt.Normalize(vmin=global_mag_min, vmax=global_mag_max)
    )
    axes[1].set_title(f"Scaled Production Inference Depth (iters={production_iters})\n[High-Precision Physics Refinement]", fontsize=12)
    axes[1].axis('off')

    # Add uniform colorbars
    fig.colorbar(quiver_low, ax=axes[0], orientation='horizontal', pad=0.03, label='Velocity Magnitude (Pixels/Frame)')
    fig.colorbar(quiver_high, ax=axes[1], orientation='horizontal', pad=0.03, label='Velocity Magnitude (Pixels/Frame)')

    plt.suptitle("Inference Iteration Scaling Comparison on Unseen Validation Timesteps", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_netcdf_verification(nc_filepath, title="NetCDF4 Export Verification (Institutional Spec)"):
    """
    Loads an exported NetCDF4 file using xarray and plots Cloud-Top Pressure and Wind Velocity Magnitude.
    
    Args:
        nc_filepath (str): Path to the NetCDF file to open and verify.
        title (str): Title of the plot window.
    """
    # Load the dataset
    ds = xr.open_dataset(nc_filepath)

    # Set up side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left Panel: Cloud Top Pressure Mapping
    ds.cloud_top_pressure.plot(ax=axes[0], cmap="viridis_r")
    axes[0].set_title("Exported Cloud-Top Vertical Pressure (P)")
    axes[0].axis("off")

    # Right Panel: Velocity Magnitude Reconstruction
    magnitude = (ds.u_wind**2 + ds.v_wind**2)**0.5
    magnitude.plot(ax=axes[1], cmap="jet")
    axes[1].set_title("Exported Wind Velocity Magnitude")
    axes[1].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
