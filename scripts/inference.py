import os
import argparse
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

# Import modular package modules
from configs import load_config
from datasets import SatelliteDataset
from models import PhysicsInformedRAFT
from evaluation import plot_inference_scaling_comparison, plot_netcdf_verification
from export import export_amv_netcdf

def run_inference(config_path):
    # Load configuration
    cfg = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device context: {device}")

    # Set up validation data stream
    print("Initializing validation satellite data stream...")
    val_datasets = [
        SatelliteDataset(
            satellite=cfg['dataset']['satellite'],
            product=cfg['dataset']['product'],
            band=cfg['dataset']['band'],
            year=cfg['dataset']['year'],
            day_of_year=cfg['dataset']['day_of_year'],
            hour=h,
            sequence_length=cfg['dataset'].get('sequence_length', 2)
        )
        for h in cfg['dataset']['val_hours']
    ]
    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=False)
    print(f"Total validation tracking frames compiled: {len(val_dataset)}")

    # Instantiate model
    model = PhysicsInformedRAFT(
        hidden_dim=cfg['model']['hidden_dim'],
        corr_radius=cfg['model']['corr_radius']
    ).to(device)

    # Load weights
    checkpoint_path = cfg['model']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint path '{checkpoint_path}' not found! Running with untrained random weights.")

    model.eval()

    # Get a fresh sample batch from the validation datastream
    images, dem = next(iter(val_loader))
    images, dem = images.to(device), dem.to(device)

    # 1. Compute inference at standard training depth (iters=4)
    standard_iters = cfg['inference']['standard_iters']
    print(f"Running standard inference depth (iters={standard_iters})...")
    with torch.no_grad():
        flow_low_res, _ = model(images, dem, iters=standard_iters)

    # 2. Compute inference at scaled production depth (iters=12)
    production_iters = cfg['inference']['production_iters']
    print(f"Running scaled production inference depth (iters={production_iters})...")
    with torch.no_grad():
        flow_high_res, height_pred = model(images, dem, iters=production_iters)

    # Average adjacent-pair flows into a final single exported flow field
    flow_low_res = flow_low_res.mean(dim=1)
    flow_high_res = flow_high_res.mean(dim=1)
    height_pred = height_pred.mean(dim=1)

    # 3. Visualize comparisons if enabled
    if cfg['inference']['visualize']:
        print("Rendering iteration scaling comparison...")
        settings = cfg['inference']['visualization_settings']
        plot_inference_scaling_comparison(
            images[:, 0],
            flow_low_res,
            flow_high_res,
            standard_iters=standard_iters,
            production_iters=production_iters,
            scale_factor=settings['scale_factor'],
            stride=settings['stride']
        )

    # 4. Export results to CF-Compliant NetCDF4
    u_array = flow_high_res.squeeze(0)[0].cpu().numpy()
    v_array = flow_high_res.squeeze(0)[1].cpu().numpy()
    p_array = height_pred.squeeze(0)[0].cpu().numpy()

    output_dir = cfg['export']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    nc_filepath = os.path.join(output_dir, cfg['export']['filename'])
    
    metadata = {
        'title': cfg['export']['title'],
        'institution': cfg['export']['institution'],
        'source': cfg['export']['source'],
        'history': cfg['export']['history']
    }

    export_amv_netcdf(u_array, v_array, p_array, nc_filepath, metadata=metadata)

    # 5. Load and verify exported NetCDF file
    if cfg['inference']['visualize']:
        print("Verifying exported NetCDF dataset via xarray visualization...")
        plot_netcdf_verification(nc_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PI-RAFT Model and Export Winds")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to config file")
    args = parser.parse_args()
    run_inference(args.config)
