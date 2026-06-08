import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

# Import modular package classes
from configs import load_config
from datasets import SatelliteDataset
from models import PhysicsInformedRAFT
from losses import PhysicsInformedLoss

def save_checkpoint(epoch_num, model, optimizer, criterion, metrics, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Move optimizer state to CPU before saving to prevent device-locked checkpoints
    optimizer_state_on_cpu = {}
    for k, v in optimizer.state_dict().items():
        if isinstance(v, dict):
            optimizer_state_on_cpu[k] = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    optimizer_state_on_cpu[k][sub_k] = sub_v.cpu()
                else:
                    optimizer_state_on_cpu[k][sub_k] = sub_v
        elif isinstance(v, torch.Tensor):
            optimizer_state_on_cpu[k] = v.cpu()
        else:
            optimizer_state_on_cpu[k] = v

    # Build checkpoint payload matching baseline structure
    checkpoint_payload = {
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state_on_cpu,
        'final_val_loss': metrics['final_val_loss'],
        'hyperparameters': {
            'alpha': criterion.alpha,
            'beta': criterion.beta,
            'gamma': criterion.gamma
        }
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"PI_RAFT_epoch_{epoch_num}.pth")
    torch.save(checkpoint_payload, checkpoint_path)
    print(f"Successfully checkpointed model state to disk container: {checkpoint_path}")

def run_training(config_path):
    # Load configuration
    cfg = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running pipeline on device context: {device}")

    # Initialize datasets and loaders
    print("Initializing multi-hour satellite dataset streams...")
    hourly_datasets = [
        SatelliteDataset(
            satellite=cfg['dataset']['satellite'],
            product=cfg['dataset']['product'],
            band=cfg['dataset']['band'],
            year=cfg['dataset']['year'],
            day_of_year=cfg['dataset']['day_of_year'],
            hour=h,
            sequence_length=cfg['dataset'].get('sequence_length', 2)
        )
        for h in cfg['dataset']['train_hours']
    ]
    
    sequence_dataset = ConcatDataset(hourly_datasets)
    print(f"Total available tracking frames in sequence pool: {len(sequence_dataset)}")
    
    # Batch size is typically 1 in baseline
    sequence_loader = DataLoader(sequence_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=False)

    print("Initializing validation satellite data stream...")
    # Hour 18 provides a completely fresh window of convective dynamics to test against
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

    # Instantiate model, loss, and optimizer
    model = PhysicsInformedRAFT(
        hidden_dim=cfg['model']['hidden_dim'],
        corr_radius=cfg['model']['corr_radius']
    ).to(device)
    
    criterion = PhysicsInformedLoss(
        alpha=cfg['loss']['alpha'],
        beta=cfg['loss']['beta'],
        gamma=cfg['loss']['gamma'],
        epsilon=cfg['loss']['epsilon']
    )

    print(f"Initializing {cfg['optimizer']['type']} Optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'])

    epochs = cfg['optimizer']['epochs']
    step_counter = 0

    for epoch in range(1, epochs + 1):
        # ==================== TRAINING PHASE ====================
        model.train()
        print(f"\n--- Starting Training Epoch {epoch}/{epochs} ---")

        for batch_idx, (images, dem) in enumerate(sequence_loader):
            images, dem = images.to(device), dem.to(device)

            optimizer.zero_grad()
            flow_pred, height_pred = model(images, dem, iters=cfg['model']['train_iters'])
            total_loss, data_loss, physics_loss = criterion(flow_pred, height_pred, images)

            total_loss.backward()
            optimizer.step()

            step_counter += 1
            if step_counter % cfg['logging']['log_interval'] == 0:
                print(
                    f"Step: {step_counter:03d} | Batch: {batch_idx+1}/{len(sequence_loader)} | "
                    f"TRAIN TOTAL: {total_loss.item():.4f} ---> "
                    f"[Data: {data_loss.item():.4f} | Physics: {physics_loss.item():.4f}]"
                )

        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_total, val_data, val_physics = 0.0, 0.0, 0.0

        print(f"\n--- Running Validation Eval for Epoch {epoch} ---")
        with torch.no_grad():
            for images_v, dem_v in val_loader:
                images_v, dem_v = images_v.to(device), dem_v.to(device)

                flow_pred_v, height_pred_v = model(images_v, dem_v, iters=cfg['model']['train_iters'])
                v_total, v_data, v_physics = criterion(flow_pred_v, height_pred_v, images_v)

                val_total += v_total.item()
                val_data += v_data.item()
                val_physics += v_physics.item()

        # Calculate dataset-wide average scores
        num_val_batches = len(val_loader)
        avg_val_total_loss = val_total / num_val_batches
        print(f"====== VAL METRICS EPOCH {epoch} ======")
        print(f"Avg VAL Total Loss:     {avg_val_total_loss:.4f}")
        print(f"Avg VAL Data Domain:    {val_data / num_val_batches:.4f}")
        print(f"Avg VAL Physics Domain: {val_physics / num_val_batches:.4f}")
        print("======================================\n")

        # Save checkpoint
        val_metrics = {'final_val_loss': avg_val_total_loss}
        save_checkpoint(epoch, model, optimizer, criterion, val_metrics, cfg['logging']['checkpoint_dir'])

    print("Multi-epoch training and verification pipeline execution complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PI-RAFT Model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config file")
    args = parser.parse_args()
    run_training(args.config)
