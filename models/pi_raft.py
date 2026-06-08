import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder
from .convgru import ConvGRUCell
from .pressure_head import PressureHead
from .correlation import AllPairsCorrelationVolume

class PhysicsInformedRAFT(nn.Module):
    """
    Physics-Informed Recurrent All-Pairs Field Transform (PI-RAFT) architecture.
    
    API Contract & Shape Documentation:
    -----------------------------------
    Target Multi-Frame Sequence Format:
        Input shape:  [B, T, C, H, W]  (where T is sequence length, e.g. 2 or 4)
        Output shape: [B, T-1, 2, H, W] (for horizontal displacement field [u, v])
        
    Current 2-Frame Prototype Interface:
        Inputs:
            image1 (torch.Tensor): [B, 1, H, W] - Satellite image frame at time t (dtype: torch.float32)
            image2 (torch.Tensor): [B, 1, H, W] - Satellite image frame at time t+1 (dtype: torch.float32)
            dem (torch.Tensor):    [B, 1, H, W] - Topography Digital Elevation Model (dtype: torch.float32)
            iters (int):           Number of recurrent refinement iterations (default: 4)
        Outputs:
            final_flow (torch.Tensor):   [B, 2, H, W] - Spatial wind displacement field [u, v] (dtype: torch.float32)
            final_height (torch.Tensor): [B, 1, H, W] - Cloud-top vertical pressure placement in hPa (dtype: torch.float32)
    """
    def __init__(self, hidden_dim=128, corr_radius=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.corr_radius = corr_radius

        self.feature_encoder = FeatureEncoder(output_dim=128)
        self.context_encoder = FeatureEncoder(output_dim=128)
        self.update_block = ConvGRUCell(hidden_dim=128, input_dim=49 + 2)

        self.flow_head = nn.Conv2d(128, 2, kernel_size=3, padding=1)   # Outputs: [u, v]
        self.height_head = PressureHead(128, 1, kernel_size=3, padding=1) # Outputs: Pressure Height

    def forward(self, images, dem, iters=4):
        b, t, c, h, w = images.shape

        # Feature extraction for every frame in the sequence
        features = [self.feature_encoder(images[:, frame_idx]) for frame_idx in range(t)]

        # Use the first frame and DEM as the initial context reference
        c1 = self.context_encoder(images[:, 0] + dem)
        hidden_state = torch.tanh(c1)

        flows = []
        heights = []
        for frame_idx in range(t - 1):
            f1 = features[frame_idx]
            f2 = features[frame_idx + 1]
            corr_volume = AllPairsCorrelationVolume(f1, f2)

            coords0 = torch.zeros(b, 2, h // 8, w // 8, device=images.device)
            coords1 = torch.zeros(b, 2, h // 8, w // 8, device=images.device)

            for _ in range(iters):
                current_flow = coords1 - coords0
                corr_features = corr_volume.lookup(coords1, radius=self.corr_radius)

                x = torch.cat([corr_features, current_flow], dim=1)
                hidden_state = self.update_block(hidden_state, x)

                delta_flow = self.flow_head(hidden_state)
                coords1 = coords1 + delta_flow

            flow = F.interpolate(coords1 - coords0, size=(h, w), mode='bilinear', align_corners=True) * 8.0
            height = F.interpolate(self.height_head(hidden_state), size=(h, w), mode='bilinear', align_corners=True)

            flows.append(flow)
            heights.append(height)

        return torch.stack(flows, dim=1), torch.stack(heights, dim=1)
