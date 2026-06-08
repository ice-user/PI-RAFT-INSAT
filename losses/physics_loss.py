import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.01, epsilon=0.001):
        """
        Physics-Informed Loss Engine for High-Resolution Atmospheric Motion Vector Retrieval.

        Args:
            alpha (float): Regularization weight for Fluid Smoothness (SC).
            beta (float): Optimization weight for Constancy Gradient (GC).
            gamma (float): Baseline anchor weight for Hinting Background (E_W).
            epsilon (float): Structural outlier dampening constant.
        """
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def warp(self, img, flow):
        """
        Executes backward warping via differentiable bilinear interpolation to calculate I(x+U).
        Maps Frame t+1 features back into the tracking frame coordinates of Frame t.
        """
        B, C, H, W = img.size()

        # Build 2D coordinate space meshgrid
        yy, xx = torch.meshgrid(
            torch.arange(0, H, device=img.device, dtype=torch.float32),
            torch.arange(0, W, device=img.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # Shape: [B, 2, H, W]
        vgrid = grid + flow # Apply spatial vector displacement field

        # Standardize matrix coordinates into PyTorch grid range [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1) # Shift shape configuration to [B, H, W, 2]

        return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_spatial_gradients(self, tensor):
        """Calculates discrete spatial differences across dimensions (X-axis and Y-axis)."""
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

        # Replicate borders to keep spatial sizing dimensions uniform
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        return dx, dy

    def apply_charbonnier(self, residual_tensor):
        """Wraps mathematical tracking outliers with the Charbonnier threshold formula."""
        return torch.sqrt(residual_tensor ** 2 + self.epsilon ** 2)

    def forward(self, flow_pred, height_pred, img_sequence, background_flow=None):
        """
        Evaluates fluid transport parameters against a temporal image sequence.

        Args:
            flow_pred (torch.Tensor): [B, T-1, 2, H, W]
            height_pred (torch.Tensor): [B, T-1, 1, H, W]
            img_sequence (torch.Tensor): [B, T, C, H, W]
        """
        # TODO: scientific validation required
        # Note: height_pred is currently ignored in the loss formulation.

        total_loss = 0.0
        total_data_loss = 0.0
        total_smoothness_loss = 0.0

        for t in range(img_sequence.shape[1] - 1):
            img1 = img_sequence[:, t]
            img2 = img_sequence[:, t + 1]
            flow = flow_pred[:, t]

            # 1. Brightness Constancy: warp the later frame back to the reference frame
            img2_warped = self.warp(img2, flow)
            bc_residual = img2_warped - img1
            loss_bc = torch.mean(self.apply_charbonnier(bc_residual))

            # 2. Spatial Gradient Constancy: compare image gradients after warping
            img1_dx, img1_dy = self.compute_spatial_gradients(img1)
            warped_dx, warped_dy = self.compute_spatial_gradients(img2_warped)
            gc_residual_x = warped_dx - img1_dx
            gc_residual_y = warped_dy - img1_dy
            loss_gc = torch.mean(self.apply_charbonnier(gc_residual_x) + self.apply_charbonnier(gc_residual_y))

            # 3. Fluid Smoothness Regularization: penalize abrupt flow changes
            u_channel, v_channel = flow[:, 0:1, :, :], flow[:, 1:2, :, :]
            u_dx, u_dy = self.compute_spatial_gradients(u_channel)
            v_dx, v_dy = self.compute_spatial_gradients(v_channel)
            loss_sc = torch.mean(
                self.apply_charbonnier(u_dx) + self.apply_charbonnier(u_dy) +
                self.apply_charbonnier(v_dx) + self.apply_charbonnier(v_dy)
            )

            # 4. Optional background guidance loss on predicted flow
            if background_flow is not None:
                bg_residual = flow - background_flow[:, t]
                loss_ew = torch.mean(self.apply_charbonnier(bg_residual))
            else:
                loss_ew = torch.tensor(0.0, device=flow_pred.device)

            step_loss = loss_bc + (self.beta * loss_gc) + (self.alpha * loss_sc) + (self.gamma * loss_ew)
            total_loss += step_loss
            total_data_loss += loss_bc + (self.beta * loss_gc)
            total_smoothness_loss += loss_sc

        num_pairs = img_sequence.shape[1] - 1
        total_loss = total_loss / num_pairs
        data_tracking_metric = total_data_loss / num_pairs
        fluid_smoothness_metric = total_smoothness_loss / num_pairs

        return total_loss, data_tracking_metric, fluid_smoothness_metric
