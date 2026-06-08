import torch
import torch.nn.functional as F

class AllPairsCorrelationVolume:
    """
    Computes a local all-pairs correlation feature map between two encoded frames.
    """
    def __init__(self, feature1, feature2):
        self.feature1 = feature1
        self.feature2 = feature2
        self.scale = torch.sqrt(torch.tensor(feature1.shape[1], dtype=torch.float32, device=feature1.device))

    def lookup(self, coords, radius=3):
        """
        Extracts a local correlation patch around each pixel for the current flow estimate.
        """
        _, _, h, w = self.feature1.shape
        pad = radius
        padded_f2 = F.pad(self.feature2, (pad, pad, pad, pad), mode='replicate')

        correlations = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = padded_f2[:, :, pad + dy : pad + dy + h, pad + dx : pad + dx + w]
                corr = torch.sum(self.feature1 * shifted, dim=1, keepdim=True) / self.scale
                correlations.append(corr)

        return torch.cat(correlations, dim=1)
