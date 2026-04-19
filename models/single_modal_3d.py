"""
Single-modality 3D CNN from Zou et al. 2017 (Fig. 4, Section IV-C).

Architecture (both fMRI and sMRI use the same structure, differing only
in the initial max-pool kernel):

  Input (1, D, H, W)
  → MaxPool(2×2×2 for fMRI  |  4×4×4 for sMRI)
  → C1: Conv3d(1→32, 5×5×5) + BN + ReLU
  → MaxPool(2×2×2)
  → C2: Conv3d(32→64, 3×3×3) + BN + ReLU
  → C3: Conv3d(64→64, 3×3×3) + BN + ReLU
  → C4: Conv3d(64→64, 3×3×3) + BN + ReLU
  → Flatten
  → F5: Linear(flat→512) + BN + ReLU + Dropout(0.5)
  → F6: Linear(512→2)          [logits — use CrossEntropyLoss]

Weight init: Xavier uniform on all Conv3d and Linear weights.
Dropout(0.5) applied at *inputs* of F5 and F6 per the paper.
"""

import torch
import torch.nn as nn


class SingleModal3DCNN(nn.Module):
    def __init__(self, input_shape: tuple, initial_pool_size: tuple):
        """
        input_shape       — (D, H, W) of the cropped volume, e.g. (47, 60, 46)
        initial_pool_size — (2,2,2) for fMRI features, (4,4,4) for sMRI features
        """
        super().__init__()

        self.initial_pool = nn.MaxPool3d(kernel_size=initial_pool_size)

        # Four convolutional layers (all with no padding, stride 1)
        self.conv_block = nn.Sequential(
            # C1
            nn.Conv3d(1, 32, kernel_size=5),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            # C2
            nn.Conv3d(32, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # C3
            nn.Conv3d(64, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # C4
            nn.Conv3d(64, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size by running a dummy forward pass
        flat_size = self._get_flat_size(input_shape)

        # F5 and F6: dropout applied at the *input* of each layer per the paper
        self.dropout = nn.Dropout(p=0.5)

        self.f5 = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.f6 = nn.Linear(512, 2)

        self._init_weights()

    def _get_flat_size(self, input_shape: tuple) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            dummy = self.initial_pool(dummy)
            dummy = self.conv_block(dummy)
            return dummy.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_pool(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)         # dropout at input of F5
        x = self.f5(x)
        x = self.dropout(x)         # dropout at input of F6
        x = self.f6(x)
        return x                    # raw logits for CrossEntropyLoss

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim feature vector (used by multi-modal branch)."""
        x = self.initial_pool(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.f5(x)
        return x


# Convenience constructors matching the paper's two input types
def fmri_cnn() -> SingleModal3DCNN:
    """fMRI branch: input 47×60×46, initial pool 2×2×2."""
    return SingleModal3DCNN(input_shape=(47, 60, 46), initial_pool_size=(2, 2, 2))


def smri_cnn() -> SingleModal3DCNN:
    """sMRI branch: input 90×117×100, initial pool 4×4×4."""
    return SingleModal3DCNN(input_shape=(90, 117, 100), initial_pool_size=(4, 4, 4))
