"""
Multi-modality 3D CNN from Zou et al. 2017 (Fig. 4, Section IV-D).

Two parallel SingleModal3DCNN branches (one fMRI, one sMRI), each
producing a 512-dim feature vector. These are concatenated → 1024 dims
→ FC(2) → logits.

  fMRI input (1, 47, 60, 46)  → fMRI branch  → 512-dim ─┐
                                                           → cat(1024) → Linear(2)
  sMRI input (1, 90, 117, 100) → sMRI branch → 512-dim ─┘
"""

import torch
import torch.nn as nn

from models.single_modal_3d import fmri_cnn, smri_cnn


class MultiModal3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmri_branch = fmri_cnn()
        self.smri_branch = smri_cnn()
        self.classifier  = nn.Linear(1024, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, fmri: torch.Tensor, smri: torch.Tensor) -> torch.Tensor:
        f_feat = self.fmri_branch.get_features(fmri)  # (B, 512)
        s_feat = self.smri_branch.get_features(smri)  # (B, 512)
        combined = torch.cat([f_feat, s_feat], dim=1)  # (B, 1024)
        return self.classifier(combined)               # (B, 2) logits
