import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.preprocessing import build_subject_file_map, load_phenotypic, preprocess_volume


class ADHDDataset(Dataset):
    """
    Single-modality dataset. Returns (volume_tensor, label) per subject.

    volume_tensor: float32 tensor of shape (1, D, H, W)
    label:         int — 0=ADHD, 1=TDC
    """

    def __init__(self, subject_ids: list[str], file_map: dict, labels: dict,
                 derivative: str, transform=None):
        """
        subject_ids — ordered list of subject IDs to include
        file_map    — { subject_id: { derivative: path } }
        labels      — { subject_id: int }
        derivative  — which feature to load: 'falff', 'reho', or 'gm'
        transform   — optional callable applied to the tensor after preprocessing
        """
        self.derivative = derivative
        self.transform = transform
        self.samples = [
            (file_map[sid][derivative], labels[sid])
            for sid in subject_ids
            if sid in file_map and derivative in file_map[sid]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = preprocess_volume(path, self.derivative)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


class ADHDMultiModalDataset(Dataset):
    """
    Multi-modality dataset for the dual-branch CNN.
    Returns (fmri_tensor, smri_tensor, label) per subject.

    fmri_tensor: shape (1, 47, 60, 46)  — fALFF or ReHo
    smri_tensor: shape (1, 90, 117, 100) — GM
    label:       int — 0=ADHD, 1=TDC
    """

    def __init__(self, subject_ids: list[str], file_map: dict, labels: dict,
                 fmri_derivative: str = "falff", smri_derivative: str = "gm",
                 transform=None):
        self.fmri_derivative = fmri_derivative
        self.smri_derivative = smri_derivative
        self.transform = transform
        self.samples = [
            (file_map[sid][fmri_derivative], file_map[sid][smri_derivative], labels[sid])
            for sid in subject_ids
            if sid in file_map
            and fmri_derivative in file_map[sid]
            and smri_derivative in file_map[sid]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fmri_path, smri_path, label = self.samples[idx]
        fmri = preprocess_volume(fmri_path, self.fmri_derivative)
        smri = preprocess_volume(smri_path, self.smri_derivative)
        if self.transform:
            fmri = self.transform(fmri)
            smri = self.transform(smri)
        return fmri, smri, label


def build_datasets(data_dir: str,
                   derivative: str = "falff",
                   fmri_derivative: str = "falff",
                   smri_derivative: str = "gm",
                   multi_modal: bool = False,
                   subject_ids: list[str] | None = None,
                   transform=None):
    """
    Convenience function: loads phenotypic CSV and file map, returns a
    single Dataset ready to hand to a DataLoader or cross-validation splitter.

    data_dir      — root download dir (contains phenotypic.csv + subject dirs)
    derivative    — for single-modal mode
    fmri/smri_*   — for multi-modal mode
    multi_modal   — if True, returns ADHDMultiModalDataset
    subject_ids   — optional subset; defaults to all subjects in phenotypic.csv
    """
    pheno = load_phenotypic(data_dir)
    file_map = build_subject_file_map(data_dir)

    labels = dict(zip(pheno["subject_id"].astype(str), pheno["label"]))

    if subject_ids is None:
        subject_ids = [
            sid for sid in pheno["subject_id"].astype(str).tolist()
            if sid in file_map
        ]

    if multi_modal:
        return ADHDMultiModalDataset(
            subject_ids, file_map, labels,
            fmri_derivative=fmri_derivative,
            smri_derivative=smri_derivative,
            transform=transform,
        )
    return ADHDDataset(
        subject_ids, file_map, labels,
        derivative=derivative,
        transform=transform,
    )
