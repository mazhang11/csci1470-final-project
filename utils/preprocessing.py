import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch

# Target crop shapes from SPEC
FMRI_CROP = (47, 60, 46)    # ReHo, fALFF
SMRI_CROP = (90, 117, 100)  # GM

FMRI_DERIVATIVES = {"reho", "falff"}
SMRI_DERIVATIVES = {"gm"}


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def center_crop(volume: np.ndarray, target: tuple) -> np.ndarray:
    pad_width = []
    for c, t in zip(volume.shape, target):
        deficit = max(0, t - c)
        pad_width.append((deficit // 2, deficit - deficit // 2))
    
    if any(p != (0, 0) for p in pad_width):
        volume = np.pad(volume, pad_width, mode="constant")
    
    starts = [(c - t) // 2 for c, t in zip(volume.shape, target)]
    slices = tuple(slice(s, s + t) for s, t in zip(starts, target))
    return volume[slices]


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    std = volume.std()
    if std < 1e-8:
        return volume - volume.mean()
    return (volume - volume.mean()) / std


def get_crop_shape(derivative: str) -> tuple:
    d = derivative.lower()
    if d in FMRI_DERIVATIVES:
        return FMRI_CROP
    if d in SMRI_DERIVATIVES:
        return SMRI_CROP
    raise ValueError(
        f"Unknown derivative '{derivative}'. "
        f"Expected one of {FMRI_DERIVATIVES | SMRI_DERIVATIVES}"
    )


def preprocess_volume(path: str, derivative: str) -> torch.Tensor:
    """
    Load a NIfTI file, pad if needed, center-crop to spec dims,
    z-score normalize, return float32 tensor of shape (1, D, H, W).
    """
    volume = load_nifti(path)
    target = get_crop_shape(derivative)

    pad_width = []
    for c, t in zip(volume.shape, target):
        deficit = max(0, t - c)
        pad_width.append((deficit // 2, deficit - deficit // 2))
    if any(p != (0, 0) for p in pad_width):
        volume = np.pad(volume, pad_width, mode="constant")

    volume = center_crop(volume, target)
    volume = zscore_normalize(volume)
    return torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)


def load_phenotypic(data_dir: str) -> pd.DataFrame:
    """
    Load the phenotypic.csv written by download_adhd200.py.
    Returns a DataFrame with at minimum: subject_id, label (0=ADHD, 1=TDC).
    """
    path = os.path.join(data_dir, "phenotypic.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"phenotypic.csv not found in {data_dir}. "
            "Run scripts/run_downloads.sh first."
        )
    return pd.read_csv(path)


def build_subject_file_map(data_dir: str,
                           derivatives: list[str] | None = None
                           ) -> dict[str, dict[str, str]]:
    """
    Scan the flat per-subject download layout and return:
        { subject_id: { derivative: absolute_path } }

    data_dir    — root download dir, e.g. data/raw
    derivatives — subset to include; defaults to ['falff', 'reho', 'gm']

    Expected layout on disk:
        data_dir/<subject_id>/falff.nii.gz
        data_dir/<subject_id>/reho.nii.gz
        data_dir/<subject_id>/gm.nii.gz
    """
    if derivatives is None:
        derivatives = ["falff", "reho", "gm"]

    subject_map: dict[str, dict[str, str]] = {}

    for entry in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(subject_dir):
            continue
        paths = {}
        for deriv in derivatives:
            fpath = os.path.join(subject_dir, f"{deriv}.nii.gz")
            if os.path.exists(fpath):
                paths[deriv] = os.path.abspath(fpath)
        if paths:
            subject_map[entry] = paths

    return subject_map
