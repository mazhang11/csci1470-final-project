"""
visualize_mri.py

Generates brain MRI visualization figures for the project report:

  fig_mri_1  — Axial/coronal/sagittal slice montage for one subject (all 3 derivatives)
  fig_mri_2  — Side-by-side ADHD vs Control mean maps (fALFF & ReHo)
  fig_mri_3  — GM maps: ADHD vs Control mean
  fig_mri_4  — Group difference map (ADHD - Control) for each derivative
  fig_mri_5  — ABIDE: ASD vs TDC mean fALFF map
  fig_mri_6  — Dataset overview: class balance + age/sex distributions

Usage:
  python scripts/visualize_mri.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nibabel as nib
import pandas as pd

ROOT      = Path(__file__).parent.parent
ADHD_RAW  = ROOT / 'data' / 'raw'
ABIDE_RAW = ROOT / 'data' / 'raw_abide'
FIG_DIR   = ROOT / 'outputs_oscar' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vol(path):
    """Load NIfTI, return float32 numpy array."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def zscore(vol):
    """Z-score normalize, masking out near-zero voxels."""
    mask = np.abs(vol) > 1e-6
    if mask.sum() == 0:
        return vol
    mu, sigma = vol[mask].mean(), vol[mask].std()
    if sigma < 1e-9:
        return vol
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mu) / sigma
    return out


def mid_slices(vol):
    """Return (axial, coronal, sagittal) mid-plane slices."""
    x, y, z = vol.shape
    return vol[x//2, :, :], vol[:, y//2, :], vol[:, :, z//2]


def load_phenotypic(data_dir):
    path = Path(data_dir) / 'phenotypic.csv'
    return pd.read_csv(path)


def subject_dirs(data_dir):
    return sorted([p for p in Path(data_dir).iterdir()
                   if p.is_dir() and (p / 'falff.nii.gz').exists()])


def mean_map(dirs, derivative):
    """Average z-scored volumes across a list of subject directories.
    Crops all volumes to the smallest common shape first."""
    vols = []
    for d in dirs:
        p = d / f'{derivative}.nii.gz'
        if p.exists():
            vols.append(zscore(load_vol(p)))
    if not vols:
        return None
    # Crop to minimum shape along each axis so all arrays are the same size.
    min_shape = tuple(min(v.shape[i] for v in vols) for i in range(3))
    cropped = [v[:min_shape[0], :min_shape[1], :min_shape[2]] for v in vols]
    return np.mean(cropped, axis=0)


# ---------------------------------------------------------------------------
# Figure 1 — Single subject: 3 derivatives × 3 planes
# ---------------------------------------------------------------------------

def fig_mri_1_single_subject():
    dirs = subject_dirs(ADHD_RAW)
    if not dirs:
        print('Fig MRI 1: no ADHD data')
        return
    sid = dirs[0]
    derivatives = ['falff', 'reho', 'gm']
    planes      = ['Axial', 'Coronal', 'Sagittal']

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row, deriv in enumerate(derivatives):
        p = sid / f'{deriv}.nii.gz'
        if not p.exists():
            continue
        vol = zscore(load_vol(p))
        slices = mid_slices(vol)
        for col, (sl, plane) in enumerate(zip(slices, planes)):
            ax = axes[row, col]
            ax.imshow(sl.T, cmap='gray', origin='lower', aspect='auto')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(deriv.upper(), fontsize=13, fontweight='bold')
            if row == 0:
                ax.set_title(plane, fontsize=13)

    fig.suptitle(f'ADHD-200 Subject {sid.name} — z-scored mid-plane slices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_1_single_subject.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 2 — Group mean fALFF & ReHo: ADHD vs Control
# ---------------------------------------------------------------------------

def fig_mri_2_group_means_fmri():
    pheno = load_phenotypic(ADHD_RAW)
    # label 0 = ADHD, 1 = Control
    adhd_ids = set(str(s) for s in pheno[pheno['label'] == 0]['subject_id'])
    ctrl_ids  = set(str(s) for s in pheno[pheno['label'] == 1]['subject_id'])

    all_dirs  = subject_dirs(ADHD_RAW)
    adhd_dirs = [d for d in all_dirs if d.name in adhd_ids]
    ctrl_dirs = [d for d in all_dirs if d.name in ctrl_ids]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    titles = ['ADHD mean', 'Control mean', 'ADHD mean', 'Control mean']
    row_labels = ['fALFF', 'ReHo']

    for row, deriv in enumerate(['falff', 'reho']):
        adhd_mean = mean_map(adhd_dirs, deriv)
        ctrl_mean = mean_map(ctrl_dirs, deriv)
        if adhd_mean is None or ctrl_mean is None:
            continue
        for col, (label, vol) in enumerate([('ADHD mean', adhd_mean),
                                             ('Control mean', ctrl_mean),
                                             ('ADHD mean', adhd_mean),
                                             ('Control mean', ctrl_mean)]):
            ax = axes[row, col]
            # alternate axial / sagittal
            sl = vol[vol.shape[0]//2, :, :] if col < 2 else vol[:, :, vol.shape[2]//2]
            vmax = np.percentile(np.abs(vol), 97)
            im = ax.imshow(sl.T, cmap='RdBu_r', origin='lower',
                           aspect='auto', vmin=-vmax, vmax=vmax)
            ax.axis('off')
            plane = 'Axial' if col < 2 else 'Sagittal'
            ax.set_title(f'{label}\n({plane})', fontsize=10)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=12, fontweight='bold')

    fig.suptitle('ADHD-200 — Group Mean fMRI Maps (z-scored)', fontsize=14, fontweight='bold')
    plt.colorbar(axes[0,0].get_images()[0], ax=axes[0,:].tolist(), shrink=0.6, label='z-score')
    plt.colorbar(axes[1,0].get_images()[0], ax=axes[1,:].tolist(), shrink=0.6, label='z-score')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_2_group_means_fmri.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 3 — Group mean GM: ADHD vs Control
# ---------------------------------------------------------------------------

def fig_mri_3_group_means_gm():
    pheno = load_phenotypic(ADHD_RAW)
    adhd_ids = set(str(s) for s in pheno[pheno['label'] == 0]['subject_id'])
    ctrl_ids  = set(str(s) for s in pheno[pheno['label'] == 1]['subject_id'])
    all_dirs  = subject_dirs(ADHD_RAW)
    adhd_dirs = [d for d in all_dirs if d.name in adhd_ids]
    ctrl_dirs = [d for d in all_dirs if d.name in ctrl_ids]

    adhd_mean = mean_map(adhd_dirs, 'gm')
    ctrl_mean = mean_map(ctrl_dirs, 'gm')
    if adhd_mean is None:
        print('Fig MRI 3: no GM data')
        return

    planes = [
        ('Axial',    lambda v: v[v.shape[0]//2, :, :]),
        ('Coronal',  lambda v: v[:, v.shape[1]//2, :]),
        ('Sagittal', lambda v: v[:, :, v.shape[2]//2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, (group, vol) in enumerate([('ADHD', adhd_mean), ('Control', ctrl_mean)]):
        vmax = np.percentile(np.abs(vol), 97)
        for col, (plane_name, slicer) in enumerate(planes):
            ax = axes[row, col]
            sl = slicer(vol)
            ax.imshow(sl.T, cmap='hot', origin='lower', aspect='auto',
                      vmin=0, vmax=vmax)
            ax.axis('off')
            if row == 0:
                ax.set_title(plane_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(group, fontsize=13, fontweight='bold')

    fig.suptitle('ADHD-200 — Grey Matter Density: ADHD vs Control', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_3_group_means_gm.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 4 — Difference maps (ADHD - Control)
# ---------------------------------------------------------------------------

def fig_mri_4_difference_maps():
    pheno = load_phenotypic(ADHD_RAW)
    adhd_ids = set(str(s) for s in pheno[pheno['label'] == 0]['subject_id'])
    ctrl_ids  = set(str(s) for s in pheno[pheno['label'] == 1]['subject_id'])
    all_dirs  = subject_dirs(ADHD_RAW)
    adhd_dirs = [d for d in all_dirs if d.name in adhd_ids]
    ctrl_dirs = [d for d in all_dirs if d.name in ctrl_ids]

    derivatives = ['falff', 'reho', 'gm']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, deriv in zip(axes, derivatives):
        adhd_mean = mean_map(adhd_dirs, deriv)
        ctrl_mean = mean_map(ctrl_dirs, deriv)
        if adhd_mean is None or ctrl_mean is None:
            ax.set_title(f'{deriv.upper()} — no data')
            continue
        diff = adhd_mean - ctrl_mean
        sl   = diff[diff.shape[0]//2, :, :]   # axial mid-slice
        vmax = np.percentile(np.abs(diff), 98)
        im = ax.imshow(sl.T, cmap='RdBu_r', origin='lower',
                       aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{deriv.upper()}\n(ADHD − Control)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Δ z-score')

    fig.suptitle('ADHD-200 — Group Difference Maps (axial mid-slice)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_4_difference_maps.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 5 — ABIDE: ASD vs TDC mean fALFF (axial strips)
# ---------------------------------------------------------------------------

def fig_mri_5_abide_group_means():
    pheno_path = ABIDE_RAW / 'phenotypic.csv'
    if not pheno_path.exists():
        print('Fig MRI 5: no ABIDE phenotypic.csv')
        return

    pheno   = pd.read_csv(pheno_path)
    asd_ids = set(str(s) for s in pheno[pheno['label'] == 0]['subject_id'])
    tdc_ids = set(str(s) for s in pheno[pheno['label'] == 1]['subject_id'])

    all_dirs = sorted([p for p in ABIDE_RAW.iterdir()
                       if p.is_dir() and (p / 'falff.nii.gz').exists()])
    asd_dirs = [d for d in all_dirs if d.name in asd_ids]
    tdc_dirs = [d for d in all_dirs if d.name in tdc_ids]

    asd_mean = mean_map(asd_dirs, 'falff')
    tdc_mean = mean_map(tdc_dirs, 'falff')

    # Build list of available groups (may only have ASD locally)
    groups = []
    if asd_mean is not None:
        groups.append(('ASD', asd_mean))
    if tdc_mean is not None:
        groups.append(('TDC', tdc_mean))
    if not groups:
        print('Fig MRI 5: no ABIDE fALFF data')
        return

    n_slices = 5
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, n_slices, figsize=(14, 4 * n_groups),
                             squeeze=False)

    all_vals = np.concatenate([g[1].ravel() for g in groups])
    vmax = np.percentile(np.abs(all_vals), 97)
    z_indices = np.linspace(5, groups[0][1].shape[2] - 6, n_slices, dtype=int)

    for row, (group, vol) in enumerate(groups):
        for col, z in enumerate(z_indices):
            ax = axes[row, col]
            ax.imshow(vol[:, :, z].T, cmap='RdBu_r', origin='lower',
                      aspect='auto', vmin=-vmax, vmax=vmax)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(group, fontsize=13, fontweight='bold')
            if row == 0:
                ax.set_title(f'z={z}', fontsize=9)

    note = '' if tdc_mean is not None else ' (ASD only — TDC not downloaded locally)'
    fig.suptitle(f'ABIDE — Mean fALFF: axial slices, z-scored{note}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_5_abide_group_means.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 6 — Dataset overview: class balance + age/sex distributions
# ---------------------------------------------------------------------------

def fig_mri_6_dataset_overview():
    adhd_pheno  = load_phenotypic(ADHD_RAW)
    abide_pheno = pd.read_csv(ABIDE_RAW / 'phenotypic.csv') if (ABIDE_RAW / 'phenotypic.csv').exists() else None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # --- ADHD-200 class balance ---
    ax = axes[0]
    counts = adhd_pheno['label'].value_counts().sort_index()
    bars = ax.bar(['ADHD', 'Control'], [counts.get(0, 0), counts.get(1, 0)],
                  color=['steelblue', 'salmon'])
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                str(int(b.get_height())), ha='center', fontweight='bold')
    ax.set_title('ADHD-200\nClass Balance', fontsize=12)
    ax.set_ylabel('Subjects')
    ax.grid(axis='y', alpha=0.3)

    # --- ADHD-200 age distribution ---
    ax = axes[1]
    for label, name, color in [(0, 'ADHD', 'steelblue'), (1, 'Control', 'salmon')]:
        ages = adhd_pheno[adhd_pheno['label'] == label]['age'].dropna()
        ax.hist(ages, bins=12, alpha=0.6, label=name, color=color)
    ax.set_title('ADHD-200\nAge Distribution', fontsize=12)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- ABIDE class balance ---
    ax = axes[2]
    if abide_pheno is not None:
        counts = abide_pheno['label'].value_counts().sort_index()
        bars = ax.bar(['ASD', 'TDC'], [counts.get(0, 0), counts.get(1, 0)],
                      color=['mediumpurple', 'mediumseagreen'])
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    str(int(b.get_height())), ha='center', fontweight='bold')
        ax.set_title('ABIDE\nClass Balance', fontsize=12)
        ax.set_ylabel('Subjects')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.set_title('ABIDE — no data')

    # --- ADHD-200 sex distribution per group ---
    ax = axes[3]
    sex_map = {0.0: 'F', 1.0: 'M'}
    for label, grp_name, color in [(0, 'ADHD', 'steelblue'), (1, 'Control', 'salmon')]:
        sub = adhd_pheno[adhd_pheno['label'] == label]
        male   = (sub['sex'] == 1.0).sum()
        female = (sub['sex'] == 0.0).sum()
        offset = -0.2 if label == 0 else 0.2
        x = np.array([0, 1])
        ax.bar(x + offset, [male, female], 0.35, label=grp_name, color=color, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_title('ADHD-200\nSex by Group', fontsize=12)
    ax.set_ylabel('Subjects')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Dataset Overview — ADHD-200 & ABIDE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'fig_mri_6_dataset_overview.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Generating MRI visualization figures...\n')
    fig_mri_1_single_subject()
    fig_mri_2_group_means_fmri()
    fig_mri_3_group_means_gm()
    fig_mri_4_difference_maps()
    fig_mri_5_abide_group_means()
    fig_mri_6_dataset_overview()
    print(f'\nAll figures saved to {FIG_DIR}')
