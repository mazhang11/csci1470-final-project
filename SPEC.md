# ADHD-200 3D CNN Replication — Project Specification

**Paper:** Zou et al. 2017, "3D CNN Based Automatic Diagnosis of Attention Deficit Hyperactivity Disorder Using Functional and Structural MRI"  
**Published in:** IEEE Access, DOI: 10.1109/ACCESS.2017.2762703  
**Goal:** Replicate the paper's pipeline and match its state-of-the-art accuracy of **69.15%** on the ADHD-200 hold-out test set, then extend to Major Depressive Disorder classification.

---

## Repo Structure

```
csci1470-final-project/
├── data/
│   └── raw/
│       └── nyu/              ← downloaded .nii.gz files land here
├── models/
│   ├── __init__.py
│   ├── baseline_2d.py        ← optional 2D baseline for comparison
│   ├── single_modal_3d.py    ← single-modality 3D CNN (TO BUILD)
│   └── multi_modal_3d.py     ← dual-branch multi-modality CNN (TO BUILD)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_visualization.ipynb
├── scripts/
│   ├── download_adhd200.py   ← downloads from S3, DONE
│   ├── run_downloads.sh      ← runs 3 downloads (ReHo, fALFF, GM) for NYU, DONE
│   ├── train.py              ← training loop (TO BUILD)
│   ├── evaluate.py           ← evaluation script (TO BUILD)
│   └── run_ablations.sh      ← ablation experiments (TO BUILD)
├── utils/
│   ├── __init__.py
│   ├── dataset.py            ← PyTorch Dataset class (TO BUILD)
│   ├── metrics.py            ← accuracy tracking (TO BUILD)
│   └── preprocessing.py     ← NIfTI loading + normalization (TO BUILD)
├── SPEC.md                   ← this file
└── requirements.txt          ← TO FILL IN
```

---

## Step 1: Data — DONE

### What was downloaded

Using `scripts/run_downloads.sh`, which calls `scripts/download_adhd200.py` three times.

Files are fetched from the Preprocessed Connectomes Project S3 bucket:
`https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200/`

Pipeline used: **Athena**, Strategy: **filt_global**, Site: **NYU**

Three feature types downloaded per subject:
- `reho` — Regional Homogeneity map
- `falff` — Fractional Amplitude of Low Frequency Fluctuations map
- `gm_tissue` — Gray Matter density map

Each file is a `.nii.gz` NIfTI file — a 3D brain volume where each voxel holds one scalar value.

### Quality filtering (already applied in download script)
- `func_mean_fd < 0.2` — excludes subjects with excessive head motion (framewise displacement threshold)
- `FILE_ID != 'no_filename'` — excludes subjects without preprocessed data

### Dataset splits (from paper, Table 1)
| Split | ADHD | TDC | Total |
|---|---|---|---|
| Preprocessed training | 197 (158 male) | 362 (190 male) | 559 (348 male) |
| Hold-out test | 77 (60 male) | 94 (46 male) | 171 (106 male) |

Labels are in the phenotypic CSV:
- `DX_GROUP = 1` → ADHD
- `DX_GROUP = 2` → TDC (Typically Developing Control)

Note: The paper excludes 108 subjects from the original training set whose fMRI was marked 'questionable' quality.

---

## Step 2: Preprocessing — TO BUILD (`utils/preprocessing.py`)

### What this file needs to do

1. **Load NIfTI files** using `nibabel`. Each file loads as a 3D numpy array via `nib.load(path).get_fdata()`.

2. **Crop to brain bounding box** — exclude zero-padded boundary regions:
   - fMRI features (ReHo, fALFF): crop to **47 × 60 × 46**
   - sMRI features (GM, WM, CSF): crop to **90 × 117 × 100**

3. **Normalize** each volume to zero mean and unit variance (z-score normalization, per subject).

4. **Parse the phenotypic CSV** to extract per-subject labels. The CSV is downloaded from:
   `https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200/ADHD200_phenotypic_preprocessed.csv`
   Key columns: `FILE_ID`, `SITE_ID`, `AGE_AT_SCAN`, `SEX`, `DX_GROUP`, `func_mean_fd`

5. **Return** a list of `(volume_tensor, label)` pairs ready for the DataLoader.

### Key libraries needed
```
nibabel      # load .nii.gz NIfTI files
numpy        # array operations
torch        # convert to tensors
```

---

## Step 3: Dataset & DataLoader — TO BUILD (`utils/dataset.py`)

### What this file needs to do

Implement a PyTorch `Dataset` class:

```python
class ADHDDataset(Dataset):
    def __init__(self, subject_ids, feature_paths, labels, transform=None):
        ...
    def __len__(self):
        ...
    def __getitem__(self, idx):
        # load volume, normalize, return (tensor, label)
        ...
```

For the **multi-modality** model, the dataset should return both fMRI and sMRI volumes for each subject so both CNN branches get fed simultaneously.

### Cross-validation setup
- 4-fold cross-validation on the training set (559 subjects)
- Repeated **50 times** with different random seeds
- Report mean accuracy and variance across all 50 runs
- Final evaluation on the hold-out test set of 171 subjects

---

## Step 4: Models — TO BUILD

### 4a. Single-Modality 3D CNN (`models/single_modal_3d.py`)

This architecture is used independently for each of the 6 feature types.

**Input dimensions:**
- fMRI features (ReHo, fALFF, VMHC): `47 × 60 × 46`
- sMRI features (GM, WM, CSF): `90 × 117 × 100`

**Architecture (layer by layer):**

| Layer | Operation | Output size (fMRI) | Output size (sMRI) |
|---|---|---|---|
| Input | — | 47×60×46 | 90×117×100 |
| Max-pool | 2×2×2 (fMRI) / 4×4×4 (sMRI) | 23×30×23 | 22×29×25 |
| C1 | 32 kernels, 5×5×5, stride 1 + BN + ReLU | — | — |
| Max-pool | 2×2×2 | 9×13×9 | 9×13×11 |
| C2 | 64 kernels, 3×3×3 + BN + ReLU | — | — |
| C3 | 64 kernels, 3×3×3 + BN + ReLU | — | — |
| C4 | 64 kernels, 3×3×3 + BN + ReLU | — | — |
| Flatten | — | — | — |
| F5 | FC 512 neurons + BN + ReLU + Dropout(0.5) | — | — |
| F6 | FC 2 neurons + Softmax | — | — |

**Key implementation details:**
- Use `nn.Conv3d`, `nn.MaxPool3d`, `nn.BatchNorm3d`, `nn.Linear`
- Weight initialization: **Xavier uniform** (`nn.init.xavier_uniform_`)
- Dropout probability: **0.5** applied at F5 and F6 input
- Activation: **ReLU** throughout, **Softmax** at output
- Batch norm after **every** conv layer and FC layer

### 4b. Multi-Modality 3D CNN (`models/multi_modal_3d.py`)

Two parallel branches of the single-modality architecture above, then merged.

```
fMRI input → [Single-modal CNN branch] → 512-dim feature vector ─┐
                                                                    → Concat (1024) → FC(2) → Softmax
sMRI input → [Single-modal CNN branch] → 512-dim feature vector ─┘
```

**Why this works:** fMRI captures functional activity patterns; sMRI captures structural/morphological differences. ADHD involves both functional dysregulation and subtle structural changes, so combining them is complementary. The paper's best result (69.15%) comes from combining just **fALFF + GM density**.

---

## Step 5: Training — TO BUILD (`scripts/train.py`)

### Hyperparameters (exact values from paper)
```python
optimizer = SGD(lr=0.0001, momentum=0.9)   # momentum value typical for this setup
lr_scheduler = StepLR(step_size=20, gamma=0.5)  # decay LR by 0.5 every 20 epochs
batch_size = 20
num_epochs = 100
loss_fn = CrossEntropyLoss()
weight_init = xavier_uniform
dropout = 0.5
n_folds = 4
n_repeats = 50
```

### Training loop structure
```
for repeat in range(50):
    shuffle and split training data into 4 folds
    for fold in range(4):
        train on 3 folds, validate on 1
        run for 100 epochs with lr decay
    record fold accuracy
report mean and variance of accuracy across all runs
evaluate final model on hold-out test set (171 subjects)
```

### Overfitting prevention (all used in paper)
- Partial connectivity and weight sharing (intrinsic to CNN)
- Max-pooling (reduces spatial dimensions)
- Batch normalization after every layer
- Dropout = 0.5 at fully connected layers

---

## Step 6: Evaluation — TO BUILD (`utils/metrics.py`, `scripts/evaluate.py`)

### Metrics to track
- **Classification accuracy** (primary metric, matches paper's reported numbers)
- Per-site accuracy breakdown: PekingU, KKI, NYU (see paper Table 3)
- Mean and variance across 50 runs

### Target numbers to match (from paper)
| Feature | Accuracy |
|---|---|
| fALFF only | 66.04% |
| GM density only | 65.86% |
| **fALFF + GM (multi-modal)** | **69.15%** ← main target |
| NYU site only | 70.50% |

---

## Requirements — TO FILL IN (`requirements.txt`)

```
torch>=2.0.0
torchvision
nibabel          # NIfTI file loading
numpy
pandas           # phenotypic CSV parsing
scikit-learn     # cross-validation splits
matplotlib       # visualization
nilearn          # optional: brain image visualization
tqdm             # progress bars
```

Install with:
```bash
pip install torch torchvision nibabel numpy pandas scikit-learn matplotlib tqdm
```

---

## Extension: Major Depressive Disorder (future work)

After replicating ADHD results, apply the same pipeline to MDD classification using a separate large brain scan dataset. The same 3D CNN architecture should transfer with minimal modification — the main changes will be in the data loading and potentially the choice of fMRI features most relevant to MDD (e.g., default mode network connectivity).

---

## Key Scientific Concepts to Know

**ReHo (Regional Homogeneity):** Measures how synchronous a voxel's BOLD signal is with its 26 nearest neighbors using Kendall's coefficient of concordance. High ReHo = locally coordinated activity. ADHD children show abnormal ReHo particularly in frontal regions.

**fALFF (Fractional ALFF):** The ratio of power in the 0.01–0.1 Hz band to total detectable power. More specific than raw ALFF because it normalizes out physiological noise at higher frequencies. The paper's best single fMRI feature.

**Gray Matter (GM) Density:** From voxel-based morphometry (VBM) of structural T1 scans. Each voxel holds the probability it belongs to gray matter. ADHD is associated with subtle GM reductions in prefrontal and cerebellar regions. The paper's best single sMRI feature.

**MNI Space:** A standardized brain coordinate system (Montreal Neurological Institute template). All brains are warped to this common space so voxel (x,y,z) means the same anatomical location across all subjects.

**Why 3D CNN over flat features:** Previous methods flattened the 3D voxel maps into 1D vectors before feeding to classifiers (SVM, DBN), discarding all spatial neighborhood structure. The 3D CNN preserves and explicitly learns local 3D spatial patterns — analogous to how a radiologist reads spatial relationships across brain slices.

---

## Citation

Zou, L., Zheng, J., Miao, C., McKeown, M. J., & Wang, Z. J. (2017). 3D CNN based automatic diagnosis of attention deficit hyperactivity disorder using functional and structural MRI. *IEEE Access*, 5, 23626–23636. https://doi.org/10.1109/ACCESS.2017.2762703

Data source: Bellec, P., et al. (2017). The Neuro Bureau ADHD-200 Preprocessed repository. *NeuroImage*, 144, 275–286. https://doi.org/10.1016/j.neuroimage.2016.06.034
