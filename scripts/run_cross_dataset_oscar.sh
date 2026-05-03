#!/bin/bash
#SBATCH --job-name=adhd_to_abide
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=../outputs/cross_dataset/oscar_%j.log
#SBATCH --mail-type=END,FAIL

# Load Python and CUDA, then activate venv
set -euo pipefail
module purge
module load python/3.12.8-zr3c
module load cuda/12.9.0-cinr
source ~/csci1470-final-project/venv/bin/activate

cd ~/csci1470-final-project/scripts

echo "=== OSCAR environment diagnostics ==="
python - <<'PY'
import sys, os, torch
print('python', sys.version)
print('torch', torch.__version__)
print('torch.version.cuda', torch.version.cuda)
print('torch.cuda.is_available()', torch.cuda.is_available())
print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
if torch.cuda.is_available():
    print('cuda device count:', torch.cuda.device_count())
    print('current device:', torch.cuda.current_device())
    print('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
PY

mkdir -p ../outputs/cross_dataset

# Evaluate fALFF checkpoints (trained on ADHD-200) on ABIDE subjects.
echo "===== Cross-dataset: fALFF (ADHD-200 → ABIDE) ====="
python evaluate_cross_dataset.py \
    --adhd-ckpt-dir ../outputs/falff \
    --abide-dir     ../data/raw_abide \
    --output-dir    ../outputs/cross_dataset/falff \
    --mode single \
    --derivative falff

# Evaluate ReHo checkpoints on ABIDE.
echo "===== Cross-dataset: ReHo (ADHD-200 → ABIDE) ====="
python evaluate_cross_dataset.py \
    --adhd-ckpt-dir ../outputs/reho \
    --abide-dir     ../data/raw_abide \
    --output-dir    ../outputs/cross_dataset/reho \
    --mode single \
    --derivative reho

# Evaluate GM checkpoints on ABIDE.
echo "===== Cross-dataset: GM (ADHD-200 → ABIDE) ====="
python evaluate_cross_dataset.py \
    --adhd-ckpt-dir ../outputs/gm \
    --abide-dir     ../data/raw_abide \
    --output-dir    ../outputs/cross_dataset/gm \
    --mode single \
    --derivative gm

# Evaluate multi-modal checkpoints on ABIDE.
echo "===== Cross-dataset: Multi-modal (ADHD-200 → ABIDE) ====="
python evaluate_cross_dataset.py \
    --adhd-ckpt-dir ../outputs/falff_gm_multi \
    --abide-dir     ../data/raw_abide \
    --output-dir    ../outputs/cross_dataset/falff_gm_multi \
    --mode multi \
    --fmri-derivative falff \
    --smri-derivative gm
