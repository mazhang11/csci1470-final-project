#!/bin/bash
#SBATCH --job-name=adhd_gm_multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=../outputs/oscar_%j.log
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

echo "===== Experiment 3: GM (paper target: 65.43%) ====="
python train.py \
    --derivative gm \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs/gm

echo "===== Experiment 4: Multi-modal fALFF+GM (paper target: 69.15%) ====="
python train.py \
    --mode multi \
    --fmri-derivative falff \
    --smri-derivative gm \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs/falff_gm_multi
