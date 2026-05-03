#!/bin/bash
#SBATCH --job-name=abide_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=../outputs_abide/oscar_%j.log
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

# 1. DOWNLOAD THE DATA DIRECTLY TO OSCAR FIRST
if [ ! -f ../data/raw_abide/phenotypic.csv ]; then
  echo "===== Downloading Full ABIDE Dataset ====="
  python download_abide.py -o ../data/raw_abide
else
  echo "ABIDE data already exists, skipping download."
fi

# 2. START THE TRAINING LOOPS
echo "===== ABIDE Experiment 1: fALFF ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative falff \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --num-workers 4 \
    --pin-memory \
    --cache-data \
    --output-dir ../outputs_abide/falff

echo "===== ABIDE Experiment 2: ReHo ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative reho \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --num-workers 4 \
    --pin-memory \
    --cache-data \
    --output-dir ../outputs_abide/reho

echo "===== ABIDE Experiment 3: GM ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative gm \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --num-workers 4 \
    --pin-memory \
    --cache-data \
    --output-dir ../outputs_abide/gm

echo "===== ABIDE Experiment 4: Multi-modal fALFF+GM ====="
python train.py \
    --data-dir ../data/raw_abide \
    --mode multi \
    --fmri-derivative falff \
    --smri-derivative gm \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --num-workers 4 \
    --pin-memory \
    --cache-data \
    --output-dir ../outputs_abide/falff_gm_multi