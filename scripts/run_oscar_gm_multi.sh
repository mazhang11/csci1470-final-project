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
module load python/3.12.8-zr3c
module load cuda/12.9.0-cinr
source ~/csci1470-final-project/venv/bin/activate

cd ~/csci1470-final-project/scripts

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
