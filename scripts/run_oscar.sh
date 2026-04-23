#!/bin/bash
#SBATCH --job-name=adhd_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=../outputs/oscar_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mara_oancea@brown.edu

# Load Python and activate venv
module load python/3.11.0
source ~/csci1470-final-project/venv/bin/activate

cd ~/csci1470-final-project/scripts

echo "===== Experiment 1: fALFF (paper target: 62.06%) ====="
python train.py \
    --derivative falff \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs/falff

echo "===== Experiment 2: ReHo (paper target: 60.27%) ====="
python train.py \
    --derivative reho \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs/reho

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
