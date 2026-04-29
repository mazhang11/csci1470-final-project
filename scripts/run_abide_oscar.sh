#!/bin/bash
#SBATCH --job-name=abide_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=../outputs_abide/oscar_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mara_oancea@brown.edu

# Load Python and CUDA, then activate venv
module load python/3.12.8-zr3c
module load cuda/12.9.0-cinr
source ~/csci1470-final-project/venv/bin/activate

cd ~/csci1470-final-project/scripts

# 1. DOWNLOAD THE DATA DIRECTLY TO OSCAR FIRST
echo "===== Downloading Full ABIDE Dataset ====="
python download_abide.py -o ../data/raw_abide

# 2. START THE TRAINING LOOPS
echo "===== ABIDE Experiment 1: fALFF ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative falff \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs_abide/falff

echo "===== ABIDE Experiment 2: ReHo ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative reho \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
    --output-dir ../outputs_abide/reho

echo "===== ABIDE Experiment 3: GM ====="
python train.py \
    --data-dir ../data/raw_abide \
    --derivative gm \
    --mode single \
    --n-repeats 50 \
    --epochs 100 \
    --batch-size 20 \
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
    --output-dir ../outputs_abide/falff_gm_multi