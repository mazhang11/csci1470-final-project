#!/bin/bash

# Navigate to the directory where the script is located to ensure relative paths work
cd "$(dirname "$0")"

# Define the output directory (stepping up one level out of /scripts into /data)
OUT_DIR="../data/raw/nyu"

# Define the pipeline and strategy based on the Athena pipeline for ADHD-200
PIPELINE="athena"
STRATEGY="filt_global"

# Create the data directory if it doesn't exist
mkdir -p $OUT_DIR

echo "Starting downloads for NYU site..."

# 1. Download ReHo maps
echo "Downloading ReHo..."
python download_adhd200.py -d reho -p $PIPELINE -s $STRATEGY -t NYU -o $OUT_DIR

# 2. Download fALFF maps
echo "Downloading fALFF..."
python download_adhd200.py -d falff -p $PIPELINE -s $STRATEGY -t NYU -o $OUT_DIR

# 3. Download GM (Gray Matter) maps
echo "Downloading Gray Matter (GM)..."
python download_adhd200.py -d gm_tissue -p $PIPELINE -s $STRATEGY -t NYU -o $OUT_DIR

echo "All specified NYU downloads complete. Files saved to $OUT_DIR"