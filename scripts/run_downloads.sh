#!/bin/bash
cd "$(dirname "$0")"

OUT_DIR="../data/raw"
ABIDE_OUT_DIR="../data/raw_abide"

echo "Downloading ADHD-200 (CPAC pipeline, all sites, all subjects)..."
python download_adhd200.py -o $OUT_DIR

echo "Downloading ABIDE (Autism)..."
python download_abide.py -o $ABIDE_OUT_DIR

echo "Done. Files saved to $ADHD_OUT_DIR and $ABIDE_OUT_DIR"