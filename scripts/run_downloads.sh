#!/bin/bash
cd "$(dirname "$0")"

OUT_DIR="../data/raw"

echo "Downloading ADHD-200 (CPAC pipeline, all sites, all subjects)..."
python download_adhd200.py -o $OUT_DIR

echo "Done. Files saved to $OUT_DIR"
