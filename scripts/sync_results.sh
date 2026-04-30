#!/bin/bash
# sync_results.sh
# Pull latest results from Oscar into the correct local directories.
# Run this from anywhere — paths are absolute.
#
# Usage: bash scripts/sync_results.sh

OSCAR="moancea@ssh.ccv.brown.edu"
LOCAL_ROOT="/Users/maraoancea/Desktop/Brown_2526/deep_learning/csci1470-final-project"

echo "=== Syncing ADHD-200 outputs ==="
rsync -avz --progress \
    "$OSCAR:~/csci1470-final-project/outputs/" \
    "$LOCAL_ROOT/outputs_oscar/"

echo ""
echo "=== Syncing ABIDE outputs ==="
rsync -avz --progress \
    "$OSCAR:~/csci1470-final-project/outputs_abide/" \
    "$LOCAL_ROOT/outputs_abide/"

echo ""
echo "=== Done. Regenerating figures ==="
python "$LOCAL_ROOT/scripts/visualize_results.py"
