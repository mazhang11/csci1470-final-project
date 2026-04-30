"""
plot_results.py

Generate publication-quality graphs from training results for poster presentation.

Generates two types of plots:
1. Learning curves: train/val loss and accuracy for first few folds
2. Repeat progression: mean accuracy across 50 CV repeats

Usage:
    python plot_results.py --output-dir ../outputs/falff
    python plot_results.py --output-dir ../outputs/falff_gm_multi --title "Multi-modal fALFF+GM"
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def read_training_history(csv_path):
    """Read a training history CSV and return lists of metrics."""
    epochs, tr_losses, tr_accs, val_losses, val_accs = [], [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            tr_losses.append(float(row['train_loss']))
            tr_accs.append(float(row['train_acc']))
            val_losses.append(float(row['val_loss']))
            val_accs.append(float(row['val_acc']))
    return epochs, tr_losses, tr_accs, val_losses, val_accs


def plot_learning_curves(output_dir, title=None, n_folds_to_plot=4):
    """
    Plot learning curves (loss and accuracy over epochs) for the first few folds.
    
    Args:
        output_dir: Directory containing training_history_*.csv files
        title: Optional title for the figure
        n_folds_to_plot: Number of folds to display (2x2 grid for 4 folds)
    """
    csv_files = sorted(glob.glob(os.path.join(output_dir, "training_history_*.csv")))
    
    if not csv_files:
        print(f"Warning: No training_history_*.csv files found in {output_dir}")
        return
    
    # Take first repeat's folds (repeat 00, folds 0-3)
    csv_files = [f for f in csv_files if '_r00_' in f][:n_folds_to_plot]
    
    if not csv_files:
        print(f"Warning: No repeat 00 CSVs found")
        return
    
    # Create 2x2 subplot grid
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    if title is None:
        title = os.path.basename(output_dir.rstrip('/'))
    fig.suptitle(f'Learning Curves: {title.upper()}', fontsize=16, fontweight='bold')
    
    for idx, csv_file in enumerate(csv_files):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Parse filename to get fold number
        fold_num = int(csv_file.split('_f')[1].split('.')[0])
        
        # Read data
        epochs, tr_losses, tr_accs, val_losses, val_accs = read_training_history(csv_file)
        
        # Plot on secondary y-axis for loss
        ax2 = ax.twinx()
        
        # Left y-axis: accuracy
        l1 = ax.plot(epochs, np.array(tr_accs)*100, 'o-', linewidth=2, markersize=3,
                     color='#1f77b4', label='Train Accuracy')
        l2 = ax.plot(epochs, np.array(val_accs)*100, 's-', linewidth=2, markersize=3,
                     color='#ff7f0e', label='Val Accuracy')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11, color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax.grid(True, alpha=0.3)
        
        # Right y-axis: loss
        l3 = ax2.plot(epochs, tr_losses, '^--', linewidth=2, markersize=3,
                      color='#2ca02c', label='Train Loss', alpha=0.6)
        l4 = ax2.plot(epochs, val_losses, 'v--', linewidth=2, markersize=3,
                      color='#d62728', label='Val Loss', alpha=0.6)
        ax2.set_ylabel('Loss', fontsize=11, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Title for subplot
        ax.set_title(f'Fold {fold_num}', fontsize=12, fontweight='bold')
        
        # Combined legend
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=9)
    
    output_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_repeat_progression(output_dir, title=None):
    """
    Plot mean validation accuracy across all 50 CV repeats.
    Shows how performance converges and stabilizes.
    
    Args:
        output_dir: Directory containing results.txt
    """
    results_path = os.path.join(output_dir, 'results.txt')
    
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found")
        return
    
    # Parse results.txt
    repeat_means = None
    with open(results_path, 'r') as f:
        for line in f:
            if 'all_repeat_means:' in line:
                # Extract list from string like: [0.62, 0.65, ...]
                list_str = line.split('all_repeat_means:')[1].strip()
                repeat_means = eval(list_str)
                break
    
    if repeat_means is None:
        print(f"Warning: Could not parse all_repeat_means from {results_path}")
        return
    
    # Convert to numpy array and percentage
    accs = np.array(repeat_means) * 100
    repeats = np.arange(1, len(accs) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot accuracy line
    ax.plot(repeats, accs, 'o-', linewidth=2.5, markersize=5, 
            color='#1f77b4', label='Mean Validation Accuracy')
    
    # Add mean and std bands
    mean_acc = accs.mean()
    std_acc = accs.std()
    ax.axhline(y=mean_acc, color='r', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {mean_acc:.2f}%')
    ax.fill_between(repeats, mean_acc - std_acc, mean_acc + std_acc, 
                    alpha=0.2, color='r', label=f'±1 Std: {std_acc:.2f}%')
    
    # Formatting
    ax.set_xlabel('Cross-Validation Repeat Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Validation Accuracy (%)', fontsize=12, fontweight='bold')
    
    if title is None:
        title = os.path.basename(output_dir.rstrip('/'))
    ax.set_title(f'CV Repeat Progression: {title.upper()}', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(0, len(accs) + 1)
    ax.set_ylim(min(accs) - 5, max(accs) + 5)
    
    output_path = os.path.join(output_dir, 'repeat_progression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comparison(output_dirs, labels=None, output_path='comparison.png'):
    """
    Compare repeat progression across multiple experiments.
    
    Args:
        output_dirs: List of directories containing results.txt
        labels: List of labels for each experiment (e.g., ['fALFF', 'ReHo', 'GM', 'Multi-modal'])
        output_path: Path to save the comparison figure
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (output_dir, color) in enumerate(zip(output_dirs, colors)):
        results_path = os.path.join(output_dir, 'results.txt')
        
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found, skipping")
            continue
        
        # Parse results.txt
        repeat_means = None
        with open(results_path, 'r') as f:
            for line in f:
                if 'all_repeat_means:' in line:
                    list_str = line.split('all_repeat_means:')[1].strip()
                    repeat_means = eval(list_str)
                    break
        
        if repeat_means is None:
            continue
        
        accs = np.array(repeat_means) * 100
        repeats = np.arange(1, len(accs) + 1)
        
        label = labels[idx] if labels else os.path.basename(output_dir.rstrip('/'))
        ax.plot(repeats, accs, 'o-', linewidth=2.5, markersize=4,
                color=color, label=label, alpha=0.8)
    
    ax.set_xlabel('Cross-Validation Repeat Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('ADHD-200 Model Comparison: Single vs Multi-Modal', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(0, 51)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output-dir', required=True,
                        help='Output directory containing results and training histories')
    parser.add_argument('--title', default=None,
                        help='Optional custom title for plots')
    
    args = parser.parse_args()
    
    print(f"Generating plots from: {args.output_dir}\n")
    
    # Generate both plot types
    print("1. Generating learning curves...")
    plot_learning_curves(args.output_dir, title=args.title)
    
    print("2. Generating repeat progression...")
    plot_repeat_progression(args.output_dir, title=args.title)
    
    print("\nDone! Open the PNG files in the output directory to view.")
