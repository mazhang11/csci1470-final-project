"""
generate_all_plots.py

Generate all poster-quality graphs after training completes.
Automatically plots results for all modalities (fALFF, ReHo, GM, Multi-modal).

Usage:
    python generate_all_plots.py --results-dir ../outputs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from plot_results import plot_learning_curves, plot_repeat_progression, plot_comparison


def generate_all_plots(results_dir):
    """Generate all plots for a results directory containing multiple experiments."""
    
    experiments = {
        'falff': 'Single-Modal fALFF',
        'reho': 'Single-Modal ReHo',
        'gm': 'Single-Modal GM',
        'falff_gm_multi': 'Multi-Modal (fALFF + GM)',
    }
    
    output_dirs = []
    valid_experiments = []
    
    for exp_name, exp_label in experiments.items():
        exp_dir = os.path.join(results_dir, exp_name)
        if os.path.exists(exp_dir):
            output_dirs.append(exp_dir)
            valid_experiments.append(exp_label)
            print(f"\n{'='*60}")
            print(f"Plotting: {exp_label}")
            print(f"{'='*60}")
            
            # Generate individual plots
            plot_learning_curves(exp_dir, title=exp_label)
            plot_repeat_progression(exp_dir, title=exp_label)
        else:
            print(f"⚠ Skipping {exp_name}: directory not found at {exp_dir}")
    
    # Generate comparison plot across all modalities
    if len(output_dirs) > 1:
        print(f"\n{'='*60}")
        print("Generating comparison plot...")
        print(f"{'='*60}")
        comparison_path = os.path.join(results_dir, 'comparison_all_modalities.png')
        plot_comparison(output_dirs, labels=valid_experiments, output_path=comparison_path)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results-dir', required=True,
                        help='Root results directory (e.g., ../outputs)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    generate_all_plots(args.results_dir)
