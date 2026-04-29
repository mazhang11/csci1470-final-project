"""
evaluate_cross_dataset.py

Cross-dataset evaluation: load the best checkpoints trained on ADHD-200
and test them directly on ABIDE subjects (no retraining).

This answers the question: do brain patterns learned from ADHD generalize
to autism classification?

Usage:
  # Evaluate fALFF checkpoint on ABIDE
  python evaluate_cross_dataset.py --derivative falff

  # Evaluate multi-modal checkpoint on ABIDE
  python evaluate_cross_dataset.py --mode multi
"""

import argparse
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.multi_modal_3d import MultiModal3DCNN
from models.single_modal_3d import fmri_cnn, smri_cnn
from utils.dataset import build_datasets
from utils.preprocessing import FMRI_DERIVATIVES


def get_model(args):
    if args.mode == "multi":
        return MultiModal3DCNN()
    if args.derivative.lower() in FMRI_DERIVATIVES:
        return fmri_cnn()
    return smri_cnn()


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, model, loader, loss_fn, device, multi):
    """Load a checkpoint and evaluate it on the given DataLoader."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        if multi:
            fmri, smri, labels = batch
            fmri, smri, labels = fmri.to(device), smri.to(device), labels.to(device)
            logits = model(fmri, smri)
        else:
            volumes, labels = batch
            volumes, labels = volumes.to(device), labels.to(device)
            logits = model(volumes)

        loss = loss_fn(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n += bs

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / n, total_acc / n, all_preds, all_labels


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    multi = args.mode == "multi"

    # Load ABIDE as the test set — no training, inference only.
    abide_dataset = build_datasets(
        args.abide_dir,
        derivative=args.derivative,
        fmri_derivative=args.fmri_derivative,
        smri_derivative=args.smri_derivative,
        multi_modal=multi,
    )
    print(f"ABIDE test set: {len(abide_dataset)} subjects")

    loader = DataLoader(
        abide_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Find all ADHD-200 checkpoints in the output directory.
    ckpt_pattern = os.path.join(args.adhd_ckpt_dir, "best_r*_f*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found at {ckpt_pattern}. "
            "Run train.py on ADHD-200 first."
        )
    print(f"Found {len(checkpoints)} ADHD-200 checkpoints")

    # Evaluate every checkpoint and collect per-checkpoint accuracy.
    accs, losses = [], []
    for ckpt in checkpoints:
        model = get_model(args).to(device)
        loss, acc, preds, labels = evaluate_checkpoint(
            ckpt, model, loader, loss_fn, device, multi
        )
        accs.append(acc)
        losses.append(loss)
        print(f"  {os.path.basename(ckpt):30s}  acc={acc:.4f}  loss={loss:.4f}")

    mean_acc = np.mean(accs) * 100
    std_acc  = np.std(accs)  * 100
    best_acc = np.max(accs)  * 100

    # Chance level depends on class balance in ABIDE.
    print("\n" + "="*60)
    print(f"Cross-dataset evaluation: ADHD-200 → ABIDE")
    print(f"Derivative / mode : {args.derivative if not multi else 'multi'}")
    print(f"Checkpoints used  : {len(checkpoints)}")
    print(f"Mean acc          : {mean_acc:.2f}%")
    print(f"Std acc           : {std_acc:.2f}%")
    print(f"Best checkpoint   : {best_acc:.2f}%")
    print(f"Chance level      : ~50% (roughly balanced ABIDE)")

    # Save results.
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "cross_dataset_results.txt")
    with open(results_path, "w") as f:
        f.write(f"eval_type:        ADHD-200 → ABIDE\n")
        f.write(f"mode:             {args.mode}\n")
        f.write(f"derivative:       {args.derivative if not multi else f'{args.fmri_derivative}+{args.smri_derivative}'}\n")
        f.write(f"n_checkpoints:    {len(checkpoints)}\n")
        f.write(f"mean_acc:         {mean_acc:.4f}%\n")
        f.write(f"std_acc:          {std_acc:.4f}%\n")
        f.write(f"best_acc:         {best_acc:.4f}%\n")
        f.write(f"all_accs:         {[round(a*100,4) for a in accs]}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--adhd-ckpt-dir", default="../outputs/falff",
                        help="Directory containing ADHD-200 best_r*_f*.pt checkpoints")
    parser.add_argument("--abide-dir",     default="../data/raw_abide",
                        help="ABIDE data directory (with phenotypic.csv)")
    parser.add_argument("--output-dir",    default="../outputs/cross_dataset")
    parser.add_argument("--mode",          default="single", choices=["single", "multi"])
    parser.add_argument("--derivative",    default="falff",  choices=["falff", "reho", "gm"])
    parser.add_argument("--fmri-derivative", default="falff")
    parser.add_argument("--smri-derivative", default="gm")
    parser.add_argument("--batch-size",    type=int, default=8)

    args = parser.parse_args()
    run(args)
