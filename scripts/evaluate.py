"""
evaluate.py

Run post-training analysis on one or more checkpoints and inspect whether
predictions follow the class prior.

This is useful to test the hypothesis that the model may be exploiting an
imbalanced label distribution rather than learning meaningful brain-based
features.
"""

import argparse
import glob
import os
import sys

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


def analyze_predictions(preds, labels):
    preds = np.asarray(preds, dtype=int)
    labels = np.asarray(labels, dtype=int)
    assert preds.shape == labels.shape

    counts = np.bincount(labels, minlength=2)
    pred_counts = np.bincount(preds, minlength=2)
    total = len(labels)
    accuracy = (preds == labels).mean()
    class_acc = np.zeros(2, dtype=float)
    for k in (0, 1):
        mask = labels == k
        class_acc[k] = (preds[mask] == k).mean() if mask.sum() > 0 else 0.0

    confusion = np.zeros((2, 2), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    majority_prior = counts.max() / total
    prior_class = int(np.argmax(counts))

    return {
        "total": int(total),
        "count_label_0": int(counts[0]),
        "count_label_1": int(counts[1]),
        "prior_majority_class": int(prior_class),
        "prior_majority_accuracy": float(majority_prior),
        "pred_count_0": int(pred_counts[0]),
        "pred_count_1": int(pred_counts[1]),
        "accuracy": float(accuracy),
        "class_acc_0": float(class_acc[0]),
        "class_acc_1": float(class_acc[1]),
        "confusion_matrix": confusion,
    }


def evaluate_checkpoint(ckpt_path, model, loader, loss_fn, device, multi):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
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
            n += bs
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    assert n == len(all_labels)
    analysis = analyze_predictions(all_preds, all_labels)
    analysis["loss"] = total_loss / n if n > 0 else float('nan')
    return analysis


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    multi = args.mode == "multi"

    dataset = build_datasets(
        args.data_dir,
        derivative=args.derivative,
        fmri_derivative=args.fmri_derivative,
        smri_derivative=args.smri_derivative,
        multi_modal=multi,
    )

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    checkpoint_paths = []
    if args.checkpoint_path:
        checkpoint_paths = [args.checkpoint_path]
    else:
        pattern = os.path.join(args.checkpoint_dir, "best_r*_f*.pt")
        checkpoint_paths = sorted(glob.glob(pattern))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {args.checkpoint_dir} with pattern best_r*_f*.pt"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "bias_analysis.txt")

    lines = []
    lines.append(f"Evaluation checkpoint directory: {args.checkpoint_dir}")
    lines.append(f"Dataset directory: {args.data_dir}")
    lines.append(f"Mode: {args.mode}")
    lines.append(f"Derivative: {args.derivative if not multi else f'{args.fmri_derivative}+{args.smri_derivative}'}")
    lines.append("")

    for ckpt in checkpoint_paths:
        model = get_model(args)
        analysis = evaluate_checkpoint(ckpt, model, loader, loss_fn, device, multi)
        basename = os.path.basename(ckpt)
        lines.append(f"Checkpoint: {basename}")
        lines.append(f"  loss: {analysis['loss']:.4f}")
        lines.append(f"  accuracy: {analysis['accuracy']:.4f}")
        lines.append(f"  class 0 accuracy: {analysis['class_acc_0']:.4f}")
        lines.append(f"  class 1 accuracy: {analysis['class_acc_1']:.4f}")
        lines.append(f"  prior majority class: {analysis['prior_majority_class']}")
        lines.append(f"  prior majority accuracy: {analysis['prior_majority_accuracy']:.4f}")
        lines.append(f"  prediction counts: class0={analysis['pred_count_0']} class1={analysis['pred_count_1']}")
        lines.append("  confusion matrix:")
        lines.append(f"    [{analysis['confusion_matrix'][0,0]} {analysis['confusion_matrix'][0,1]}]")
        lines.append(f"    [{analysis['confusion_matrix'][1,0]} {analysis['confusion_matrix'][1,1]}]")
        lines.append("")

    print("\n".join(lines))
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved bias analysis to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../data/raw",
                        help="Dataset directory to evaluate")
    parser.add_argument("--checkpoint-dir", default="../outputs/falff",
                        help="Directory containing best checkpoint files")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Single checkpoint file path to evaluate")
    parser.add_argument("--output-dir", default="../outputs/bias_analysis",
                        help="Directory for saving bias analysis output")
    parser.add_argument("--mode", default="single", choices=["single", "multi"])
    parser.add_argument("--derivative", default="falff", choices=["falff", "reho", "gm"])
    parser.add_argument("--fmri-derivative", default="falff")
    parser.add_argument("--smri-derivative", default="gm")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args)
