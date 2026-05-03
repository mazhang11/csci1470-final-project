"""
train.py — Replication of Zou et al. 2017 training procedure.

Hyperparameters (exact from paper):
  optimizer : SGD, lr=0.0001, momentum=0.9
  scheduler : StepLR, step_size=20 epochs, gamma=0.5
  batch_size: 20
  epochs    : 100
  loss      : CrossEntropyLoss
  cv        : 4-fold, repeated 50 times

Usage examples:
  # Single-modal fALFF
  python train.py --derivative falff

  # Single-modal GM
  python train.py --derivative gm

  # Multi-modal fALFF + GM  (paper's best: 69.15%)
  python train.py --mode multi

  # Quick smoke-test (2 repeats, 2 epochs)
  python train.py --derivative falff --n-repeats 2 --epochs 2
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.multi_modal_3d import MultiModal3DCNN
from models.single_modal_3d import fmri_cnn, smri_cnn
from utils.dataset import ADHDDataset, ADHDMultiModalDataset, build_datasets
from utils.preprocessing import FMRI_DERIVATIVES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model(args):
    if args.mode == "multi":
        return MultiModal3DCNN()
    derivative = args.derivative.lower()
    if derivative in FMRI_DERIVATIVES:
        return fmri_cnn()
    return smri_cnn()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, multi):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, multi):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
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

    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device count = {torch.cuda.device_count()}")
        print(f"cuda current device = {torch.cuda.current_device()}")
        print(f"cuda device name = {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if device.type == "cpu" and torch.cuda.is_available():
        print("Warning: CUDA is available but the selected device is CPU.")

    multi = args.mode == "multi"

    # Build full dataset
    dataset = build_datasets(
        args.data_dir,
        derivative=args.derivative,
        fmri_derivative=args.fmri_derivative,
        smri_derivative=args.smri_derivative,
        multi_modal=multi,
        cache_in_memory=args.cache_data,
    )
    n = len(dataset)
    indices = np.arange(n)
    print(f"Dataset: {n} subjects  |  mode={args.mode}  |  "
          f"derivative={args.derivative if not multi else f'{args.fmri_derivative}+{args.smri_derivative}'}")

    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(args.output_dir, exist_ok=True)

    all_val_accs = []   # one entry per (repeat, fold)
    repeat_means = []   # mean val accuracy per repeat

    t0 = time.time()

    for repeat in range(args.n_repeats):
        rng = np.random.default_rng(seed=repeat)
        shuffled = rng.permutation(indices)

        kf = KFold(n_splits=args.n_folds, shuffle=False)
        fold_accs = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(shuffled)):
            train_subset = Subset(dataset, shuffled[train_idx].tolist())
            val_subset   = Subset(dataset, shuffled[val_idx].tolist())

            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,   # keeps BN happy — never a batch of 1
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and device.type == "cuda",
                persistent_workers=(args.num_workers > 0),
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and device.type == "cuda",
                persistent_workers=(args.num_workers > 0),
            )

            model = get_model(args).to(device)
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

            # Create CSV file for epoch-by-epoch logging
            epoch_csv_path = os.path.join(
                args.output_dir,
                f"training_history_r{repeat:02d}_f{fold}.csv"
            )
            csv_file = open(epoch_csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            csv_file.flush()

            best_val_acc = 0.0

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, device, multi
                )
                val_loss, val_acc = evaluate(
                    model, val_loader, loss_fn, device, multi
                )
                scheduler.step()

                # Log every epoch to CSV
                csv_writer.writerow([epoch, tr_loss, tr_acc, val_loss, val_acc])
                csv_file.flush()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"best_r{repeat:02d}_f{fold}.pt"
                    )
                    torch.save(model.state_dict(), ckpt_path)

                if epoch % 10 == 0 or epoch == 1:
                    elapsed = time.time() - t0
                    print(
                        f"  repeat={repeat+1:2d}/{args.n_repeats}  "
                        f"fold={fold+1}/{args.n_folds}  "
                        f"epoch={epoch:3d}/{args.epochs}  "
                        f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                        f"[{elapsed:.0f}s]"
                    )

            csv_file.close()

            fold_accs.append(best_val_acc)
            all_val_accs.append(best_val_acc)
            print(f"  → repeat={repeat+1} fold={fold+1} best_val_acc={best_val_acc:.4f}")

        repeat_mean = np.mean(fold_accs)
        repeat_means.append(repeat_mean)
        print(f"Repeat {repeat+1:2d} mean val acc: {repeat_mean:.4f}")

        # Write incremental checkpoint after every repeat so progress is never lost.
        results_path = os.path.join(args.output_dir, "results.txt")
        deriv_str = args.derivative if not multi else f"{args.fmri_derivative}+{args.smri_derivative}"
        completed = len(repeat_means)
        running_mean = np.mean(repeat_means) * 100
        running_var  = np.var(repeat_means)  * 10000
        with open(results_path, "w") as f:
            f.write(f"mode:            {args.mode}\n")
            f.write(f"derivative:      {deriv_str}\n")
            f.write(f"n_repeats:       {args.n_repeats}\n")
            f.write(f"n_folds:         {args.n_folds}\n")
            f.write(f"epochs:          {args.epochs}\n")
            f.write(f"repeats_done:    {completed}/{args.n_repeats}\n")
            f.write(f"mean_val_acc:    {running_mean:.4f}%\n")
            f.write(f"variance:        {running_var:.4f}\n")
            f.write(f"best_run:        {max(repeat_means)*100:.4f}%\n")
            f.write(f"all_repeat_means: {[round(x*100,4) for x in repeat_means]}\n")

    # Final summary
    mean_acc = np.mean(repeat_means) * 100
    var_acc  = np.var(repeat_means)  * 10000   # variance of %-accuracy
    print("\n" + "="*60)
    print(f"Mean val accuracy : {mean_acc:.2f}%")
    print(f"Variance          : {var_acc:.2f}")
    print(f"Best single run   : {max(repeat_means)*100:.2f}%")
    print(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data-dir",       default="../data/raw")
    parser.add_argument("--output-dir",     default="../outputs")
    parser.add_argument("--mode",           default="single",
                        choices=["single", "multi"],
                        help="single: one derivative; multi: fALFF+GM fusion")
    parser.add_argument("--derivative",     default="falff",
                        choices=["falff", "reho", "gm"],
                        help="Which derivative for single-modal mode")
    parser.add_argument("--fmri-derivative", default="falff",
                        choices=["falff", "reho"])
    parser.add_argument("--smri-derivative", default="gm")

    # Training hyperparameters (paper defaults)
    parser.add_argument("--n-repeats",  type=int,   default=50)
    parser.add_argument("--n-folds",    type=int,   default=4)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--num-workers", type=int,   default=4,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Enable DataLoader pin_memory when using CUDA")
    parser.add_argument("--cache-data", action="store_true",
                        help="Preload all dataset tensors into memory before training")
    parser.add_argument("--lr",         type=float, default=0.0001)
    parser.add_argument("--momentum",   type=float, default=0.9)
    parser.add_argument("--lr-step",    type=int,   default=20)
    parser.add_argument("--lr-gamma",   type=float, default=0.5)

    args = parser.parse_args()
    run(args)
