"""
plot_metrics.py
Reads metrics.csv from a training run and generates:
  1. Total loss curve (train vs val)
  2. Distortion loss curve (train vs val)
  3. BPP curve (train vs val)

All three are saved as a single 'loss_curves.png' in the run folder.

Usage:
    python scripts/plot_metrics.py --run ml/models/saved/<RUN_NAME>
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: str) -> dict:
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to the run folder (contains metrics.csv)")
    args = parser.parse_args()

    csv_path = os.path.join(args.run, "metrics.csv")
    assert os.path.exists(csv_path), f"metrics.csv not found: {csv_path}"

    d = load_csv(csv_path)
    epochs = d["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    plot_cfg = [
        ("Total Loss",       "train_loss",      "val_loss"),
        ("Distortion Loss",  "train_dist_loss", "val_dist_loss"),
        ("BPP (Rate Loss)",  "train_bpp",       "val_bpp"),
    ]

    for ax, (title, train_key, val_key) in zip(axes, plot_cfg):
        ax.plot(epochs, d[train_key], label="Train", linewidth=2, color="#4C9BE8")
        ax.plot(epochs, d[val_key],   label="Val",   linewidth=2, color="#E8754C", linestyle="--")

        # Mark best val epoch
        best_epoch = int(epochs[d[val_key].index(min(d[val_key]))])
        best_val   = min(d[val_key])
        ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1, alpha=0.8)
        ax.annotate(
            f"Best @ ep {best_epoch}\n{best_val:.4f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.5, best_val * 1.05),
            fontsize=8, color="gray",
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.run, "loss_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
