import contextlib
import os
import csv
import datetime
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the root project directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import get_dataloaders
from ml.models.pframe_encoder import AsymmetricAutoencoder
from ml.utils.device import get_device, maybe_compile

# --- 1. CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')
BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
LAMBDA_BITRATE = 0.02      # Adjusted moderately for the 3-frame temporal window
LATENT_CHANNELS = 64       # Slimmer bottleneck based on PCA of run *624 in ml/saved
EPOCHS = 100
DEVICE = get_device()
USE_CUDA = DEVICE.type == "cuda"
# DataLoader workers (only used when USE_CUDA; CPU/mac often faster with 0)
DATALOADER_NUM_WORKERS = min(8, (os.cpu_count() or 4))
TORCH_COMPILE = USE_CUDA  # first epoch pays compile cost; then faster on CUDA


def _cuda_amp():
    if USE_CUDA:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# POC caps — set to None to use the full dataset
TRAIN_MAX_SAMPLES = 15_000  # 3,000 sequences
VAL_MAX_SAMPLES   = 3_000  # 600 sequences
PATIENCE    = 0.1 * EPOCHS
COHERENCE_WEIGHT = 0.05          # Temporal stability weight


def run_name_generator():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# --- HELPERS ---
def _compute_distortion_loss(
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Distortion loss: L1.
    Compares residual prediction against residual target.
    """
    return F.l1_loss(output, target)


def _compute_bpp_loss(
    likelihoods: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Rate loss: estimated bits-per-pixel from the entropy model's likelihoods.
    BPP = -sum(log2(p(y))) / num_pixels
    """
    num_pixels = target.shape[0] * target.shape[2] * target.shape[3]
    return -torch.sum(torch.log2(likelihoods + 1e-6)) / num_pixels


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    likelihoods: torch.Tensor,
    lambda_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined Rate-Distortion loss: L = Distortion + lambda * BPP
    Distortion = L1
    Returns (total_loss, distortion_loss, bpp_loss) for logging.
    """
    distortion = _compute_distortion_loss(output, target)
    bpp        = _compute_bpp_loss(likelihoods, target)
    total      = (distortion * 10) + lambda_rate * bpp
    return total, distortion, bpp


def get_scheduler(
    optimizer: optim.Optimizer,
    train_loader: DataLoader
) -> optim.lr_scheduler.LRScheduler:
    """
    # TODO: Add more schedulers
    OneCycleLR: ramps LR up then down over the full training run.
    steps_per_epoch = number of train batches; pct_start=0.3 (default).
    """
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,
    )


def get_model_and_optimizer(
    in_channel: int = 3,
    latent_channels: int = 64,
    optimizer_type: str = "adam",
):
    """
    # TODO: Add more models if developed, add more optimizers if required.
    Returns a model and optimizer.
    """
    model = AsymmetricAutoencoder(in_channels=in_channel, latent_channels=latent_channels).to(DEVICE)
    if optimizer_type == "adam":
        try:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=USE_CUDA)
        except TypeError:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        print(f"Unknown optimizer type: {optimizer_type}, returning Adam")
        try:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=USE_CUDA)
        except TypeError:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer

# ---------------------------------------------------------------------------
# --- TRAINING --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train():
    print(f"Using device: {DEVICE}")
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"CUDA opts: cudnn.benchmark, TF32, AMP fp16, compile={TORCH_COMPILE}, "
            f"DataLoader workers={DATALOADER_NUM_WORKERS}, pin_memory"
        )
    save_folder = f"ml/models/saved/{run_name_generator()}"
    os.makedirs(save_folder, exist_ok=True)

    # Open metrics CSV for this run
    csv_path = f"{save_folder}/metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "train_loss", "train_dist_loss", "train_bpp",
        "val_loss",   "val_dist_loss",   "val_bpp",
    ])

    # --- 2. PREPARE 75-15-10 DATA SPLIT ---
    train_loader, val_loader = get_dataloaders(
        train_max_samples=TRAIN_MAX_SAMPLES,
        val_max_samples=VAL_MAX_SAMPLES,
        batch_size=BATCH_SIZE,
        data_path=DATA_PATH,
        sequence_len=10,
        global_prob=0.20,
        num_workers=DATALOADER_NUM_WORKERS if USE_CUDA else 0,
        pin_memory=USE_CUDA,
        persistent_workers=USE_CUDA and DATALOADER_NUM_WORKERS > 0,
        prefetch_factor=2,
    )

    print(f"Batches: {len(train_loader)} train | {len(val_loader)} val")

    # --- 3. MODEL, OPTIMIZER, SCHEDULER & LOSS ---
    model, optimizer = get_model_and_optimizer()
    if TORCH_COMPILE:
        model = maybe_compile(model, DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    scheduler = get_scheduler(optimizer, train_loader)

    # --- 4. MAIN LOOP ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- TRAINING PASS ---
        model.train()
        train_loss, train_dist, train_bpp = 0, 0, 0
        train_pframes = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=True)
        for i, frames in enumerate(train_bar):
            frames = frames.to(DEVICE, non_blocking=USE_CUDA)

            # Reset state for each GOP sequence
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0
            batch_psteps = 0
            f_hat_prev = None
            y_prev = None  # Track the latent 'thought' of the last frame
            had_backward = False

            # Predict residuals between consecutive GT frames (no GOP / no I-frame path).
            for t in range(1, frames.shape[1]):
                f_prev_gt = frames[:, t - 1]
                f_curr = frames[:, t]
                target = f_curr - f_prev_gt
                with _cuda_amp():
                    res_hat, p_y, y_curr = model(target, training=True)
                    loss, dist_loss, bpp_loss = compute_loss(
                        res_hat, target, p_y, LAMBDA_BITRATE
                    )
                    if y_prev is not None:
                        tcp_loss = F.mse_loss(y_curr, y_prev.detach())
                        loss = loss + COHERENCE_WEIGHT * tcp_loss

                if not torch.isnan(loss):
                    total_step_loss = loss / max(1, (frames.shape[1] - 1))
                    scaler.scale(total_step_loss).backward()
                    had_backward = True

                    batch_loss += float(loss.detach())
                    train_loss += float(loss.detach())
                    train_dist += float(dist_loss.detach())
                    train_bpp += float(bpp_loss.detach())
                    train_pframes += 1
                    batch_psteps += 1

                y_prev = y_curr.detach()

            if had_backward:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            train_bar.set_postfix({
                "batch_loss": f"{batch_loss / max(1, batch_psteps):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # --- VALIDATION PASS ---
        model.eval()
        val_loss, val_dist, val_bpp = 0, 0, 0
        val_pframes = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=True)

        with torch.no_grad():
            for i, frames in enumerate(val_bar):
                frames = frames.to(DEVICE, non_blocking=USE_CUDA)
                y_prev = None

                for t in range(1, frames.shape[1]):
                    f_prev_gt = frames[:, t - 1]
                    f_curr = frames[:, t]
                    target = f_curr - f_prev_gt
                    with _cuda_amp():
                        res_hat, p_y, y_curr = model(target, training=False)
                        v_loss, dist_v, bpp_v = compute_loss(
                            res_hat, target, p_y, LAMBDA_BITRATE
                        )
                        if y_prev is not None:
                            tcp_v = F.mse_loss(y_curr, y_prev)
                            v_loss = v_loss + COHERENCE_WEIGHT * tcp_v

                    val_loss += float(v_loss)
                    val_dist += float(dist_v)
                    val_bpp += float(bpp_v)
                    val_pframes += 1

                    y_prev = y_curr

                val_bar.set_postfix({"val_dist": f"{dist_v.item():.4f}"})

        avg_val_loss = val_loss / max(1, val_pframes)

        # --- CHECKPOINT & EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_folder}/best_model.pth")
            print("New best model saved!")
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_folder}/baseline_epoch_{epoch}.pth")
        if os.path.exists(f"{save_folder}/baseline_epoch_{epoch-15}.pth"):
            os.remove(f"{save_folder}/baseline_epoch_{epoch-15}.pth")

        n_train = max(1, train_pframes)
        n_val   = max(1, val_pframes)

        # --- CSV ROW ---
        csv_writer.writerow([
            epoch,
            f"{train_loss / n_train:.6f}",
            f"{train_dist / n_train:.6f}",
            f"{train_bpp  / n_train:.6f}",
            f"{avg_val_loss:.6f}",
            f"{val_dist   / n_val:.6f}",
            f"{val_bpp    / n_val:.6f}",
        ])
        csv_file.flush()  # Write immediately so progress is visible between epochs

        print(f"Training Loss: {train_loss / n_train:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}  |  Best: {best_val_loss:.4f}  |  Patience: {patience_counter}/{PATIENCE}")
        print("\n")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

    csv_file.close()
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    train()
