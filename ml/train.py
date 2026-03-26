import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import SSIM
import os
import csv
import datetime
import sys
from tqdm import tqdm
import random

# Ensure the root project directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import ViratDataset, get_dataloaders
from ml.models.autoencoder import AsymmetricAutoencoder

# --- 1. CONFIGURATION ---
DATA_PATH = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'
BATCH_SIZE     = 12         # Reduced because GOP=10 uses 10x more frames per step
LEARNING_RATE  = 2.4e-4
LAMBDA_BITRATE = 0.02      # Adjusted moderately for the 10-frame temporal window
LATENT_CHANNELS = 64       # Slimmer bottleneck based on PCA of run *624 in ml/saved
EPOCHS = 100
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# POC caps — set to None to use the full dataset
TRAIN_MAX_SAMPLES = 375 * BATCH_SIZE  # 3,000 sequences
VAL_MAX_SAMPLES   = 75  * BATCH_SIZE  # 600 sequences
PATIENCE    = 0.1 * EPOCHS
IFRAME_PROB = 0.10  # This is now controlled by sequence_len (1 I-frame per 10 frames)
                    # Production target: 1/30 ≈ 0.033 (one I-frame per second at 30fps, GOP=30)
ALPHA_SSIM      = 0.84         # Distortion weight
COHERENCE_WEIGHT = 0.05          # Temporal stability weight


def run_name_generator():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# --- HELPERS ---
def _compute_distortion_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    ssim_module: SSIM,
    is_iframe: bool,
    alpha: float = ALPHA_SSIM,
) -> torch.Tensor:
    """
    Perceptual distortion loss: alpha * (1 - SSIM) + (1 - alpha) * L1

    I-frames are in [0, 1] and can be fed to SSIM directly.
    P-frames (residuals) are in [-1, 1], so we shift them to [0, 1]
    before computing SSIM to satisfy its data_range assumption.
    """
    if is_iframe:
        o_norm = output
        t_norm = target
    else:
        o_norm = ((output + 1.0) / 2.0).clamp(0, 1)
        t_norm = ((target + 1.0) / 2.0).clamp(0, 1)

    # Downsample to 50% resolution for SSIM — perceptual quality is well-preserved
    # at half-size, and this roughly halves the sliding-window conv cost.
    # L1 is computed at full 352×352 to retain accurate pixel-level gradients.
    o_small = F.interpolate(o_norm, scale_factor=0.5, mode="bilinear", align_corners=False)
    t_small = F.interpolate(t_norm, scale_factor=0.5, mode="bilinear", align_corners=False)

    ssim_loss = 1.0 - ssim_module(o_small, t_small)
    l1_loss   = F.l1_loss(output, target)
    return alpha * ssim_loss + (1.0 - alpha) * l1_loss


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
    ssim_module: SSIM,
    is_iframe: bool,
    lambda_rate: float,
    alpha: float = ALPHA_SSIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined Rate-Distortion loss: L = Distortion + lambda * BPP
    Distortion = alpha * (1 - SSIM) + (1 - alpha) * L1
    Returns (total_loss, distortion_loss, bpp_loss) for logging.
    """
    distortion = _compute_distortion_loss(output, target, ssim_module, is_iframe, alpha)
    bpp        = _compute_bpp_loss(likelihoods, target)
    total      = distortion + lambda_rate * bpp
    return total, distortion, bpp


def get_loss(
    data_range: float = 1.0,
    size_average: bool = True,
    channel: int = 3
) -> SSIM:
    """
    # TODO: Add more loss modules
    Returns a loss module.
    """
    return SSIM(data_range=data_range, size_average=size_average, channel=channel).to(DEVICE)


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
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        print(f"Unknown optimizer type: {optimizer_type}, returning Adam")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer

# ---------------------------------------------------------------------------
# --- TRAINING --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train():
    print(f"Using device: {DEVICE}")
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
    )

    print(f"Batches: {len(train_loader)} train | {len(val_loader)} val")

    # --- 3. MODEL, OPTIMIZER, SCHEDULER & LOSS ---
    model, optimizer = get_model_and_optimizer()
    loss_module = get_loss()
    scheduler = get_scheduler(optimizer, train_loader)

    # --- 4. MAIN LOOP ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- TRAINING PASS ---
        model.train()
        train_loss, train_dist, train_bpp = 0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=True)
        for i, frames in enumerate(train_bar):
            frames = frames.to(DEVICE)
            
            # Reset state for each GOP sequence
            optimizer.zero_grad()
            batch_loss = 0
            f_hat_prev = None  
            y_prev     = None  # Track the latent 'thought' of the last frame
            
            # Iterate through the sequence (GOP=10)
            for t in range(frames.shape[1]):
                f_curr = frames[:, t]
                is_iframe = (t == 0) # Force first frame of sequence to be I-frame
                
                # P-frame residual is relative to PREVIOUS RECONSTRUCTED frame
                target = f_curr if is_iframe else (f_curr - f_hat_prev)
                
                res_hat, p_y, y_curr = model(target, training=True)
                
                # Reconstruction logic: I-frame is direct, P-frame is additive
                f_hat_curr = res_hat if is_iframe else (f_hat_prev + res_hat)
                f_hat_curr = f_hat_curr.clamp(0, 1)
                
                # Main Rate-Distortion Loss
                loss, dist_loss, bpp_loss = compute_loss(
                    res_hat, target, p_y, loss_module, is_iframe, LAMBDA_BITRATE
                )

                # Temporal Coherence Loss (TCP): penalise latent flickering
                if not is_iframe and y_prev is not None:
                   tcp_loss = F.mse_loss(y_curr, y_prev.detach())
                   loss += COHERENCE_WEIGHT * tcp_loss
                
                if not torch.isnan(loss):
                    # We scale by 1/SEQ_LEN to keep the loss scale consistent
                    total_step_loss = loss / frames.shape[1]
                    total_step_loss.backward()
                    
                    batch_loss += loss.item()
                    train_loss += loss.item()
                    train_dist += dist_loss.item()
                    train_bpp  += bpp_loss.item()
                
                # Transition state to next frame in sequence
                f_hat_prev = f_hat_curr.detach()
                y_prev     = y_curr.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({
                "batch_loss": f"{batch_loss / frames.shape[1]:.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # --- VALIDATION PASS ---
        model.eval()
        val_loss, val_dist, val_bpp = 0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=True)

        with torch.no_grad():
            for i, frames in enumerate(val_bar):
                frames = frames.to(DEVICE)
                f_hat_prev = None
                y_prev     = None
                
                for t in range(frames.shape[1]):
                    f_curr = frames[:, t]
                    is_iframe = (t == 0)
                    target = f_curr if is_iframe else (f_curr - f_hat_prev)

                    res_hat, p_y, y_curr = model(target, training=False)
                    f_hat_curr = res_hat if is_iframe else (f_hat_prev + res_hat)
                    f_hat_curr = f_hat_curr.clamp(0, 1)

                    v_loss, dist_v, bpp_v = compute_loss(
                        res_hat, target, p_y, loss_module, is_iframe, LAMBDA_BITRATE
                    )
                    
                    if not is_iframe and y_prev is not None:
                        tcp_v = F.mse_loss(y_curr, y_prev)
                        v_loss += COHERENCE_WEIGHT * tcp_v

                    val_loss += v_loss.item()
                    val_dist  += dist_v.item()
                    val_bpp  += bpp_v.item()
                    
                    f_hat_prev = f_hat_curr
                    y_prev     = y_curr
                
                val_bar.set_postfix({"val_dist": f"{dist_v.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

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

        n_train = len(train_loader)
        n_val   = len(val_loader)

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
