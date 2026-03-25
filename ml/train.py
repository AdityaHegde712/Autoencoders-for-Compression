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

from ml.dataset import ViratDataset
from ml.models.autoencoder import AsymmetricAutoencoder

# --- 1. CONFIGURATION ---
DATA_PATH = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'
BATCH_SIZE = 16 # Faster training with extracted frames
LEARNING_RATE = 3e-4
LAMBDA_BITRATE = 0.01
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
# Remaining 0.10 is the test split (reserved, not loaded during training)

# POC caps — set to None to use the full dataset
TRAIN_MAX_SAMPLES = 375 * BATCH_SIZE  # 6,000 sequences
VAL_MAX_SAMPLES   = 75  * BATCH_SIZE  # 1,200 sequences
PATIENCE    = 10
IFRAME_PROB = 0.10  # POC: ~10% I-frames for broad coverage
                    # Production target: 1/30 ≈ 0.033 (one I-frame per second at 30fps, GOP=30)
ALPHA_SSIM  = 0.84  # Weight of SSIM in the distortion loss (1-ALPHA_SSIM = L1 weight)
                    # 0.84 is a well-validated value from perceptual compression literature


def run_name_generator():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_NAME = run_name_generator()

# ---------------------------------------------------------------------------
# --- LOSS HELPERS ----------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_distortion_loss(
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


def compute_bpp_loss(likelihoods: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    distortion = compute_distortion_loss(output, target, ssim_module, is_iframe, alpha)
    bpp        = compute_bpp_loss(likelihoods, target)
    total      = distortion + lambda_rate * bpp
    return total, distortion, bpp


# ---------------------------------------------------------------------------
# --- TRANSFORMS ------------------------------------------------------------
# ---------------------------------------------------------------------------

class RandomGlobalOrLocal(torch.nn.Module):
    """
    Custom transform that randomly switches between two strategies per sample:

    - LOCAL  (1 - global_prob): RandomCrop(crop_size)
        Teaches fine-grained texture and motion detail.

    - GLOBAL (global_prob):     Resize((crop_size, crop_size))
        Teaches the full spatial layout of the scene by squeezing the whole
        frame down to the target size. Aspect ratio is slightly distorted
        but global structure (roads, buildings, open areas) is preserved.

    Both paths produce the same output shape, so they can be mixed freely
    in the same DataLoader batch.
    """
    def __init__(self, crop_size: int, global_prob: float = 0.20):
        super().__init__()
        self.crop        = transforms.RandomCrop(crop_size)
        self.resize      = transforms.Resize((crop_size, crop_size), antialias=True)
        self.global_prob = global_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.global_prob:
            return self.resize(x)   # Full-frame global view
        return self.crop(x)         # Local detail crop

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"crop={self.crop.size}, global_prob={self.global_prob})")


# ---------------------------------------------------------------------------
# --- TRAINING --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train():
    print(f"Using device: {DEVICE}")
    save_folder = f"ml/models/saved/{RUN_NAME}"
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
    all_folders = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
    random.seed(42)
    random.shuffle(all_folders)

    n = len(all_folders)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)

    train_folders = all_folders[:train_end]
    val_folders   = all_folders[train_end:val_end]
    # test_folders  = all_folders[val_end:]  # Reserved for final evaluation

    print(f"Split: {len(train_folders)} train | {len(val_folders)} val | {n - val_end} test videos")

    # RandomGlobalOrLocal: 80% local crop (detail), 20% global resize (scene layout)
    transform = RandomGlobalOrLocal(crop_size=352, global_prob=0.20)

    train_dataset = ViratDataset(DATA_PATH, sequence_len=2, transform=transform, video_folders=train_folders, max_samples=TRAIN_MAX_SAMPLES)
    val_dataset   = ViratDataset(DATA_PATH, sequence_len=2, transform=transform, video_folders=val_folders,   max_samples=VAL_MAX_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Batches: {len(train_loader)} train | {len(val_loader)} val")

    # --- 3. MODEL, OPTIMIZER, SCHEDULER & LOSS ---
    model = AsymmetricAutoencoder(in_channels=3, latent_channels=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # SSIM module (shared between train and val, no trainable params)
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)

    # OneCycleLR: ramps LR up then down over the full training run.
    # steps_per_epoch = number of train batches; pct_start=0.3 (default).
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,
    )

    # --- 4. MAIN LOOP ---
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- TRAINING PASS ---
        model.train()
        train_loss, train_dist, train_bpp = 0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=True)
        for i, frames in enumerate(train_bar):
            frames = frames.to(DEVICE)
            f_prev, f_curr = frames[:, 0], frames[:, 1]

            # I-frame: compress the full current frame (no reference)
            # P-frame: compress the residual (current - previous)
            is_iframe = random.random() < IFRAME_PROB
            target = f_curr if is_iframe else (f_curr - f_prev)

            optimizer.zero_grad()
            res_hat, p_y = model(target, training=True)

            loss, dist_loss, bpp_loss = compute_loss(
                res_hat, target, p_y, ssim_module, is_iframe, LAMBDA_BITRATE
            )

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                train_dist  += dist_loss.item()
                train_bpp  += bpp_loss.item()

            frame_type = "I" if is_iframe else "P"
            train_bar.set_postfix({
                "type": frame_type,
                "loss": f"{loss.item():.4f}",
                "dist": f"{dist_loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # --- VALIDATION PASS ---
        model.eval()
        val_loss, val_dist, val_bpp = 0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=True)

        with torch.no_grad():
            for i, frames in enumerate(val_bar):
                frames = frames.to(DEVICE)
                f_prev, f_curr = frames[:, 0], frames[:, 1]

                # Apply same I/P-frame mix during validation for a consistent metric
                is_iframe = random.random() < IFRAME_PROB
                target = f_curr if is_iframe else (f_curr - f_prev)

                res_hat, p_y = model(target, training=False)
                v_loss, dist_v, bpp_v = compute_loss(
                    res_hat, target, p_y, ssim_module, is_iframe, LAMBDA_BITRATE
                )

                val_loss += v_loss.item()
                val_dist  += dist_v.item()
                val_bpp  += bpp_v.item()
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
