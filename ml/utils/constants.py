"""
Shared project constants and configuration.
Central source of truth for paths, dataset splits, and device detection.
All scripts should import from here rather than redefining these values.
"""
from pathlib import Path
import sys

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_frames"
MODEL_SAVE_DIR = PROJECT_ROOT / "ml" / "models" / "saved"

# Ensure the project root is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Dataset splits ───────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
# Test split is implicitly 1.0 - TRAIN_SPLIT - VAL_SPLIT = 0.10

# ── Pipeline constants ───────────────────────────────────────────────────────
IFRAME_PROB = 0.10  # Probability of I-frame when not using fixed GOP

def get_device():
    """Returns the available torch device (cuda if available, else cpu)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
