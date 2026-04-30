import torch
import numpy as np

def residual_crop_from_two_frames(
    prev_frame: torch.Tensor,
    curr_frame: torch.Tensor,
    block_size: int = 352,
    eps: float = 1e-8,
):
    """
    Build residual (curr - prev), find non-zero motion bounds, and return a crop
    whose width/height are multiples of block_size.

    Args:
        prev_frame: Tensor (C,H,W), (1,C,H,W), (H,W,C), or (1,H,W,C)
        curr_frame: Same shape/layout as prev_frame
        block_size: Required crop granularity (default 352)
        eps: Motion threshold for non-zero mask

    Returns:
        cropped_residual: Tensor (C, Hc, Wc)
        coords: (top, left, bottom, right) in CHW space (bottom/right exclusive)
    """

    def _to_chw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError("Batch dimension must be 1 for 4D input.")
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")
        # HWC -> CHW
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1)
        return x.float()

    prev = _to_chw(prev_frame)
    curr = _to_chw(curr_frame)
    if prev.shape != curr.shape:
        raise ValueError(f"Shape mismatch: prev {tuple(prev.shape)} vs curr {tuple(curr.shape)}")

    residual = curr - prev
    _, h, w = residual.shape

    # Any channel moving at a pixel marks it as non-zero motion.
    motion_mask = residual.abs().amax(dim=0) > eps  # (H, W)
    nonzero = motion_mask.nonzero(as_tuple=False)

    if nonzero.numel() == 0:
        # No motion: return smallest valid top-left block.
        crop_h = min(block_size, h)
        crop_w = min(block_size, w)
        return residual[:, :crop_h, :crop_w], (0, 0, crop_h, crop_w)

    top = int(nonzero[:, 0].min().item())
    bottom = int(nonzero[:, 0].max().item()) + 1
    left = int(nonzero[:, 1].min().item())
    right = int(nonzero[:, 1].max().item()) + 1

    box_h = bottom - top
    box_w = right - left
    target_h = int(np.ceil(box_h / block_size) * block_size)
    target_w = int(np.ceil(box_w / block_size) * block_size)

    extra_h = target_h - box_h
    extra_w = target_w - box_w
    top -= extra_h // 2
    bottom += extra_h - (extra_h // 2)
    left -= extra_w // 2
    right += extra_w - (extra_w // 2)

    # Clamp to image bounds, then slide window to keep exact target size if possible.
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    if bottom - top < target_h:
        if top == 0:
            bottom = min(h, top + target_h)
        elif bottom == h:
            top = max(0, bottom - target_h)
    if right - left < target_w:
        if left == 0:
            right = min(w, left + target_w)
        elif right == w:
            left = max(0, right - target_w)

    cropped = residual[:, top:bottom, left:right]
    return cropped, (top, left, bottom, right)


def residual_crop_from_sequence(
    frames_tchw: torch.Tensor,
    t_prev: int,
    t_curr: int,
    block_size: int = 352,
    eps: float = 1e-8,
):
    """
    Compute residual crop from a frame sequence shaped (T, C, H, W).

    Args:
        frames_tchw: Tensor with shape (T, C, H, W), where T is time index.
        t_prev: Previous-frame time index.
        t_curr: Current-frame time index.
        block_size: Required crop granularity (default 352).
        eps: Motion threshold for non-zero mask.

    Returns:
        cropped_residual: Tensor (C, Hc, Wc)
        coords: (top, left, bottom, right)
    """
    if frames_tchw.ndim != 4:
        raise ValueError(f"Expected input shape (T, C, H, W), got {tuple(frames_tchw.shape)}")
    t = frames_tchw.shape[0]
    if not (0 <= t_prev < t and 0 <= t_curr < t):
        raise IndexError(f"Time indices out of range for T={t}: t_prev={t_prev}, t_curr={t_curr}")

    prev_frame = frames_tchw[t_prev]
    curr_frame = frames_tchw[t_curr]
    return residual_crop_from_two_frames(
        prev_frame=prev_frame,
        curr_frame=curr_frame,
        block_size=block_size,
        eps=eps,
    )


def restore_residual_from_crop(
    cropped_residual: torch.Tensor,
    coords: tuple[int, int, int, int],
    full_shape: tuple[int, int, int],
):
    """
    Reconstruct a full residual tensor from a cropped residual and crop coordinates.
    Areas outside the crop are filled with zeros.

    Args:
        cropped_residual: Tensor with shape (C, Hc, Wc)
        coords: (top, left, bottom, right) with bottom/right exclusive
        full_shape: Target full residual shape (C, H, W)

    Returns:
        full_residual: Tensor with shape (C, H, W)
    """
    if cropped_residual.ndim != 3:
        raise ValueError(
            f"Expected cropped_residual shape (C, Hc, Wc), got {tuple(cropped_residual.shape)}"
        )
    if len(full_shape) != 3:
        raise ValueError(f"Expected full_shape (C, H, W), got {full_shape}")

    c_full, h_full, w_full = full_shape
    c_crop, h_crop, w_crop = cropped_residual.shape
    if c_crop != c_full:
        raise ValueError(f"Channel mismatch: cropped C={c_crop}, full C={c_full}")

    top, left, bottom, right = coords
    if not (0 <= top < bottom <= h_full and 0 <= left < right <= w_full):
        raise ValueError(
            f"Invalid coords {coords} for full_shape (C={c_full}, H={h_full}, W={w_full})"
        )
    if (bottom - top) != h_crop or (right - left) != w_crop:
        raise ValueError(
            "Crop size mismatch: coords imply "
            f"({bottom - top}, {right - left}) but cropped is ({h_crop}, {w_crop})"
        )

    full_residual = torch.zeros(
        (c_full, h_full, w_full),
        dtype=cropped_residual.dtype,
        device=cropped_residual.device,
    )
    full_residual[:, top:bottom, left:right] = cropped_residual
    return full_residual

