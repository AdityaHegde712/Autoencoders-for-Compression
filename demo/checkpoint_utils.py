"""Helpers for loading weights saved under different wrappers (e.g. torch.compile)."""

from __future__ import annotations


def strip_torch_compile_prefix(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix from keys produced by torch.compile() state_dict()."""
    prefix = "_orig_mod."
    if not any(str(k).startswith(prefix) for k in state_dict):
        return state_dict
    return {
        (str(k)[len(prefix) :] if str(k).startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
