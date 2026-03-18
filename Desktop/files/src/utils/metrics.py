"""
src/utils/metrics.py
---------------------
Evaluation metrics used throughout the paper.
"""

import torch
import numpy as np


def relative_l2_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    """
    Relative L2 error: ||u_pred - u_exact||_2 / ||u_exact||_2

    This is the primary metric reported in all tables and figures.
    A value of 6.7e-4 means 0.067% relative error.

    Args:
        u_pred  : predicted values (any shape)
        u_exact : ground-truth values (same shape)

    Returns:
        scalar float
    """
    with torch.no_grad():
        num = torch.norm(u_pred - u_exact)
        den = torch.norm(u_exact)
        return (num / den).item()


def relative_l2_numpy(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """NumPy version — useful when loading reference solutions from files."""
    num = np.linalg.norm(u_pred - u_exact)
    den = np.linalg.norm(u_exact)
    return float(num / den)
