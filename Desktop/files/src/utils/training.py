"""
src/utils/training.py
----------------------
Training utilities: L-BFGS full-batch training (as used in the paper)
and an optional Adam warmup stage.
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TrainingLog:
    """Stores loss history for plotting and analysis."""
    iterations: list = field(default_factory=list)
    total_loss: list = field(default_factory=list)
    mse_u: list = field(default_factory=list)
    mse_f: list = field(default_factory=list)
    elapsed_s: float = 0.0


def train_lbfgs(
    model: torch.nn.Module,
    loss_fn: Callable[[], tuple],
    max_iter: int = 50_000,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    log_every: int = 500,
    verbose: bool = True,
) -> TrainingLog:
    """
    Full-batch L-BFGS training loop.

    The paper uses L-BFGS for all experiments.  PyTorch's LBFGS requires
    a closure that re-evaluates the loss and calls backward().

    Args:
        model       : PINN model (nn.Module)
        loss_fn     : callable returning (total, mse_u, mse_f) — no arguments
        max_iter    : maximum L-BFGS iterations
        log_every   : print / record loss every N iterations

    Returns:
        TrainingLog with full history
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=50,
        line_search_fn='strong_wolfe',
    )
    log = TrainingLog()
    iteration = [0]
    t0 = time.time()

    def closure():
        optimizer.zero_grad()
        total, mse_u, mse_f = loss_fn()
        total.backward()

        it = iteration[0]
        if it % log_every == 0:
            log.iterations.append(it)
            log.total_loss.append(total.item())
            log.mse_u.append(mse_u.item())
            log.mse_f.append(mse_f.item())
            if verbose:
                print(f"  iter {it:6d} | loss {total.item():.3e} "
                      f"| MSE_u {mse_u.item():.3e} | MSE_f {mse_f.item():.3e}")
        iteration[0] += 1
        return total

    optimizer.step(closure)
    log.elapsed_s = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {log.elapsed_s:.1f}s")
    return log


def train_adam(
    model: torch.nn.Module,
    loss_fn: Callable[[], tuple],
    n_iter: int = 5_000,
    lr: float = 1e-3,
    log_every: int = 500,
    verbose: bool = True,
) -> TrainingLog:
    """
    Adam optimizer training (optional warmup before L-BFGS).

    Useful for larger datasets or when L-BFGS struggles with initialization.
    Adam is NOT used in the original paper but is a natural baseline/extension.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = TrainingLog()
    t0 = time.time()

    for it in range(n_iter):
        optimizer.zero_grad()
        total, mse_u, mse_f = loss_fn()
        total.backward()
        optimizer.step()

        if it % log_every == 0:
            log.iterations.append(it)
            log.total_loss.append(total.item())
            log.mse_u.append(mse_u.item())
            log.mse_f.append(mse_f.item())
            if verbose:
                print(f"  iter {it:6d} | loss {total.item():.3e} "
                      f"| MSE_u {mse_u.item():.3e} | MSE_f {mse_f.item():.3e}")

    log.elapsed_s = time.time() - t0
    return log
