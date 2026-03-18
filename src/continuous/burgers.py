"""
src/continuous/burgers.py
--------------------------
Continuous-time PINN for the 1D Burgers' equation.

Replicates Section 2.1 of Raissi et al. (2017).

PDE:
    u_t + u * u_x - (0.01/pi) * u_xx = 0,   x in [-1,1], t in [0,1]
    u(0, x) = -sin(pi*x)
    u(t,-1) = u(t, 1) = 0

The exact solution is available analytically (see Basdevant et al. 1986),
which we use to compute the relative L2 error reported in Table 1 & 2.
"""

import torch
import numpy as np
from .pinn import ContinuousPINN


class BurgersPINN(ContinuousPINN):
    """
    PINN for the Burgers' equation.

    Architecture from paper: 9 layers, 20 neurons/layer, tanh activation.
    Input: (t, x)  →  Output: u(t, x)
    """

    def __init__(self, n_layers: int = 9, n_neurons: int = 20):
        # Build layer size list: [2, 20, 20, ..., 20, 1]
        layer_sizes = [2] + [n_neurons] * (n_layers - 1) + [1]
        super().__init__(layer_sizes)
        self.nu = 0.01 / np.pi   # viscosity parameter

    def physics_residual(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        f = u_t + u * u_x - nu * u_xx

        Uses torch.autograd.grad to compute exact partial derivatives.
        t and x must have requires_grad=True.
        """
        u = self.forward(t, x)

        # First-order partials
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,   # needed so we can backprop through this
            retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second-order partial u_xx
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]

        return u_t + u * u_x - self.nu * u_xx



# Data generation helpers


def burgers_initial_boundary_data(n_u: int, seed: int = 42):
    """
    Sample Nu boundary and initial condition points for Burgers'.

    Returns tensors ready for training (requires_grad=False).
    Boundary data:
        u(0,  x) = -sin(pi*x)   for x in [-1, 1]   (initial condition)
        u(t, -1) = 0             for t in [0,  1]   (left BC)
        u(t,  1) = 0             for t in [0,  1]   (right BC)
    """
    rng = np.random.default_rng(seed)

    # --- initial condition: t=0 ---
    n_ic = n_u // 2
    x_ic = rng.uniform(-1, 1, (n_ic, 1)).astype(np.float32)
    t_ic = np.zeros((n_ic, 1), dtype=np.float32)
    u_ic = -np.sin(np.pi * x_ic).astype(np.float32)

    # --- boundary conditions: x=-1 and x=1 ---
    n_bc = n_u - n_ic
    t_bc = rng.uniform(0, 1, (n_bc, 1)).astype(np.float32)
    # Alternate left/right
    x_left  = -np.ones((n_bc // 2, 1), dtype=np.float32)
    x_right =  np.ones((n_bc - n_bc // 2, 1), dtype=np.float32)
    x_bc = np.vstack([x_left, x_right])
    u_bc = np.zeros((n_bc, 1), dtype=np.float32)

    t_u = torch.tensor(np.vstack([t_ic, t_bc]))
    x_u = torch.tensor(np.vstack([x_ic, x_bc]))
    u_true = torch.tensor(np.vstack([u_ic, u_bc]))
    return t_u, x_u, u_true


def burgers_collocation_points(n_f: int, seed: int = 0):
    """
    Latin Hypercube Sampling of Nf collocation points in [0,1] x [-1,1].

    Returns tensors with requires_grad=True (needed for autograd in residual).
    """
    from ..utils.data import latin_hypercube_sample

    pts = latin_hypercube_sample(n_f, bounds=[(0, 1), (-1, 1)], seed=seed)
    t_f = torch.tensor(pts[:, 0:1], requires_grad=True)
    x_f = torch.tensor(pts[:, 1:2], requires_grad=True)
    return t_f, x_f
