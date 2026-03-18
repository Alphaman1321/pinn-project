"""
src/discrete/allen_cahn.py
---------------------------
Discrete-time PINN for the Allen-Cahn equation.

Replicates Section 3.1.1 of Raissi et al. (2017).

PDE:
    u_t - 0.0001 u_xx + 5u^3 - 5u = 0,   x in [-1,1], t in [0,1]
    u(0,  x) = x^2 cos(pi*x)
    u(t, -1) = u(t, 1)          (periodic)
    u_x(t,-1) = u_x(t, 1)      (periodic derivative)

Network: 4 hidden layers, 200 neurons/layer
RK stages: q=100, time step: Δt=0.8  (t^n=0.1 → t^{n+1}=0.9)
"""

import torch
import numpy as np
from .rk_pinn import RKPINN


class AllenCahnPINN(RKPINN):
    """
    Discrete-time PINN for the Allen-Cahn equation.

    The nonlinear operator is:
        N[u] = -0.0001 * u_xx + 5*u^3 - 5*u
    """

    def __init__(self, q: int = 100, dt: float = 0.8,
                 n_layers: int = 4, n_neurons: int = 200):
        # Output size = q+1 (q stages + final step)
        layer_sizes = [1] + [n_neurons] * n_layers + [q + 1]
        super().__init__(layer_sizes, q=q, dt=dt)

    def nonlinear_operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        N[u] = -0.0001 * u_xx + 5*u^3 - 5*u

        Args:
            u : (N, 1) solution values at current stage
            x : (N, 1) spatial locations, requires_grad=True
        """
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        return -0.0001 * u_xx + 5.0 * u**3 - 5.0 * u

    def boundary_loss(self, x_boundary):
        """
        Periodic BCs: u(-1) = u(1) and u_x(-1) = u_x(1) at all RK stages.
        """
        x_l = torch.tensor([[-1.0]], device=next(self.parameters()).device,
                            requires_grad=True)
        x_r = torch.tensor([[ 1.0]], device=next(self.parameters()).device,
                            requires_grad=True)

        U_l = self.forward(x_l)   # (1, q+1)
        U_r = self.forward(x_r)   # (1, q+1)

        # u(-1) = u(1) for all stages + final
        sse_b = torch.sum((U_l - U_r) ** 2)

        # u_x(-1) = u_x(1)
        for j in range(self.q + 1):
            ux_l = torch.autograd.grad(U_l[0, j], x_l, create_graph=True)[0]
            ux_r = torch.autograd.grad(U_r[0, j], x_r, create_graph=True)[0]
            sse_b = sse_b + (ux_l - ux_r) ** 2

        return sse_b.squeeze()
