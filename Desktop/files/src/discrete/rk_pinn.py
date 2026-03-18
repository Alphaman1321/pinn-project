"""
src/discrete/rk_pinn.py
------------------------
Discrete-time PINN using implicit Runge-Kutta time-stepping.

Replicates Section 3 of Raissi et al. (2017).

Key idea: instead of placing collocation points across the full
spatio-temporal domain, we use a q-stage implicit Runge-Kutta scheme
to step from time t^n to t^{n+1}.

The network takes x as input and outputs q+1 quantities:
    [u^{n+c1}(x), ..., u^{n+cq}(x), u^{n+1}(x)]

The Butcher tableau (a_ij, b_j, c_j) for the implicit RK scheme is
precomputed using Gauss-Legendre quadrature nodes.  With q stages the
method is order 2q, yielding temporal error O(Δt^{2q}).

For q=500, Δt=0.8:  Δt^{2q} = 0.8^1000 ≈ 10^{-97}  (below machine precision)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
from ..utils.data import gauss_legendre_butcher


class RKPINN(nn.Module):
    """
    Discrete-time implicit Runge-Kutta PINN.

    Args:
        layer_sizes : [input=1, h1, ..., output=(q+1)] — x → (u stages + u_next)
        q           : number of Runge-Kutta stages
        dt          : time-step size Δt
    """

    def __init__(self, layer_sizes: List[int], q: int, dt: float):
        super().__init__()
        self.q = q
        self.dt = dt

        # Build the network: input is x (1D), output is q+1 values
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])   # remove last Tanh

        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Butcher tableau (CPU tensors; moved to device in forward)
        IRK_weights, IRK_times = gauss_legendre_butcher(q)
        # IRK_weights: (q, q) array — a_ij coefficients
        # IRK_times:   (q,)   array — c_i  nodes
        self.register_buffer('IRK_alpha', torch.tensor(IRK_weights, dtype=torch.float32))
        self.register_buffer('IRK_beta',  torch.tensor(IRK_times,   dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1) spatial locations
        Returns: (N, q+1) — intermediate stage solutions + final solution
        """
        return self.net(x)

    def nonlinear_operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        N[u] = nonlinear PDE operator applied to u(x).
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def loss(
        self,
        x_n: torch.Tensor,
        u_n: torch.Tensor,
        x_boundary: List[torch.Tensor],
    ) -> tuple:
        """
        Compute SSE = SSE_n + SSE_b.

        SSE_n: ensures the Runge-Kutta update equations are satisfied at
               the Nn spatial data points from time t^n.
        SSE_b: enforces boundary conditions at every RK stage and final step.

        Args:
            x_n       : (Nn, 1) spatial data points at time t^n
            u_n       : (Nn, 1) observed solution values at t^n
            x_boundary: list of boundary x-values to enforce BCs at
        """
        # Network outputs: (Nn, q+1)
        U = self.forward(x_n)               # U[:,  :q] = stages, U[:, q] = u^{n+1}
        U_stages = U[:, :self.q]            # (Nn, q)
        U_next   = U[:, self.q:self.q+1]    # (Nn, 1)

        # Compute N[u^{n+c_j}] for each stage j
        N_stages = []
        for j in range(self.q):
            u_j = U_stages[:, j:j+1]
            u_j = u_j.detach().requires_grad_(True)   # fresh leaf for autograd
            # We need to call the spatial derivative through the network
            # Re-evaluate u_j through network to keep the graph connected
            N_j = self.nonlinear_operator(U_stages[:, j:j+1], x_n)
            N_stages.append(N_j)
        N = torch.stack(N_stages, dim=1).squeeze(-1)    # (Nn, q)

        # Build u^n predictions from RK formula: eq (8)-(9) in paper
        # u^n_i = u^{n+c_i} + dt * sum_j a_{ij} N[u^{n+c_j}]   for i=1..q
        # u^n_{q+1} = u^{n+1} + dt * sum_j b_j   N[u^{n+c_j}]

        # IRK_alpha: (q, q), N: (Nn, q)
        U_n_pred_stages = U_stages + self.dt * (N @ self.IRK_alpha.T)   # (Nn, q)
        U_n_pred_final  = U_next   + self.dt * (N @ self.IRK_beta.unsqueeze(1))  # (Nn, 1)

        # SSE_n: each of the q+1 predictions should equal u_n
        u_n_expanded = u_n.expand(-1, self.q)    # (Nn, q)
        sse_n = torch.sum((U_n_pred_stages - u_n_expanded) ** 2)
        sse_n += torch.sum((U_n_pred_final  - u_n) ** 2)

        # SSE_b: boundary conditions (subclass provides bc_loss)
        sse_b = self.boundary_loss(x_boundary)

        return sse_n + sse_b, sse_n, sse_b

    def boundary_loss(self, x_boundary: List[torch.Tensor]) -> torch.Tensor:
        """Compute boundary condition loss.  Override in subclasses."""
        raise NotImplementedError
