"""
src/continuous/schrodinger.py
------------------------------
Continuous-time PINN for the 1D nonlinear Schrödinger equation.

Replicates Section 2.2 of Raissi et al. (2017).

PDE (complex-valued):
    i h_t + 0.5 h_xx + |h|^2 h = 0,   x in [-5,5], t in [0, pi/2]
    h(0, x)    = 2 sech(x)              (initial condition)
    h(t,-5)    = h(t,5)                 (periodic BC)
    h_x(t,-5)  = h_x(t,5)              (periodic BC on derivative)

Because h is complex, we split h = u + iv and train a 2-output network.
"""

import torch
import numpy as np
from .pinn import MLP


class SchrodingerPINN(torch.nn.Module):
    """
    Multi-output PINN for the nonlinear Schrödinger equation.

    Network: (t, x) → (u, v)  where h = u + iv
    Architecture from paper: 5 layers, 100 neurons/layer, tanh.

    Loss = MSE_0 (initial data) + MSE_b (periodic BC) + MSE_f (PDE residual)
    """

    def __init__(self, n_layers: int = 5, n_neurons: int = 100):
        super().__init__()
        layer_sizes = [2] + [n_neurons] * (n_layers - 1) + [2]
        self.net = MLP(layer_sizes)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Returns (u, v) — real and imaginary parts of h(t,x)."""
        inp = torch.cat([t, x], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2]   # u, v

    def physics_residual(self, t: torch.Tensor, x: torch.Tensor):
        """
        f1 = u_t + 0.5 v_xx + (u^2 + v^2) v   (imaginary part of residual)
        f2 = v_t - 0.5 u_xx - (u^2 + v^2) u   (real part)

        Returns (f1, f2) — both should be ~0 at collocation points.
        """
        u, v = self.forward(t, x)

        def grad(y, z):
            return torch.autograd.grad(
                y, z, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True
            )[0]

        u_t = grad(u, t);  v_t = grad(v, t)
        u_x = grad(u, x);  v_x = grad(v, x)
        u_xx = grad(u_x, x); v_xx = grad(v_x, x)

        h2 = u ** 2 + v ** 2  # |h|^2

        # i h_t + 0.5 h_xx + |h|^2 h = 0
        # Real part: -v_t + 0.5 u_xx + |h|^2 u = 0  → f2
        # Imag part:  u_t + 0.5 v_xx + |h|^2 v = 0  → f1
        f1 = u_t + 0.5 * v_xx + h2 * v
        f2 = v_t - 0.5 * u_xx - h2 * u
        return f1, f2

    def loss(self, data: dict) -> tuple:
        """
        Compute MSE_0 + MSE_b + MSE_f.

        data keys:
            x0, h0_u, h0_v  : initial condition (N0 points)
            t_b             : boundary time points (Nb points)
            t_f, x_f        : collocation points (Nf points)
        """
        # --- MSE_0: initial condition at t=0 ---
        t0 = torch.zeros_like(data['x0'])
        u0_pred, v0_pred = self.forward(t0, data['x0'])
        mse_0 = torch.mean((u0_pred - data['h0_u'])**2 +
                           (v0_pred - data['h0_v'])**2)

        # --- MSE_b: periodic boundary conditions ---
        t_b = data['t_b']
        x_left  = -5.0 * torch.ones_like(t_b, requires_grad=True)
        x_right =  5.0 * torch.ones_like(t_b, requires_grad=True)

        u_l, v_l = self.forward(t_b, x_left)
        u_r, v_r = self.forward(t_b, x_right)

        # h(-5) = h(5)
        mse_b = torch.mean((u_l - u_r)**2 + (v_l - v_r)**2)

        # h_x(-5) = h_x(5)
        u_lx = torch.autograd.grad(u_l, x_left,  grad_outputs=torch.ones_like(u_l), create_graph=True)[0]
        u_rx = torch.autograd.grad(u_r, x_right, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
        v_lx = torch.autograd.grad(v_l, x_left,  grad_outputs=torch.ones_like(v_l), create_graph=True)[0]
        v_rx = torch.autograd.grad(v_r, x_right, grad_outputs=torch.ones_like(v_r), create_graph=True)[0]
        mse_b += torch.mean((u_lx - u_rx)**2 + (v_lx - v_rx)**2)

        # --- MSE_f: PDE residual ---
        f1, f2 = self.physics_residual(data['t_f'], data['x_f'])
        mse_f = torch.mean(f1**2 + f2**2)

        total = mse_0 + mse_b + mse_f
        return total, mse_0, mse_b, mse_f
