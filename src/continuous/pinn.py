"""
src/continuous/pinn.py
----------------------
Base class for continuous-time Physics-Informed Neural Networks.

Implements the shared architecture and training logic described in
Raissi et al. (2017), Section 2.  Subclasses override `physics_residual`
to define the specific PDE being solved.
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional


class MLP(nn.Module):
    """Fully-connected network with uniform hidden layers and tanh activations.

    Architecture used throughout the paper:
        Input → [Linear → tanh] x (n_layers-1) → Linear → Output
    """

    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
        # Remove final activation (output layer is linear)
        self.net = nn.Sequential(*layers[:-1])
        self._init_weights()

    def _init_weights(self):
        """Xavier / Glorot initialization — standard for tanh networks."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousPINN(nn.Module):
    """
    Continuous-time PINN base class.

    The total loss is:
        MSE = MSE_u + MSE_f

    where:
        MSE_u = mean squared error at labeled boundary / initial data
        MSE_f = mean squared physics residual at collocation points

    Setting n_collocation=0 recovers a plain supervised neural network,
    which we use as a baseline in the noise robustness extension.

    Args:
        layer_sizes: list of ints [input_dim, h1, h2, ..., output_dim]
                     e.g. [2, 20, 20, ..., 20, 1] for a (t,x) -> u network
    """

    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.net = MLP(layer_sizes)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate u(t, x).  Inputs must have requires_grad=True for autograd."""
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)

    def physics_residual(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(t,x) = PDE residual evaluated at (t,x).

        Must be overridden by subclasses to define the specific PDE.
        The residual f should equal 0 everywhere in the domain if u is
        the true solution.
        """
        raise NotImplementedError("Subclasses must implement physics_residual()")

    def loss(
        self,
        t_u: torch.Tensor,
        x_u: torch.Tensor,
        u_true: torch.Tensor,
        t_f: torch.Tensor,
        x_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = MSE_u + MSE_f.

        Args:
            t_u, x_u : boundary/initial data locations  (Nu x 1)
            u_true   : observed values at those points   (Nu x 1)
            t_f, x_f : collocation points                (Nf x 1)

        Returns:
            total_loss, mse_u, mse_f
        """
        # Data loss
        u_pred = self.forward(t_u, x_u)
        mse_u = torch.mean((u_pred - u_true) ** 2)

        # Physics loss (MSE_f = 0 when Nf = 0, i.e., baseline NN)
        if t_f is not None and t_f.shape[0] > 0:
            f_pred = self.physics_residual(t_f, x_f)
            mse_f = torch.mean(f_pred ** 2)
        else:
            mse_f = torch.tensor(0.0, device=t_u.device)

        return mse_u + mse_f, mse_u, mse_f
