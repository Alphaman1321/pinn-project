"""
tests/test_pinn_forward.py
---------------------------
Sanity checks for the PINN models: parameter counts, output shapes,
and physics residual computations.
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.continuous.burgers import BurgersPINN
from src.continuous.pinn import MLP


class TestMLP:
    def test_output_shape(self):
        net = MLP([2, 20, 20, 1])
        x = torch.randn(50, 2)
        out = net(x)
        assert out.shape == (50, 1)

    def test_xavier_init(self):
        """Weights should not be zero after initialization."""
        net = MLP([2, 20, 1])
        for m in net.net.modules():
            if hasattr(m, 'weight'):
                assert not torch.all(m.weight == 0)


class TestBurgersPINN:
    def setup_method(self):
        self.model = BurgersPINN(n_layers=9, n_neurons=20)

    def test_parameter_count(self):
        """Paper reports 3021 parameters for 9-layer, 20-neuron network."""
        n = sum(p.numel() for p in self.model.parameters())
        # 9-layer tanh network: [2→20, 20→20 ×7, 20→1]
        # = (2*20+20) + 7*(20*20+20) + (20*1+1) = 60 + 2940 + 21 = 3021
        assert n == 3021, f"Expected 3021 params, got {n}"

    def test_forward_shape(self):
        t = torch.randn(32, 1)
        x = torch.randn(32, 1)
        u = self.model(t, x)
        assert u.shape == (32, 1)

    def test_physics_residual_shape(self):
        t = torch.randn(64, 1, requires_grad=True)
        x = torch.randn(64, 1, requires_grad=True)
        f = self.model.physics_residual(t, x)
        assert f.shape == (64, 1)

    def test_loss_returns_three_values(self):
        t_u = torch.randn(20, 1)
        x_u = torch.randn(20, 1)
        u_t = torch.randn(20, 1)
        t_f = torch.randn(50, 1, requires_grad=True)
        x_f = torch.randn(50, 1, requires_grad=True)
        total, mse_u, mse_f = self.model.loss(t_u, x_u, u_t, t_f, x_f)
        assert total.item() >= 0
        assert mse_u.item() >= 0
        assert mse_f.item() >= 0

    def test_baseline_nn_mse_f_is_zero(self):
        """With Nf=0, the physics loss should be exactly 0."""
        t_u = torch.randn(20, 1)
        x_u = torch.randn(20, 1)
        u_t = torch.randn(20, 1)
        total, mse_u, mse_f = self.model.loss(t_u, x_u, u_t, None, None)
        assert mse_f.item() == pytest.approx(0.0)
        assert total.item() == pytest.approx(mse_u.item())
