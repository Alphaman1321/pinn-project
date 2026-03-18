"""
tests/test_lhs_sampling.py
---------------------------
Unit tests for Latin Hypercube Sampling and relative L2 metric.
"""

import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import latin_hypercube_sample
from src.utils.metrics import relative_l2_error


class TestLHS:
    def test_shape(self):
        pts = latin_hypercube_sample(100, bounds=[(0, 1), (-1, 1)])
        assert pts.shape == (100, 2)

    def test_bounds_respected(self):
        pts = latin_hypercube_sample(1000, bounds=[(0, 1), (-5, 5)])
        assert pts[:, 0].min() >= 0.0
        assert pts[:, 0].max() <= 1.0
        assert pts[:, 1].min() >= -5.0
        assert pts[:, 1].max() <= 5.0

    def test_reproducibility(self):
        p1 = latin_hypercube_sample(50, bounds=[(0, 1)], seed=7)
        p2 = latin_hypercube_sample(50, bounds=[(0, 1)], seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = latin_hypercube_sample(50, bounds=[(0, 1)], seed=1)
        p2 = latin_hypercube_sample(50, bounds=[(0, 1)], seed=2)
        assert not np.allclose(p1, p2)


class TestRelativeL2:
    def test_zero_error(self):
        u = torch.ones(100, 1)
        assert relative_l2_error(u, u) == pytest.approx(0.0)

    def test_known_error(self):
        u_exact = torch.ones(100, 1)
        u_pred  = 2 * torch.ones(100, 1)
        # ||pred - exact|| / ||exact|| = ||ones|| / ||ones|| = 1.0
        assert relative_l2_error(u_pred, u_exact) == pytest.approx(1.0, rel=1e-5)

    def test_small_error(self):
        u_exact = torch.randn(500, 1)
        noise   = 1e-3 * torch.randn_like(u_exact)
        err = relative_l2_error(u_exact + noise, u_exact)
        assert err < 0.1   # noise is small relative to signal
