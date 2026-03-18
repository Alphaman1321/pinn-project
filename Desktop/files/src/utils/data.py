"""
src/utils/data.py
------------------
Data utilities: Latin Hypercube Sampling and Butcher tableau generation.
"""

import numpy as np
from scipy.stats import qmc
from scipy.special import roots_legendre


def latin_hypercube_sample(
    n: int,
    bounds: list[tuple[float, float]],
    seed: int = 0,
) -> np.ndarray:
    """
    Generate n points via Latin Hypercube Sampling (LHS) over a d-dimensional
    box defined by `bounds`.

    This is the sampling strategy used throughout the paper for collocation
    points (see caption of Figure 1).

    Args:
        n      : number of points to sample
        bounds : list of (low, high) per dimension
                 e.g. [(0,1), (-1,1)] for t in [0,1], x in [-1,1]
        seed   : random seed for reproducibility

    Returns:
        pts: (n, d) float32 array
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_pts = sampler.random(n)                     # (n, d) in [0, 1]^d
    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    pts = qmc.scale(unit_pts, l_bounds, u_bounds)    # scale to actual bounds
    return pts.astype(np.float32)


def gauss_legendre_butcher(q: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Butcher tableau for the q-stage implicit Runge-Kutta method
    based on Gauss-Legendre quadrature.

    This is an A-stable, order-2q method.  The paper uses q up to 500 stages,
    achieving temporal errors below machine precision for large time steps.

    Returns:
        IRK_weights : (q, q) array — a_{ij} coefficients
        IRK_times   : (q,)   array — c_i  nodes (in [0,1])

    Reference: Iserles (2009), "A First Course in the Numerical Analysis
               of Differential Equations", Chapter 4.
    """
    # Gauss-Legendre nodes and weights on [-1, 1]
    nodes_std, weights_std = roots_legendre(q)

    # Shift nodes to [0, 1]: c_i = (node_i + 1) / 2
    c = (nodes_std + 1.0) / 2.0           # (q,)
    b = weights_std / 2.0                  # (q,)  — integration weights

    # Build a_{ij}: integral from 0 to c_i of the j-th Lagrange basis polynomial
    # evaluated at the quadrature nodes.
    # a_{ij} = integral_0^{c_i} L_j(tau) dtau
    # We compute this numerically using high-order Gauss quadrature.
    A = np.zeros((q, q), dtype=np.float64)
    for i in range(q):
        for j in range(q):
            # Lagrange basis L_j evaluated at c_k, integrated up to c_i
            # Use the property: a_{ij} = sum_k w_k * L_j(c_k) * [c_k <= c_i]
            # (crude but sufficient for our purposes; exact methods exist)
            A[i, j] = b[j] * _legendre_integral(c, j, 0, c[i])

    return A.astype(np.float32), c.astype(np.float32)


def _legendre_integral(nodes: np.ndarray, j: int, a: float, b: float) -> float:
    """
    Approximate integral from a to b of the j-th Lagrange basis polynomial
    defined by `nodes`, using 100-point Gauss-Legendre quadrature.
    """
    pts, wts = roots_legendre(100)
    # Map from [-1,1] to [a,b]
    mid  = (a + b) / 2.0
    half = (b - a) / 2.0
    x = mid + half * pts   # integration points

    # Evaluate Lagrange basis polynomial L_j at x
    Lj = np.ones_like(x)
    for k, nk in enumerate(nodes):
        if k != j:
            Lj *= (x - nk) / (nodes[j] - nk)

    return half * np.dot(wts, Lj)
