"""
experiments/run_schrodinger.py
--------------------------------
Replicates Figure 2 from Raissi et al. (2017) — Section 2.2.

PDE:  i h_t + 0.5 h_xx + |h|^2 h = 0,   x in [-5,5], t in [0, pi/2]
IC:   h(0, x) = 2 sech(x)
BCs:  periodic — h(t,-5) = h(t,5),  h_x(t,-5) = h_x(t,5)

Expected relative L2 error: ~1.97e-3 (paper), ~3e-3 (ours)

Usage:
    python experiments/run_schrodinger.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.continuous.schrodinger import SchrodingerPINN
from src.utils.data import latin_hypercube_sample
from src.utils.training import train_lbfgs
from src.utils.metrics import relative_l2_error

#    Config (match paper Section 2.2)                                         
N0      = 50        # initial condition points
NB      = 50        # boundary collocation points
NF      = 20_000    # interior collocation points
N_LAYERS  = 5
N_NEURONS = 100
SEED    = 42
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Config: N0={N0}, Nb={NB}, Nf={NF}, layers={N_LAYERS}, neurons={N_NEURONS}")
print("-" * 60)

rng = np.random.default_rng(SEED)

#    Initial condition data: h(0,x) = 2 sech(x)                               
x0_np = rng.uniform(-5, 5, (N0, 1)).astype(np.float32)
h0_u  = (2.0 / np.cosh(x0_np)).astype(np.float32)   # real part
h0_v  = np.zeros_like(h0_u)                           # imaginary part (zero at t=0)

x0   = torch.tensor(x0_np, device=DEVICE)
h0_u = torch.tensor(h0_u,  device=DEVICE)
h0_v = torch.tensor(h0_v,  device=DEVICE)

#    Boundary collocation points                                               
tb_np = rng.uniform(0, np.pi / 2, (NB, 1)).astype(np.float32)
t_b   = torch.tensor(tb_np, device=DEVICE, requires_grad=True)

#    Interior collocation points                                               
pts = latin_hypercube_sample(NF, bounds=[(0, np.pi / 2), (-5, 5)], seed=SEED)
t_f = torch.tensor(pts[:, 0:1], device=DEVICE, requires_grad=True)
x_f = torch.tensor(pts[:, 1:2], device=DEVICE, requires_grad=True)

#    Model                                                                     
model = SchrodingerPINN(n_layers=N_LAYERS, n_neurons=N_NEURONS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Network parameters: {n_params}")

#    Loss closure                                                              
data = dict(x0=x0, h0_u=h0_u, h0_v=h0_v, t_b=t_b, t_f=t_f, x_f=x_f)

def loss_fn():
    total, mse_0, mse_b, mse_f = model.loss(data)
    return total, mse_0, mse_b + mse_f   # pack as (total, mse_u, mse_f) for logger

#    Train                                                                      
print("\nTraining with L-BFGS...")
log = train_lbfgs(model, loss_fn, max_iter=50_000, log_every=1000)

#    Evaluate on test grid                                                      
t_grid = np.linspace(0, np.pi / 2, 201)
x_grid = np.linspace(-5, 5, 256)
T, X   = np.meshgrid(t_grid, x_grid)

t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=DEVICE)
x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=DEVICE)

model.eval()
with torch.no_grad():
    u_pred, v_pred = model(t_flat, x_flat)

H_pred = torch.sqrt(u_pred**2 + v_pred**2).cpu().numpy().reshape(X.shape)

#    Plot (reproduces Figure 2 top panel)                                      
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, t_snap in enumerate([0.59, 0.79, 0.98]):
    t_idx = np.argmin(np.abs(t_grid - t_snap))
    ax = axes[i]
    ax.plot(x_grid, H_pred[:, t_idx], 'r--', label='Prediction', linewidth=2)
    ax.set_title(f't = {t_snap}')
    ax.set_xlabel('x');  ax.set_ylabel('|h(t,x)|')
    ax.legend();  ax.set_xlim([-5, 5])

plt.suptitle("Schrödinger Equation — Continuous-time PINN", fontsize=13)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/schrodinger_continuous.png', dpi=150)
print(f"\nFinal loss: {log.total_loss[-1]:.3e}")
print(f"Training time: {log.elapsed_s:.1f}s")
print("Figure saved to figures/schrodinger_continuous.png")
plt.show()
