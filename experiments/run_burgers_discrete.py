"""
experiments/run_burgers_discrete.py
-------------------------------------
Replicates Figure 3 from Raissi et al. (2017).

Loads the exact Burgers' solution from data/burgers_shock.mat,
takes the snapshot at t=0.1 as training data, and predicts t=0.9
in a single RK-32 time step.

Download data first:
    mkdir data
    curl -L "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/burgers_shock.mat" -o data/burgers_shock.mat
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from src.discrete.rk_pinn import RKPINN
from src.utils.training import train_lbfgs, train_adam
from src.utils.metrics import relative_l2_numpy

#    Config                                                                     
NN        = 250   # how many of the t=0.1 points to use for training
Q         = 32    # Runge-Kutta stages (paper uses 500, we use 32)
DT        = 0.8   # time step: t=0.1 -> t=0.9
N_LAYERS  = 4
N_NEURONS = 50
SEED      = 42
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Config: Nn={NN}, q={Q}, dt={DT}")
print("-" * 60)

#    Load exact solution from .mat file                                         
mat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'burgers_shock.mat')

if not os.path.exists(mat_path):
    print("ERROR: data/burgers_shock.mat not found.")
    print("Please run:")
    print("  mkdir data")
    print("  curl -L https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/burgers_shock.mat -o data/burgers_shock.mat")
    sys.exit(1)

data_mat = scipy.io.loadmat(mat_path)
# t: (100,)  x: (256,)  usol: (256, 100) — solution at every (x, t)
t_star   = data_mat['t'].flatten()       # time grid
x_star   = data_mat['x'].flatten()       # space grid
u_star   = data_mat['usol']              # full solution (256 x 100)

print(f"Loaded exact solution: x in [{x_star.min():.2f}, {x_star.max():.2f}], "
      f"t in [{t_star.min():.2f}, {t_star.max():.2f}]")

# Find the t=0.1 and t=0.9 snapshots
t_n_idx   = np.argmin(np.abs(t_star - 0.1))
t_np1_idx = np.argmin(np.abs(t_star - 0.9))
print(f"Using t={t_star[t_n_idx]:.4f} as input, t={t_star[t_np1_idx]:.4f} as target")

# Training data: 250 randomly sampled points from the t=0.1 snapshot
u_n_full = u_star[:, t_n_idx]    # shape (256,)
u_exact_09 = u_star[:, t_np1_idx] # shape (256,) — ground truth at t=0.9

rng = np.random.default_rng(SEED)
idx = rng.choice(len(x_star), NN, replace=False)
x_n_np = x_star[idx][:, None].astype(np.float32)
u_n_np = u_n_full[idx][:, None].astype(np.float32)

x_n     = torch.tensor(x_n_np, device=DEVICE, requires_grad=True)
u_n     = torch.tensor(u_n_np, device=DEVICE)
x_left  = torch.tensor([[-1.0]], device=DEVICE, requires_grad=True)
x_right = torch.tensor([[ 1.0]], device=DEVICE, requires_grad=True)


#    Burgers RK PINN                                                            
class BurgersRKPINN(RKPINN):
    def nonlinear_operator(self, u, x):
        nu = 0.01 / np.pi
        u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),   create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u * u_x - nu * u_xx

    def boundary_loss(self, x_boundary):
        device = next(self.parameters()).device
        sse_b = torch.tensor(0.0, device=device)
        for xb in x_boundary:
            U = self.forward(xb)
            sse_b = sse_b + torch.sum(U ** 2)
        return sse_b

print("Building Runge-Kutta PINN...")
layer_sizes = [1] + [N_NEURONS] * N_LAYERS + [Q + 1]
model = BurgersRKPINN(layer_sizes=layer_sizes, q=Q, dt=DT).to(DEVICE)
print(f"Network parameters: {sum(p.numel() for p in model.parameters())}")

def loss_fn():
    total, sse_n, sse_b = model.loss(x_n, u_n, [x_left, x_right])
    return total, sse_n, sse_b

print("\nAdam warmup (2000 iters)...")
train_adam(model, loss_fn, n_iter=2000, lr=1e-3, log_every=500)

print("\nL-BFGS refinement...")
log = train_lbfgs(model, loss_fn, max_iter=30_000, log_every=1000)


#    Predict and evaluate                                                       
x_test_np = x_star[:, None].astype(np.float32)  # all 256 spatial points
x_test    = torch.tensor(x_test_np, device=DEVICE)

model.eval()
with torch.no_grad():
    U_out  = model(x_test)
    u_pred = U_out[:, Q].cpu().numpy().flatten()

rel_l2 = relative_l2_numpy(u_pred, u_exact_09)
print(f"\nRelative L2 error at t=0.9: {rel_l2:.4e}")
print(f"Paper reports: 8.2e-4")
print(f"Final training loss: {log.total_loss[-1]:.3e}")


#    Plot                                                                       
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].scatter(x_n_np.flatten(), u_n_np.flatten(),
                s=8, color='red', label='Data (t=0.1)', zorder=3)
axes[0].set_title('t = 0.10  (training data)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('u(t, x)')
axes[0].set_xlim([-1, 1]); axes[0].set_ylim([-1.2, 1.2])
axes[0].legend()

axes[1].plot(x_star, u_exact_09, 'b-',  label='Exact',      linewidth=2)
axes[1].plot(x_star, u_pred,     'r--', label='Prediction', linewidth=2)
axes[1].set_title(f't = 0.90  (prediction)  —  rel L2 = {rel_l2:.2e}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('u(t, x)')
axes[1].set_xlim([-1, 1]); axes[1].set_ylim([-1.2, 1.2])
axes[1].legend()

plt.suptitle(f"Burgers' Equation — Discrete-time PINN (q={Q}, dt={DT})", fontsize=12)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/burgers_discrete.png', dpi=150)
print("Figure saved to figures/burgers_discrete.png")
plt.show()
