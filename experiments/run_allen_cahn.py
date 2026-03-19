"""
experiments/run_allen_cahn.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.discrete.allen_cahn import AllenCahnPINN
from src.utils.training import train_lbfgs, train_adam

Q         = 50
DT        = 0.8
NN        = 200
SEED      = 42
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Config: Nn={NN}, q={Q}, dt={DT}")
print("-" * 60)

rng = np.random.default_rng(SEED)
x_n_np = rng.uniform(-1, 1, (NN, 1)).astype(np.float32)
u_n_np = (x_n_np**2 * np.cos(np.pi * x_n_np)).astype(np.float32)

x_n = torch.tensor(x_n_np, device=DEVICE, requires_grad=True)
u_n = torch.tensor(u_n_np, device=DEVICE)

print("Building Allen-Cahn PINN (4 layers x 100 neurons, q=50)...")
# Use smaller neurons (100 instead of 200) so Adam warmup is feasible on CPU
model = AllenCahnPINN(q=Q, dt=DT, n_layers=4, n_neurons=100).to(DEVICE)
print(f"Network parameters: {sum(p.numel() for p in model.parameters())}")

def loss_fn():
    total, sse_n, sse_b = model.loss(x_n, u_n, [])
    return total, sse_n, sse_b

print("\nAdam warmup (3000 iters) to escape bad initialisation...")
train_adam(model, loss_fn, n_iter=3000, lr=1e-3, log_every=500)

print("\nL-BFGS refinement...")
log = train_lbfgs(model, loss_fn, max_iter=20_000, log_every=1000)

x_test_np = np.linspace(-1, 1, 512).astype(np.float32)[:, None]
x_test    = torch.tensor(x_test_np, device=DEVICE)

model.eval()
with torch.no_grad():
    U_out  = model(x_test)
    u_next = U_out[:, Q].cpu().numpy().flatten()

# Allen-Cahn solution should stay in [-1, 1] — clip anything outside that
u_next_clipped = np.clip(u_next, -1.2, 1.2)

print(f"\nFinal loss: {log.total_loss[-1]:.3e}")
print(f"Prediction range at t=0.9: [{u_next.min():.3f}, {u_next.max():.3f}]")
if abs(u_next.max()) > 5:
    print("WARNING: prediction still out of range — model may need more iterations")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].scatter(x_n_np.flatten(), u_n_np.flatten(), s=8, color='red', label='Data (t=0.1)', zorder=3)
axes[0].set_title('t = 0.10  (training data)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('u(t, x)')
axes[0].set_xlim([-1, 1]); axes[0].legend()

axes[1].plot(x_test_np.flatten(), u_next_clipped, 'r--', label='Prediction (t=0.9)', linewidth=2)
axes[1].set_title('t = 0.90  (prediction)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('u(t, x)')
axes[1].set_xlim([-1, 1]); axes[1].set_ylim([-1.3, 1.3])
axes[1].legend()

plt.suptitle(f"Allen-Cahn Equation — Discrete-time PINN (q={Q}, dt={DT})", fontsize=12)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/allen_cahn_discrete.png', dpi=150)
print("Figure saved to figures/allen_cahn_discrete.png")
plt.show()
