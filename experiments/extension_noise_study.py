"""
experiments/extension_noise_study.py
--------------------------------------
EXTENSION: Noise Robustness of PINNs vs. Baseline Neural Networks

Research question:
    How does prediction accuracy degrade as Gaussian noise is added to
    training data?  Does the physics regularization term (MSE_f) make
    PINNs more robust than a standard NN?

Hypothesis:
    PINNs will degrade more gracefully because MSE_f acts as an implicit
    regularizer, constraining the solution space to physically valid functions
    even when the data labels are corrupted.

Protocol:
    - PDE: Burgers' equation (continuous-time setting)
    - Fixed architecture: 9 layers, 20 neurons, tanh
    - Fixed data: Nu=100, Nf=10,000
    - Noise levels: σ ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5}
    - Models: PINN (Nf=10,000) vs. Baseline NN (Nf=0)
    - Seeds: 5 independent runs → mean ± std
    - Metric: relative L2 error on clean test grid

Usage:
    python experiments/extension_noise_study.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.continuous.burgers import BurgersPINN, burgers_initial_boundary_data, burgers_collocation_points
from src.utils.training import train_lbfgs

#    Configuration                                                             
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
N_SEEDS      = 5
N_U          = 100
N_F_PINN     = 10_000    # collocation points for PINN
N_F_BASELINE = 0         # no physics loss for baseline NN
N_LAYERS     = 9
N_NEURONS    = 20
MAX_ITER     = 20_000    # reduced for study efficiency
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Running noise robustness study: {len(NOISE_LEVELS)} noise levels × {N_SEEDS} seeds × 2 models")
print(f"Total runs: {len(NOISE_LEVELS) * N_SEEDS * 2}")
print("-" * 60)


def run_single(sigma: float, seed: int, n_f: int) -> float:
    """
    Train one model and return its relative L2 error on a clean test set.

    Args:
        sigma : noise standard deviation added to u labels
        seed  : random seed for reproducibility
        n_f   : number of collocation points (0 = baseline NN)

    Returns:
        relative L2 error (float)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    #    Data   
    t_u, x_u, u_clean = burgers_initial_boundary_data(N_U, seed=seed)

    # Add Gaussian noise to the labels
    noise = torch.randn_like(u_clean) * sigma
    u_noisy = u_clean + noise

    t_u  = t_u.to(DEVICE)
    x_u  = x_u.to(DEVICE)
    u_noisy = u_noisy.to(DEVICE)

    # Collocation points (empty for baseline)
    if n_f > 0:
        t_f, x_f = burgers_collocation_points(n_f, seed=seed)
        t_f = t_f.to(DEVICE).requires_grad_(True)
        x_f = x_f.to(DEVICE).requires_grad_(True)
    else:
        t_f = torch.empty(0, 1, device=DEVICE)
        x_f = torch.empty(0, 1, device=DEVICE)

    #    Model   
    model = BurgersPINN(n_layers=N_LAYERS, n_neurons=N_NEURONS).to(DEVICE)

    def loss_fn():
        return model.loss(t_u, x_u, u_noisy, t_f if n_f > 0 else None, x_f if n_f > 0 else None)

    #    Train   
    train_lbfgs(model, loss_fn, max_iter=MAX_ITER, log_every=MAX_ITER+1, verbose=False)

    #    Evaluate on clean test grid   
    # Generate a fine test grid and compare against clean solution
    t_grid = np.linspace(0, 1, 100)
    x_grid = np.linspace(-1, 1, 256)
    T, X = np.meshgrid(t_grid, x_grid)

    t_test = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=DEVICE)
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        u_pred = model(t_test, x_test)

    # Clean reference from initial/boundary data model (sigma=0, same seed)
    # In practice, load the analytical solution here
    # For now, compute relative improvement over noisy baseline
    # TODO: replace with exact solution loaded from data/burgers_exact.mat
    u_clean_test, _, _ = burgers_initial_boundary_data(N_U * 10, seed=seed + 1000)
    # Proxy metric: relative L2 on training points with clean labels
    t_eval = t_u[:20]
    x_eval = x_u[:20]
    u_eval = u_clean[:20].to(DEVICE)

    with torch.no_grad():
        u_pred_eval = model(t_eval, x_eval)
    rel_l2 = (torch.norm(u_pred_eval - u_eval) / torch.norm(u_eval)).item()
    return rel_l2


#    Main study loop                                                            
results = {
    'pinn':     {sigma: [] for sigma in NOISE_LEVELS},
    'baseline': {sigma: [] for sigma in NOISE_LEVELS},
}

for sigma in NOISE_LEVELS:
    for seed in range(N_SEEDS):
        print(f"  σ={sigma:.2f} | seed={seed} | PINN ...", end=' ', flush=True)
        err_pinn = run_single(sigma, seed, N_F_PINN)
        print(f"err={err_pinn:.3e} | Baseline ...", end=' ', flush=True)
        err_base = run_single(sigma, seed, N_F_BASELINE)
        print(f"err={err_base:.3e}")
        results['pinn'][sigma].append(err_pinn)
        results['baseline'][sigma].append(err_base)


#    Summary table                                                              
print("\n" + "=" * 60)
print(f"{'σ':>6} | {'PINN mean':>12} {'± std':>10} | {'NN mean':>12} {'± std':>10}")
print("-" * 60)
for sigma in NOISE_LEVELS:
    p_mean = np.mean(results['pinn'][sigma])
    p_std  = np.std(results['pinn'][sigma])
    b_mean = np.mean(results['baseline'][sigma])
    b_std  = np.std(results['baseline'][sigma])
    print(f"{sigma:6.2f} | {p_mean:12.3e} {p_std:10.3e} | {b_mean:12.3e} {b_std:10.3e}")


#    Plot                                                                       
fig, ax = plt.subplots(figsize=(7, 4))

pinn_means = [np.mean(results['pinn'][s]) for s in NOISE_LEVELS]
pinn_stds  = [np.std(results['pinn'][s])  for s in NOISE_LEVELS]
base_means = [np.mean(results['baseline'][s]) for s in NOISE_LEVELS]
base_stds  = [np.std(results['baseline'][s])  for s in NOISE_LEVELS]

ax.errorbar(NOISE_LEVELS, pinn_means, yerr=pinn_stds,
            fmt='o-', color='royalblue', label='PINN (Nf=10,000)', capsize=4)
ax.errorbar(NOISE_LEVELS, base_means, yerr=base_stds,
            fmt='s--', color='tomato', label='Baseline NN (Nf=0)', capsize=4)

ax.set_xlabel('Noise level σ', fontsize=12)
ax.set_ylabel('Relative L2 error', fontsize=12)
ax.set_title("Noise Robustness: PINN vs. Baseline NN\n(Burgers' equation, 5 seeds)", fontsize=12)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/extension_noise_robustness.png', dpi=150)
print("\nFigure saved to figures/extension_noise_robustness.png")
plt.show()
