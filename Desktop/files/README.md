# Physics-Informed Neural Networks — Replication & Extension

> Replication of **Raissi, Perdikaris & Karniadakis (2017)**  
> *"Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear PDEs"*  
> arXiv: [1711.10561](https://arxiv.org/abs/1711.10561) | Original code: [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)



## Replication Targets

| Experiment | PDE | Method | Paper Error | Our Error |
| 1 | Burgers' | Continuous | 6.7e-4 | TBD |
| 2 | Schrödinger | Continuous | 1.97e-3 | TBD |
| 3 | Burgers' | Discrete (RK-500) | 8.2e-4 | TBD |
| 4 | Allen-Cahn | Discrete (RK-100) | 6.99e-3 | TBD |


## Extension: Noise Robustness Study

**Question:** How does PINN performance degrade as Gaussian noise is added to training data?

**Motivation:** The paper demonstrates PINNs work with small data (Nu=100), but real measurements are noisy. The physics loss (MSE_f) acts as a regularizer — we hypothesize it provides noise robustness that a standard NN lacks.

**Protocol:**
- Fix architecture: 9 layers, 20 neurons, tanh, Nu=100, Nf=10,000
- Vary noise level σ ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5}
- Compare: PINN (Nf=10,000) vs. baseline NN (Nf=0)
- Metric: relative L2 error on clean test set
- Repeat 5 seeds → report mean ± std

**Expected result:** PINN degrades more gracefully than the baseline NN due to physics regularization.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Alphaman1321/pinn-replication
cd pinn-replication
pip install -r requirements.txt

# 2. Replicate Burgers' continuous (matches paper Fig. 1)
python experiments/run_burgers_continuous.py

# 3. Run extension (noise robustness)
python experiments/extension_noise_study.py

# 4. Interactive walkthrough
jupyter notebook notebooks/01_burgers_walkthrough.ipynb
```

---

## Requirements

- Python 3.9+
- PyTorch 2.x (we re-implement in PyTorch; original used TensorFlow 1.x)
- See `requirements.txt` for full list

**Note on framework choice:** The original paper uses TensorFlow 1.x, which is now legacy. We re-implement in **PyTorch** for readability and modern autograd support. The math is identical; `torch.autograd.grad` replaces `tf.gradients`.

---

## Key Implementation Notes

### Automatic Differentiation
The core idea is that physics residuals are computed via autograd — no finite differences needed:
```python
u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
```

### Loss Function
```
MSE_total = MSE_u  (data fit at boundary/initial points)
          + MSE_f  (physics residual at collocation points)
```
Setting `Nf=0` recovers a standard neural network — used as our baseline.

### Optimizer
We use **L-BFGS** (full-batch) as in the paper via `torch.optim.LBFGS`, with a fallback to Adam for larger datasets.

---

## References

```bibtex
@article{raissi2017physicsinformed,
  title   = {Physics Informed Deep Learning (Part I): Data-driven Solutions
             of Nonlinear Partial Differential Equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal = {arXiv preprint arXiv:1711.10561},
  year    = {2017}
}
```
