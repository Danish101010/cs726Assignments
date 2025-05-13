import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

# ==== Model Definition (Dummy placeholder) ====
# Replace this with your actual EnergyRegressor definition
class EnergyRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(EnergyRegressor, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==== Configs ====
FEAT_DIM = 10  # Set your feature dimension here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ==== Reproducibility ====
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== Energy & Gradient ====
def compute_energy_and_grad(model, x):
    x = x.clone().detach().requires_grad_(True)
    energy = model(x)
    energy.backward(torch.ones_like(energy))
    return energy.detach(), x.grad.detach()

# ==== Sampler Classes ====
class Algo1_Sampler:  # MALA
    def __init__(self, model, tau, device='cpu'):
        self.model = model.to(device).eval()
        self.tau = tau
        self.device = device

    def sample(self, init_x, num_samples=1000, burn_in=100):
        samples = []
        x = init_x.clone().to(self.device)
        for t in range(num_samples + burn_in):
            energy, grad = compute_energy_and_grad(self.model, x)
            noise = torch.randn_like(x)
            proposal = x - 0.5 * self.tau * grad + torch.sqrt(torch.tensor(self.tau)) * noise

            proposal_energy, proposal_grad = compute_energy_and_grad(self.model, proposal)

            log_q_x_given_xp = -((x - (proposal - 0.5 * self.tau * proposal_grad))**2).sum() / (4 * self.tau)
            log_q_xp_given_x = -((proposal - (x - 0.5 * self.tau * grad))**2).sum() / (4 * self.tau)

            log_alpha = (energy - proposal_energy + log_q_x_given_xp - log_q_xp_given_x).item()
            alpha = min(1, np.exp(log_alpha))

            if torch.rand(1).item() < alpha:
                x = proposal

            if t >= burn_in:
                samples.append(x.squeeze(0).detach().cpu().numpy())

        return np.array(samples)

class Algo2_Sampler:  # ULA
    def __init__(self, model, tau, device='cpu'):
        self.model = model.to(device).eval()
        self.tau = tau
        self.device = device

    def sample(self, init_x, num_samples=1000, burn_in=100):
        samples = []
        x = init_x.clone().to(self.device)
        for t in range(num_samples + burn_in):
            _, grad = compute_energy_and_grad(self.model, x)
            noise = torch.randn_like(x)
            x = x - 0.5 * self.tau * grad + torch.sqrt(torch.tensor(self.tau)) * noise

            if t >= burn_in:
                samples.append(x.squeeze(0).detach().cpu().numpy())

        return np.array(samples)

# ==== Load Model ====
model = EnergyRegressor(FEAT_DIM)
model.load_state_dict(torch.load("trained_model_weights.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# ==== Init Point ====
init_x = torch.randn(1, FEAT_DIM)

# ==== Run MALA ====
mala = Algo1_Sampler(model, tau=1e-3, device=DEVICE)
start = time.time()
mala_samples = mala.sample(init_x, num_samples=1000, burn_in=100)
mala_time = time.time() - start
print(f"[MALA] Sampling Time: {mala_time:.2f} seconds")

# ==== Run ULA ====
ula = Algo2_Sampler(model, tau=1e-3, device=DEVICE)
start = time.time()
ula_samples = ula.sample(init_x, num_samples=1000, burn_in=100)
ula_time = time.time() - start
print(f"[ULA] Sampling Time: {ula_time:.2f} seconds")

# ==== t-SNE Plotting ====
def plot_tsne(samples, title, filename):
    print(f"Running t-SNE for: {title}")
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(samples)
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.6)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot as: {filename}")

plot_tsne(mala_samples, "MALA Samples (Algo 1)", "mala_tsne.png")
plot_tsne(ula_samples, "ULA Samples (Algo 2)", "ula_tsne.png")
