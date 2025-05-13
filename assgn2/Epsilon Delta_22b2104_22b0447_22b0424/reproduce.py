import torch
import numpy as np
from ddpm import DDPM, NoiseScheduler  # Ensure these are correctly imported
from ddpm import sample
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
model = DDPM(n_dim=64, n_steps=200)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Load prior samples
prior_samples = np.load("data/albatross_prior_samples.npy")
prior_samples = torch.tensor(prior_samples, dtype=torch.float32, device=device)

# Initialize noise scheduler
noise_scheduler = NoiseScheduler(num_timesteps=200,beta_start=0.0001,beta_end=0.04)

# Perform sampling
samples = sample(model, prior_samples.shape[0], noise_scheduler)

# Save generated samples
samples = samples.cpu().numpy()
np.save("albatross_samples_reproduce.npy", samples)

print("Generated samples saved to albatross_samples_reproduce.npy")
