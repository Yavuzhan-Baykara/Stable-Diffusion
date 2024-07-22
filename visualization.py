import torch
from models import *
from sampler import Euler_Maruyama_sampler, PC_sampler, Langevin_Dynamics_sampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import functools
import matplotlib.pyplot as plt
from PIL import Image

# Function to generate and save samples for a given digit
def generate_samples_for_digit(digit, score_model, sampler, marginal_prob_std_fn, diffusion_coeff_fn, sample_batch_size, num_steps, device):
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      num_steps=num_steps,
                      device=device,
                      y=digit*torch.ones(sample_batch_size, dtype=torch.long))
    
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    sample_grid = sample_grid.permute(1, 2, 0).cpu().numpy()

    return sample_grid

# Set parameters
digits = list(range(10))
ro = 30.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
score_model = torch.nn.DataParallel(UNet_Transformer(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)
ckpt = torch.load('models/Unet_res-ckpt.pth', map_location=DEVICE)
score_model.load_state_dict(ckpt)

sample_batch_size = 64
num_steps = 250
sampler = Euler_Maruyama_sampler

# Generate samples for each digit and store them as frames
frames = []
for digit in digits:
    samples_grid = generate_samples_for_digit(digit, score_model, sampler, marginal_prob_std_fn, diffusion_coeff_fn, sample_batch_size, num_steps, DEVICE)
    frames.append(samples_grid)

# Create GIF
frame_images = [Image.fromarray((frame * 255).astype(np.uint8)) for frame in frames]
frame_images[0].save('digit_samples.gif', save_all=True, append_images=frame_images[1:], duration=200, loop=0)
