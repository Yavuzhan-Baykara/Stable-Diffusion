import torch
from models import *
from sampler import Euler_Maruyama_sampler, PC_sampler, Langevin_Dynamics_sampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import functools
import matplotlib.pyplot as plt

digit = 2
ro = 30.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)
ckpt = torch.load('models/Cifar-fine-tuning.pth', map_location=DEVICE)
score_model.load_state_dict(ckpt)


sample_batch_size = 64 
num_steps = 250
sampler = Euler_Maruyama_sampler
samples = sampler(score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        num_steps=num_steps,
        device=DEVICE,
        y=digit*torch.ones(sample_batch_size, dtype=torch.long))

samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()