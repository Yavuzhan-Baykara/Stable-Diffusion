import torch
import functools
from models import *
from torchvision.utils import make_grid
import numpy as np
from sampler import Euler_Maruyama_sampler
import matplotlib.pyplot as plt

ro = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)

score_model = torch.nn.DataParallel(UNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)

ckpt = torch.load('models/ckpt_epoch_1881.pth', map_location=DEVICE)
score_model.load_state_dict(ckpt)
sample_batch_size = 64
num_steps = 500
sampler = Euler_Maruyama_sampler

samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=DEVICE,
                  y=None)

samples = samples.clamp(0.0, 1.0)

sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# Plot the sample grid
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()