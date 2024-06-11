import torch
import functools
from dataloader import DatasetLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from models import *
from tqdm.notebook import trange, tqdm
from forward import loss_fn
from sampler import Euler_Maruyama_sampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import wandb

wandb.init(project='Mnist-Stable-Diffusion', entity='yavzan-baggins') 

ro = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

score_model = torch.nn.DataParallel(UNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)

num_steps = 500
n_epochs = 100
batch_size = 1024
lr = 1e-3
dataset = DatasetLoader(dataset_name="mnist", transform=transforms.ToTensor(), batch_size=batch_size, shuffle=True)
data_loader = dataset.create_loader()

optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

lr_schedule = {75: 1e-3, 100: 1e-4}

def get_learning_rate(epoch, lr_schedule, default_lr):
    for e in sorted(lr_schedule.keys()):
        if epoch < e:
            return lr_schedule[e]
    return default_lr

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    current_lr = get_learning_rate(epoch, lr_schedule, scheduler.get_last_lr()[0])
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for x, y in tqdm(data_loader):
        x = x.to(DEVICE)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    scheduler.step()
    lr_current = current_lr
    
    avg_loss_epoch = avg_loss / num_items
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss_epoch, lr_current))
    tqdm_epoch.set_description(f'Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss_epoch:.5f}, LR: {lr_current}')
    
    wandb.log({'epoch': epoch + 1, 'average_loss': avg_loss_epoch, 'learning_rate': lr_current,
               'ro': ro, 'tqdm_value': str(tqdm_epoch), 'num_items': num_items})
    
    if epoch == n_epochs - 1:
        samples = Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, batch_size=64, num_steps=num_steps, device=DEVICE)
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(64)))

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        sample_path = f'saves/Unet_res-sample_epoch_{epoch + 1}.png'
        plt.savefig(sample_path)
        plt.close()

        wandb.log({"sample_images": [wandb.Image(sample_grid, caption=f"Epoch {epoch + 1}")]})

torch.save(score_model.state_dict(), 'models/Unet_res-ckpt.pth')

wandb.finish()
