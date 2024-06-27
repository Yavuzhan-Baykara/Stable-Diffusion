import torch
import functools
from dataloader import DatasetLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from models import *
from tqdm.notebook import trange, tqdm
from loss import loss_fn, loss_fn_cond
from sampler import Euler_Maruyama_sampler, PC_sampler, Langevin_Dynamics_sampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import wandb

# WandB ayarları
wandb.init(project='Mnist-Stable-Diffusion', entity='yavzan-baggins')

# Hyperparametreler ve ayarlar
ro = 30.0
sampler = Euler_Maruyama_sampler
digit = 9
num_steps = 250
n_epochs = 30
batch_size = 128  # Batch boyutunu küçülttüm
lr = 1e-4
lr_schedule = {50: 1e-4, 100: 1e-5}

marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)
dataset = DatasetLoader(dataset_name="cifar10", transform=transforms.ToTensor(), batch_size=batch_size, shuffle=True)
data_loader = dataset.create_loader()
optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

def get_learning_rate(epoch, lr_schedule, default_lr):
    for e in sorted(lr_schedule.keys()):
        if epoch < e:
            return lr_schedule[e]
    return default_lr

# Eğitim döngüsü
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    current_lr = get_learning_rate(epoch, lr_schedule, scheduler.get_last_lr()[0])
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for x, y in tqdm(data_loader):
        x = x.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        
        # CUDA cache'i temizle
        torch.cuda.empty_cache()
    
    scheduler.step()
    lr_current = current_lr
    
    avg_loss_epoch = avg_loss / num_items
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss_epoch, lr_current))
    tqdm_epoch.set_description(f'Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss_epoch:.5f}, LR: {lr_current}')
    
    wandb.log({'epoch': epoch + 1, 'average_loss': avg_loss_epoch, 'learning_rate': lr_current,
               'ro': ro, 'tqdm_value': str(tqdm_epoch), 'num_items': num_items})
    
torch.save(score_model.state_dict(), 'models/Cifar2.pth')

wandb.finish()
