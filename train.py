import torch
import functools
from dataloader import DatasetLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from models import *
from tqdm.notebook import trange, tqdm
from forward import loss_fn
from sampler import Euler_Maruyama_sampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
ro = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)


score_model = torch.nn.DataParallel(UNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)


num_steps = 500
n_epochs = 200
batch_size = 2048
dataset = DatasetLoader(dataset_name="mnist", transform=transforms.ToTensor(), batch_size=batch_size, shuffle=True)
data_loader = dataset.create_loader()
lr_schedule = {100: 5e-3, 200: 5e-4}
optimizer = Adam(score_model.parameters(), lr=5e-4)

def get_learning_rate(epoch, lr_schedule):
    for e in sorted(lr_schedule.keys()):
        if epoch < e:
            return lr_schedule[e]
    return lr_schedule[max(lr_schedule.keys())]

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    current_lr = get_learning_rate(epoch, lr_schedule)
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
    
    tqdm_epoch.set_description(f'Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss / num_items:.5f}, LR: {current_lr}')
    torch.save(score_model.state_dict(), 'models/ckpt.pth')
    
    # Sample and save images after each epoch
    samples = Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, batch_size=64, num_steps=num_steps, device=DEVICE)
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(64)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(f'saves/sample_epoch_{epoch + 1}.png')
    plt.close()
    torch.save(score_model.state_dict(), 'models/ckpt2.pth')

# 18.00