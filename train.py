import os
import random
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
wandb_enabled = 1  # 0 ise WandB veri aktarımı yapma, 1 ise yap
if wandb_enabled:
    wandb.init(project='Mnist-Stable-Diffusion', entity='yavzan-baggins')

# Hyperparametreler ve ayarlar
ro = 30.0
sampler = Euler_Maruyama_sampler
digit = 9
num_steps = 250
n_epochs = 20
batch_size = 128
lr = 1e-2
lr_schedule = {5: 1e-2, 10: 1e-3, 15: 1e-4, 20: 1e-5}

marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

score_model = torch.nn.DataParallel(UNet_Transformer(marginal_prob_std=marginal_prob_std_fn))
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

# Sınıfların resimlerini kaydedecek klasörleri oluştur
def create_class_folders(base_path, class_names):
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)

# Klasörleri oluştur
base_path = 'saves'
create_class_folders(base_path, dataset.class_names)

# Resimleri uygun klasörlere kaydet
def save_images_to_folders(images, labels, base_path):
    class_names = dataset.class_names
    for i in range(len(labels)):
        class_name = class_names[labels[i].item()]
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        image_id = len(os.listdir(class_path))
        image_path = os.path.join(class_path, f'{class_name}_{image_id}.png')
        
        pil_image = transforms.ToPILImage()(images[i].squeeze().cpu())
        pil_image.save(image_path)

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
    
    if wandb_enabled:
        wandb.log({'epoch': epoch + 1, 'average_loss': avg_loss_epoch, 'learning_rate': lr_current,
                   'ro': ro, 'tqdm_value': str(tqdm_epoch), 'num_items': num_items})

    sample_batch_size = 128
    samples = sampler(score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        num_steps=num_steps,
        device=DEVICE,
        y=digit*torch.ones(sample_batch_size, dtype=torch.long))
    
    model_path = f'models/cifar_epoch_{epoch + 1}.pth'
    torch.save(score_model.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    # Resimleri kaydet
    save_images_to_folders(samples, y, base_path)
    
torch.save(score_model.state_dict(), 'models/Cifar.pth')

if wandb_enabled:
    wandb.finish()
