import argparse
import functools
import torch
from models import UNet_Transformer
from sampler import Euler_Maruyama_sampler, PC_sampler, Langevin_Dynamics_sampler
from dataloader import DatasetLoader
import torchvision.transforms as transforms
from train import train_model

# Hyperparametreler ve ayarlar
def marginal_prob_std(t, ro):
    return torch.sqrt((1 - torch.exp(-2 * ro * t)) / (2 * ro))

def diffusion_coeff(t, ro):
    return torch.sqrt(2 * (1 - torch.exp(-2 * ro * t)))

# Dataset'e göre x_shape değerini belirleyen yardımcı fonksiyon
def get_x_shape(dataset_name):
    if dataset_name == 'mnist':
        return (1, 28, 28)
    elif dataset_name in ['cifar10', 'cifar100']:
        return (3, 32, 32)
    else:
        raise ValueError("Unsupported dataset")

# Argümanları işlemek için argparse kullanımı
parser = argparse.ArgumentParser(description="Train UNet_Transformer with different datasets and samplers")
parser.add_argument('dataset', type=str, choices=['mnist', 'cifar10', 'cifar100'], help="Name of the dataset")
parser.add_argument('sampler', type=str, choices=['Euler', 'pc', 'Langevin'], help="Name of the sampler")
parser.add_argument('epoch', type=int, help="Number of epochs")
parser.add_argument('batch_size', type=int, help="Batch size")
args = parser.parse_args()

wandb_enabled = 0  # 0 ise WandB veri aktarımı yapma, 1 ise yap
if wandb_enabled:
    import wandb
    wandb.init(project='Mnist-Stable-Diffusion', entity='yavzan-baggins')

dataset_name = args.dataset
digit = 9
num_steps = 250
n_epochs = args.epoch
batch_size = args.batch_size
x_shape = get_x_shape(dataset_name)

lr = 10e-4  # Öğrenme hızını sabit olarak ayarla

if args.sampler == 'Euler':
    sampler = functools.partial(Euler_Maruyama_sampler, x_shape=x_shape)
elif args.sampler == 'pc':
    sampler = functools.partial(PC_sampler, x_shape=x_shape)
elif args.sampler == 'Langevin':
    sampler = functools.partial(Langevin_Dynamics_sampler, x_shape=x_shape)

ro = 30.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()  # CUDA belleğini temizle

score_model = torch.nn.DataParallel(UNet_Transformer(marginal_prob_std=marginal_prob_std_fn, dataset_name=dataset_name))
score_model = score_model.to(device=DEVICE)
dataset = DatasetLoader(dataset_name=dataset_name, transform=transforms.ToTensor(), batch_size=batch_size, shuffle=True)
data_loader = dataset.create_loader()
optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

train_model(score_model, data_loader, optimizer, scheduler, sampler, n_epochs, marginal_prob_std_fn, diffusion_coeff_fn, DEVICE, wandb_enabled, digit, num_steps)

if wandb_enabled:
    wandb.finish()
