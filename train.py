import torch
import functools
from dataloader import DatasetLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from models import *
from tqdm.notebook import trange, tqdm
from forward import loss_fn

ro = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, ro=ro)
diffusion_coeff_fn = functools.partial(diffusion_coeff, ro=ro)


score_model = torch.nn.DataParallel(Unet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device=DEVICE)


n_epochs = 50
batch_size = 2048
lr = 5e-4
dataset = DatasetLoader(dataset_name="mnist", transform=transforms.ToTensor(), batch_size=batch_size, shuffle=True)
data_loader = dataset.create_loader()
optimizer = Adam(score_model.parameters(), lr=lr)

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    # Iterate through mini-batches in the data loader
    for x, y in tqdm(data_loader):
        x = x.to(DEVICE)
        # Calculate the loss and perform backpropagation
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    # Print the averaged training loss for the current epoch
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Save the model checkpoint after each epoch of training
    torch.save(score_model.state_dict(), 'ckpt.pth')