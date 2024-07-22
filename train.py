import os
import torch
from tqdm.notebook import trange, tqdm
from torchvision import transforms
from loss import loss_fn_cond
import wandb

# Sınıfların resimlerini kaydedecek klasörleri oluştur
def create_class_folders(base_path, class_names):
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)

# Resimleri uygun klasörlere kaydet
def save_images_to_folders(images, labels, base_path, class_names):
    for i in range(len(labels)):
        class_name = class_names[labels[i].item()]
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        image_id = len(os.listdir(class_path))
        image_path = os.path.join(class_path, f'{class_name}_{image_id}.png')
        
        pil_image = transforms.ToPILImage()(images[i].squeeze().cpu())
        pil_image.save(image_path)

def train_model(score_model, data_loader, optimizer, scheduler, sampler, n_epochs, marginal_prob_std_fn, diffusion_coeff_fn, DEVICE, wandb_enabled, digit, num_steps):
    base_path = 'saves'
    create_class_folders(base_path, data_loader.dataset.classes)

    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        
        for x, y in tqdm(data_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)  # Ensure labels are moved to the device
            optimizer.zero_grad()
            loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            
            # CUDA cache'i temizle
            torch.cuda.empty_cache()
        
        scheduler.step()
        lr_current = optimizer.param_groups[0]['lr']
        
        avg_loss_epoch = avg_loss / num_items
        print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss_epoch, lr_current))
        tqdm_epoch.set_description(f'Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss_epoch:.5f}, LR: {lr_current}')
        
        if wandb_enabled:
            wandb.log({'epoch': epoch + 1, 'average_loss': avg_loss_epoch, 'learning_rate': lr_current,
                       'num_items': num_items})

        sample_batch_size = 1  # Only generate one sample per epoch
        for class_idx in range(len(data_loader.dataset.classes)):
            samples = sampler(score_model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                sample_batch_size,
                num_steps=num_steps,
                device=DEVICE,
                y=torch.tensor([class_idx]).to(DEVICE))  # Generate a sample for each class
            
            # Resimleri kaydet
            save_images_to_folders(samples, torch.tensor([class_idx]).to(DEVICE), base_path, data_loader.dataset.classes)

        model_path = f'models/cifar_epoch_{epoch + 1}.pth'
        torch.save(score_model.state_dict(), model_path)
        if wandb_enabled:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    torch.save(score_model.state_dict(), 'models/Cifar.pth')
