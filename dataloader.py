import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader  

class DatasetLoader:
    def __init__(self, dataset_name, transform, batch_size=64, shuffle=True, root="./data", download=False):
        self.dataset_name = dataset_name
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root = root
        self.download = download 
        self.dataset = self.load_dataset()
        self.loader = self.create_loader()
    
    def load_dataset(self):
        if self.dataset_name.lower() == "mnist":
            return datasets.MNIST(root=self.root, train=True, transform=self.transform, download=self.download)
        elif self.dataset_name.lower() == "cifar10": 
            return datasets.CIFAR10(root=self.root, train=True, transform=self.transform, download=self.download)
        else:
            raise ValueError("Unsupported dataset") 
    
    def create_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

class ImageVisualizer:
    @staticmethod
    def visualize_batch(images, n_rows=4, n_cols=16, cmap='gray'):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), sharex=True, sharey=True)
        images = images.numpy()

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                img = images[i * n_cols + j].squeeze()

                if img.shape[0] == 3:
                    img = img.transpose((1, 2, 0))

                ax.imshow(img, cmap=cmap)
                ax.axis('off')

        return fig, axes 

    @staticmethod
    def show_images(fig, axes):
        plt.show()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_loader = DatasetLoader('mnist', transform=transform, download=True)

images, labels = next(iter(dataset_loader.loader))
print(labels)
ImageVisualizer.visualize_batch(images)