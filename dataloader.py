import torch
import torchvision
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
        self.download = download  # There was a typo here (downlaod -> download)
        self.dataset = self.load_dataset()
        self.loader = self.create_loader()
    
    def load_dataset(self):
        if self.dataset_name.lower() == "mnist":
            return datasets.MNIST(root=self.root, train=True, transform=self.transform, download=self.download)
        elif self.dataset_name.lower() == "cifar10":  # Lowercase for consistency
            return datasets.CIFAR10(root=self.root, train=True, transform=self.transform, download=self.download)
        else:
            raise ValueError("Unsupported dataset")  # Fixed typo in "Unspupported"
    
    def create_loader(self):
        # This should return a DataLoader instance, not a DatasetLoader instance.
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

class ImageVisualizer:
    @staticmethod
    def visualize_batch(images, n_rows=4, n_cols=16, cmap='gray'):
        """Visualizes a batch of images in a grid-like layout.

        Args:
            images: A NumPy array of images, with shape (batch_size, height, width)
                or (batch_size, height, width, channels).
            n_rows (int, optional): Number of rows in the grid. Defaults to 4.
            n_cols (int, optional): Number of columns in the grid. Defaults to 16.
            cmap (str, optional): Colormap to use for visualization. Defaults to 'gray'.

        Returns:
            None
        """

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), sharex=True, sharey=True)
        images = images.numpy()

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                img = images[i * n_cols + j].squeeze()

                if img.shape[0] == 3:  # If the image has 3 channels, transpose
                    img = img.transpose((1, 2, 0))  # From (C, H, W) to (H, W, C)

                ax.imshow(img, cmap=cmap)
                ax.axis('off')

        return fig, axes 

    @staticmethod
    def show_images(fig, axes):
        plt.show()


#Usage
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_loader = DatasetLoader('mnist', transform=transform, download=True)

images, labels = next(iter(dataset_loader.loader))
print(labels)
# Visualization
ImageVisualizer.visualize_batch(images)