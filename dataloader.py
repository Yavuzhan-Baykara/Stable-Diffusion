import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader  

class DatasetLoader:
    def __init__(self, dataset_name, transform, batch_size=64, shuffle=True, root="./data", download=True):
        self.dataset_name = dataset_name
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root = root
        self.download = download 
        self.dataset, self.class_names = self.load_dataset()
        self.loader = self.create_loader()
    
    def load_dataset(self):
        if self.dataset_name.lower() == "mnist":
            dataset = datasets.MNIST(root=self.root, train=True, transform=self.transform, download=self.download)
            class_names = [str(i) for i in range(10)]
        elif self.dataset_name.lower() == "cifar10": 
            dataset = datasets.CIFAR10(root=self.root, train=True, transform=self.transform, download=self.download)
            class_names = dataset.classes
        elif self.dataset_name.lower() == "cifar100": 
            dataset = datasets.CIFAR100(root=self.root, train=True, transform=self.transform, download=self.download)
            class_names = dataset.classes
        elif self.dataset_name.lower() == "custom":
            dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
            class_names = dataset.classes
        else:
            raise ValueError("Unsupported dataset") 
        return dataset, class_names
    
    def create_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
