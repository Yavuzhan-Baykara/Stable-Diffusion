import unittest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataloader import DatasetLoader

class TestDatasetShapes(unittest.TestCase):
    def check_dataset_shape(self, dataset_loader, expected_shape):
        for i, (images, _) in enumerate(dataset_loader.loader):
            if i >= 10:
                break
            self.assertEqual(images.shape[1:], expected_shape)

    def test_dataset_shapes(self):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_loader = DatasetLoader('mnist', transform=transform, download=True)
        self.check_dataset_shape(mnist_loader, (1, 28, 28))
        cifar10_loader = DatasetLoader('cifar10', transform=transform, download=True)
        self.check_dataset_shape(cifar10_loader, (3, 32, 32))

if __name__ == '__main__':
    unittest.main()