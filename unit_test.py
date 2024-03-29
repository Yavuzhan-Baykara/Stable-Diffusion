import unittest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataloader import DatasetLoader
from sample.sample import SampleDiffusionProcessGenerator
from models import GaussianFourierProjection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestDatasetShapes(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.n_steps = 100
        self.t0 = 0
        self.dt = 0.1
        self.sample_diffusion = SampleDiffusionProcessGenerator(self.n_steps, self.t0, self.dt)
        self.x0 = 0
        self.noise_strength_fn = self.sample_diffusion.noise_strength_constant


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
    

    def test_forward_diffusion_1d_length(self):
        """Test the output length of the forward_diffusion_1d method."""
        x, t = self.sample_diffusion.forward_diffusion_1d(self.x0, self.noise_strength_fn)
        self.assertEqual(len(x), self.n_steps + 1)
        self.assertEqual(len(t), self.n_steps + 1)

    def test_noise_strength_constant(self):
        """Test that the noise_strength_constant method returns a constant value."""
        for t in np.linspace(0, 1, 10):
            self.assertEqual(self.sample_diffusion.noise_strength_constant(t), 1)

    def test_score_simple_values(self):
        """Test that the score_simple method returns expected values."""
        x, x0, noise_strength, t = 2, 0, 1, 1
        expected_score = -2
        score = self.sample_diffusion.score_simple(x, x0, noise_strength, t)
        self.assertEqual(score, expected_score)

        # Check for ZeroDivisionError when time is zero
        with self.assertRaises(ZeroDivisionError):
            self.sample_diffusion.score_simple(x, x0, noise_strength, 0)

        # Check for negative time
        score_neg_time = self.sample_diffusion.score_simple(x, x0, noise_strength, -1)
        self.assertGreater(score_neg_time, 0, "Score for negative time should be positive.")
    
    def test_forward_output_shape(self):
        """
        Tests that the forward method produces an output tensor with the expected shape.
        """
        embed_dim = 10
        time_steps = 5
        x = torch.randn(time_steps)
        model = GaussianFourierProjection(embed_dim=embed_dim)
        output = model(x)
        self.assertEqual(output.shape, (time_steps, embed_dim))

    def test_forward_sin_cos_values(self):
        """
        Tests that the forward method correctly calculates and combines sin and cos values.
        """
        x = torch.tensor([0, 1, 2])
        model = GaussianFourierProjection(embed_dim=4)
        output = model(x)

        expected_sin = torch.sin(x[:, None] * model.W[None, :] * 2 * np.pi)
        expected_cos = torch.cos(x[:, None] * model.W[None, :] * 2 * np.pi)

        self.assertTrue(torch.allclose(output[:, :2], expected_sin))
        self.assertTrue(torch.allclose(output[:, 2:4], expected_cos))

        
if __name__ == '__main__':
    unittest.main()