import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()

        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):    
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    