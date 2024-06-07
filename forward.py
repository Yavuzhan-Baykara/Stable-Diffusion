import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

DEVICE = "cuda"


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. -2 * eps) + eps
    std = marginal_prob_std(random_t)
    z = torch.rand_like(x)
    per_x = x + z * std[:, None, None, None]
    score = model(per_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2), dim=(1, 2, 3))

    return loss