import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CrossAttention
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(TransformerBlock, self ).__init__()
        
        self.self_attention = CrossAttention(hidden_dim, hidden_dim)
        self.cross_attention = CrossAttention(hidden_dim, hidden_dim, context_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x, context=None):
        x = self.self_attention(self.norm1(x)) + x
        x = self.cross_attention(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        
        return x
    

class SpatialTransformer(nn.Module):
    """
    x: Input tensor with shape [batch, channels, height, width]
    """
    def __init__(self, hidden_dim, context_dim):
        super().__init__(SpatialTransformer, self)
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w) 
        return x + x_in



