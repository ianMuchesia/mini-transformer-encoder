import torch
import torch.nn as nn
import math
from src.attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads ,d_ff=2048):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" 

        self.mha = MultiHeadAttention(d_model,num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        
        
        
    def forward(self,X,mask):
        
        out1,w = self.mha(X,mask)
        
        out2 = out1 + X
        
        out3 = self.norm1(out2)

        out4 = self.ffn(out3)
        
        out5 =  out4 + out3
        
        out6 = self.norm2(out5)
        
        return out6,w        
        
        