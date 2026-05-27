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
        
        
        
    def forward(self, X, mask):
        
        # --- MULTI-HEAD ATTENTION BLOCK ---
        # 1. Normalize a COPY of the input for the heavy machinery
        norm_x = self.norm1(X)
        
        # 2. Run the normalized data through the attention heads
        attn_out, w = self.mha(norm_x, mask)
        
        # 3. TRUE RESIDUAL: Add the attention output to the PURE, un-normalized X
        out1 = attn_out + X
        
        
        # --- FEED-FORWARD BLOCK ---
        # 4. Normalize a COPY of the new baseline for the FFN
        norm_out1 = self.norm2(out1)
        
        # 5. Run the normalized data through the reasoning engine
        ffn_out = self.ffn(norm_out1)
        
        # 6. TRUE RESIDUAL: Add the FFN output back to the PURE out1 baseline
        out2 = ffn_out + out1
        
        return out2, w
            
        