import torch
import torch.nn as nn
import math

class PositonalEncoding(nn.Module):
    def __init__(self,max_len,d_model):
        super().__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        
        self.register_buffer('pe', self.build_encoding_matrix())
        
        
        
        
        
        
    def forward(self,x):
        pe_len = x.size(1)
        
        x = x + self.pe[:pe_len,:]
        
        return x
    
    
    def build_encoding_matrix(self):
        
        # 1. THE WAREHOUSE (The empty canvas)
        #the empty dashboards
        # Every single dashboard has exactly d_model empty slots for speedometers.
        pe = torch.zeros((self.max_len,self.d_model))
        
        
        # 2. THE MILEAGE COUNTERS (The vertical axis)
        #to perform operations between tensors of different shapes, you use unsqueeze to align them
        
        #the vertical axis
        position = torch.arange(self.max_len).unsqueeze(1)
        
        # 3. THE GEAR SPEEDS (The horizontal row)
        #the horizontal row
        #we step by 2 because each step generates both a sine and a cosine
        div_term = torch.exp(torch.arange(0,self.d_model,2).float() * (-math.log(10000.0)/self.d_model))
        
        # 4. INSTALLING THE SPEEDOMETERS (Stamping the watermark)
        # 3. Apply sin to even indices (0, 2, 4...) and cos to odd indices (1, 3, 5...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
        
                
                
        
        
    