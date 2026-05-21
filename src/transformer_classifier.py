import torch
import torch.nn as nn
from positional_encoding import PositonalEncoding
from encoder_block import EncoderBlock


class TransformerClassifier(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,num_classes ):
        super().__init__()
        
        
        #1. The dictionary
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        
        #2. The timestamp
        self.pos_encoding = PositonalEncoding(d_model)
        
        #3. The stack of blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model,num_heads) for _ in range(num_layers)
        ])
        
        
        self.classifier = nn.Linear(d_model,num_classes)
      
        
    def forward(self,X,mask):
        
        out1 = self.embedding(X)
        
        
        out2 = self.pos_encoding(out1)
        
        out3 = out2
        
        for layer in self.layers:
            out3 = layer(out3,mask)
            
        
        out4 = out3.mean(dim=1)
        
        out5 = self.classifier(out4)
        
        return out5
            
        
        