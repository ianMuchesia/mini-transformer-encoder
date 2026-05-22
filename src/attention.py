import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" 
        
        self.head_dims = d_model//num_heads
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def compute_qkv(self,X):
        
        Q = self.W_q(X)
        
        K = self.W_k(X)
        
        V = self.W_v(X)
        
        return Q,K,V
        
        
    def split_heads(self,X):
        
        Q,K,V = self.compute_qkv(X)
        
        #print(f"Q shape before head splitting {Q.shape}")
        Q = Q.reshape(Q.size(0),Q.size(1),self.num_heads,self.head_dims).transpose(1,2)
        
        # print(f"Q shape after head splitting {Q.shape}")
        
        # print(f"K shape before head splitting {K.shape}")
        
        K = K.reshape(K.size(0),K.size(1),self.num_heads,self.head_dims).transpose(1,2)
        
        # print(f"K shape after head splitting {K.shape}")

        # print(f"V shape before head splitting {V.shape}")

        V = V.reshape(V.size(0),V.size(1),self.num_heads,self.head_dims).transpose(1,2)
        
        # print(f"V shape after head splitting {V.shape}")

        
        return Q, K, V
        
        
    def compute_scores(self,Q,K):
        
        scores = Q @ K.transpose(-2,-1)
        
        scores = scores/math.sqrt(K.shape[-1])
        
        return scores
        
        
    def softmax(self,scores,V):
        
        
        
        weights =  torch.softmax(scores,dim=-1)
    
        outputs = weights @ V
        
        return outputs,weights
    
    
    def apply_mask(self, scores, mask):
        if mask is not None:
            return scores.masked_fill(mask == 0, -1e9)
        return scores
    
    
    def forward(self,X,mask=None):
        
        q, k, v = self.split_heads(X)
        
        scores = self.compute_scores(q,k)
        
        scores = self.apply_mask(scores,mask)
        
        output,weights = self.softmax(scores,v)
        
        output = output.transpose(1,2)
        
        output = output.reshape(output.size(0), output.size(1), -1)
        
        output = self.W_o(output)
        
        return output,weights
        
        
        
        
        
  
        
        
        
        
        
        