# import numpy as np
# import torch
# from torch import nn

# class TransformerBlock():
#     def __init__(self, embedDim, numHeads):
#         self.embedDim = embedDim
#         self.numHeads = numHeads

#         # Multi-head attention
#         self.attention = nn.MultiheadAttention(embed_dim=self.embedDim, num_heads=self.numHeads, dropout=0.1)

#         # Feedforward
#         self.feedforward = nn.Sequential(nn.Linear(self.embedDim, self.embedDim), nn.ReLU())

#         # Layer normalization
#         self.layernorm = nn.LayerNorm(self.embedDim)
    
#     def forward(self, x):
#         # Multi-head attention
#         x = x + self.attention(x, x, x)[0]
#         x = self.layernorm(x)

#         # Feedforward
#         x = x + self.feedforward(x)
#         x = self.layernorm(x)
#         return x

# class Transformer():
#     def __init__(self, inputDim):
#         self.inputDim = inputDim
#         self.numBlocks = 2
#         self.embedDim = 128
#         self.numHeads = 2

#         # input embedding
#         self.inputEmbedding = nn.Linear(self.inputDim, self.embedDim)

#         # transformer blocks
#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(self.embed_dim, self.num_heads, config) 
#             for _ in range(self.num_blocks)])
    
#     def forward(self, x):
#         x = self.transformer(x)[0]  # Get the last hidden state
#         logits = self.fc(x[:, -1, :])  # Take the last token's output
#         return logits