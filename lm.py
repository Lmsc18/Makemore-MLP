import torch 
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import pickle

with open('params.pkl', 'rb') as f:
    parameters = pickle.load(f)
with open('itos.pkl', 'rb') as f:
    itos = pickle.load(f)

block_size=3
C=parameters[0]
W1=parameters[1]
b1=parameters[2]
W2=parameters[3]
b2=parameters[4]


def generator():
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    name=''.join(itos[i] for i in out[:-1])
    return name