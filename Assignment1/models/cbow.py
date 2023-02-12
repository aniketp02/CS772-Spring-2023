import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        # print('embeds', type(embeds))
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob