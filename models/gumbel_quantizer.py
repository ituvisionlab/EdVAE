import torch
from torch import nn, einsum
import torch.nn.functional as F

from models.utils import measure_perplexity

class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_config, quantizer_config):
        super(GumbelQuantizer, self).__init__()

        self.num_embeddings = codebook_config.num_embeddings
        self.embedding_dim = codebook_config.embedding_dim
        self.temp = quantizer_config.temp
        self.kl_weight = quantizer_config.kl_weight
        self.init_type = codebook_config.init_type

        self.quantizer_config = quantizer_config
        self.codebook_config = codebook_config

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        if self.init_type == "uniform": # otherwise, it is initialized with normal distribution
            self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 
                                                1.0 / self.num_embeddings)

    def forward(self, logits, iter):
        if self.training:
            # soft one-hot conversion from encoder's logits
            vector_weights = F.gumbel_softmax(logits, tau=self.temp, hard=False, dim=1)
            indices = vector_weights.argmax(dim=1)
        else:
            indices = logits.argmax(dim=1)
            vector_weights = F.one_hot(indices, self.num_embeddings).float().permute(0, 3, 1, 2)
        
        # use these soft one-hot representation as weights of each vector
        z_q = einsum('b n h w, n d -> b d h w', vector_weights, self.codebook.weight)

        # convert encoder's logits to probabilities
        posterior = F.softmax(logits, dim=1)

        # measure kl distance between these probabilities and uniform dist
        kl_loss = self.kl_weight * torch.sum(posterior * torch.log(posterior * self.num_embeddings + 1e-10), dim=1).mean()
    
        # measure perplexity
        encodings = F.one_hot(indices, self.num_embeddings).float().reshape(-1, self.num_embeddings)
        perplexity = measure_perplexity(encodings)

        return z_q, kl_loss, indices, perplexity






        