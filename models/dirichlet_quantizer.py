import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.kl import kl_divergence

from models.utils import measure_perplexity, convert_to_alpha

class DirichletQuantizer(nn.Module):
    def __init__(self, codebook_config, quantizer_config):
        super(DirichletQuantizer, self).__init__()

        self.num_embeddings = codebook_config.num_embeddings
        self.embedding_dim = codebook_config.embedding_dim
        self.temp = quantizer_config.temp
        self.init_type = codebook_config.init_type
        self.kl_weight = quantizer_config.kl_weight

        self.quantizer_config = quantizer_config
        self.codebook_config = codebook_config

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if self.init_type == "uniform": # otherwise, it is initialized with normal distribution
            self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 
                                                1.0 / self.num_embeddings)

    def forward(self, logits, iter):
        alpha = convert_to_alpha(logits, self.quantizer_config).permute(0, 2, 3, 1).contiguous()
        posterior = Dirichlet(alpha)

        S = alpha.sum(-1, keepdims=True)

        probs = alpha / S

        dist = RelaxedOneHotCategorical(temperature=self.temp, probs=probs)

        if self.training:
            sample = dist.rsample()
            vector_weights = sample.permute(0, 3, 1, 2)
            indices = vector_weights.argmax(dim=1)
            encodings = sample.reshape(-1, self.num_embeddings)
        else:
            # in eval mode, directly quantize with the corresponding codebook vector
            indices = torch.argmax(logits, dim=1)
            one_hot = F.one_hot(indices, self.num_embeddings).float()
            vector_weights = one_hot.permute(0, 3, 1, 2)
            encodings = one_hot.reshape(-1, self.num_embeddings)

        z_q = einsum('b n h w, n d -> b d h w', vector_weights, self.codebook.weight)

        prior = Dirichlet(torch.ones(alpha.shape).cuda())
        kl_loss = self.kl_weight * (kl_divergence(posterior, prior).mean())
        
        perplexity = measure_perplexity(encodings)

        return z_q, kl_loss, indices, perplexity






        