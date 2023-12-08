import torch
import torch.nn.functional as F

def sample_gumbel(shape, mean=0, scale=1, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return mean - scale * (torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, scale, tau=1, dim=-1):
    g = sample_gumbel(logits.size(), scale=scale)
    y = logits + g
    return F.softmax(y/tau, dim=dim)

def measure_perplexity(encodings):
    # measure perplexity
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()

    return perplexity

def convert_to_alpha(logits, quantizer_config):
    conversion_type = quantizer_config.alpha_type
    if conversion_type == "relu":
        return F.relu(logits) + 1
    elif conversion_type == "gelu":
        return F.gelu(logits) + 2
    elif conversion_type == "exp":
        min, max = quantizer_config.clamp
        return torch.exp(logits.clamp(min=float(min), max=max)) + 1
    elif conversion_type == "softplus":
        return F.softplus(logits) + 1
    else:
        assert "not implemented"


