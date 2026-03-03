import torch

from samplers.gumbel_sampling import sample_gumbel

def sample_one_hot_gumbel(logits):
    gumbels = sample_gumbel(logits.shape, device=logits.device)
    idx = (logits.detach() + gumbels).argmax(dim=-1)
    return torch.nn.functional.one_hot(idx, logits.shape[-1]).to(logits.dtype)

def straight_through(logits, hard=True, **kwargs):
    probs = logits.softmax(-1)
    sample = sample_one_hot_gumbel(logits)
    sample = (sample - probs).detach() + probs
    sample.soft = probs

    if not hard:
        return sample.soft
    return sample
