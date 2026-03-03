import torch

def sample_gumbel(shape, device):
    """Sample Gumbel noise of given shape.

    Returns a tensor of Gumbel(0,1) samples with the provided shape.
    """
    g = torch.empty(shape, device=device)
    g.exponential_()
    g.log_().neg_()
    return g


def gumbel_softmax(logits, tau, hard=True, **kwargs):
    gumbels = sample_gumbel(logits.shape, device=logits.device)
    probs = torch.softmax((logits + gumbels) / tau, dim=-1)

    sample = torch.nn.functional.one_hot(probs.detach().argmax(-1), probs.shape[-1]).to(
        probs.dtype
    )
    sample_st = sample + (probs - probs.detach())
    sample_st.soft = probs

    if not hard:
        return sample_st.soft
    return sample_st
