from reinmax import reinmax as reinmax_fn

def reinmax(logits, tau, **kwargs):
    sample, probs = reinmax_fn(logits.reshape(-1, logits.shape[-1]), tau=tau, hard=True)
    sample = sample.reshape_as(logits)
    sample.soft = probs
    return sample
