import torch
from functools import partial

from samplers.ddim import ddim

MIN_VAR = 1e-1

def fm_cat_denoiser_cov(x_t, t, alphas, sigmas, logits, probs):
    # q(x_t | x0)
    final_var = (probs * (1 - probs)).clamp(min=MIN_VAR)
    log_likelihood = (
        (alphas[t] * x_t - alphas[t] * sigmas[t] * probs - (alphas[t] ** 2) / 2)
    ) / (final_var * (sigmas[t] ** 2))

    # Posterior weights
    log_weights = logits + log_likelihood
    ex0 = torch.softmax(log_weights, dim=-1)
    return ex0


def redge_cov(logits, n_steps, t_1, hard=True, **kwargs):
    probs = logits.softmax(-1)
    # clip probs to avoid numerical issues
    denoiser_fn = partial(
        fm_cat_denoiser_cov,
        logits=logits,
        probs=probs,
    )

    initial_noise = probs + ((probs * (1 - probs)).clamp(min=MIN_VAR)).sqrt() * torch.randn(*logits.shape, device=logits.device)
    samples = ddim(
        initial_noise=initial_noise,
        denoiser_fn=denoiser_fn,
        n_steps=n_steps,
        t_1=t_1,
    )

    if not hard:
        return samples.soft
    return samples
