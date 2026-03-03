import torch
from functools import partial

from redge.gumbel_sampling import sample_gumbel

def bridge_mean(x_t, x_0_hat, s, t, alphas, sigmas):
    """
    s < t
    """
    mean = alphas[s] * x_0_hat + sigmas[s] * (x_t - alphas[t] * x_0_hat) / sigmas[t]
    return mean


def ddim_step(x_t, x_0_hat, s, t, alphas, sigmas):
    return bridge_mean(
        x_t=x_t,
        x_0_hat=x_0_hat,
        s=s,
        t=t,
        alphas=alphas,
        sigmas=sigmas,
    )


def sample_one_hot_gumbel(probs: torch.Tensor) -> torch.Tensor:
    logits = probs.detach().clamp_min(1e-20).log()
    gumbels = sample_gumbel(logits.shape, device=logits.device)
    idx = (logits + gumbels).argmax(dim=-1)
    return torch.nn.functional.one_hot(idx, probs.shape[-1]).to(probs.dtype)


def cat_denoiser(x_t, t, alphas, sigmas, logits):
    # q(x_t | x0)
    log_likelihood = (alphas[t] * x_t) / (sigmas[t] ** 2)

    # Posterior weights
    log_weights = logits + log_likelihood
    ex0 = torch.softmax(log_weights, dim=-1)

    return ex0


def ddim(initial_noise, denoiser_fn, n_steps, t_1):
    alphas, sigmas = mk_schedule(n_steps=n_steps, t_1=t_1)

    x_t = initial_noise

    for t in range(n_steps - 1, 0, -1):
        x0_hat = denoiser_fn(x_t, t, alphas, sigmas)
        if t-1 == 0:
            x_soft = x0_hat
            x_hard = sample_one_hot_gumbel(x0_hat)
            x_t = x_hard
        else:
            x_t = ddim_step(x_t, x0_hat, t - 1, t, alphas, sigmas)

    x_t = (x_t - x_soft).detach() + x_soft
    x_t.soft = x_soft

    return x_t


def redge(logits, n_steps, t_1, hard=True, **kwargs):
    denoiser_fn = partial(cat_denoiser, logits=logits)

    initial_noise = torch.randn(*logits.shape, device=logits.device)
    samples = ddim(
        initial_noise=initial_noise,
        denoiser_fn=denoiser_fn,
        n_steps=n_steps,
        t_1=t_1,
    )
    if not hard:
        return samples.soft
    return samples


def mk_schedule(n_steps, t_1):
    return _mk_fm_schedule(n_steps=n_steps, t_1=t_1)


def _mk_fm_schedule(n_steps, t_1):
    alpha_t1 = 1 - t_1
    alphas = torch.cat((torch.tensor([1.0]), 
                        torch.linspace(alpha_t1, 1e-4, n_steps - 1)))
    sigmas = 1 - alphas
    return alphas, sigmas
