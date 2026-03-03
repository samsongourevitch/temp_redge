import torch
from functools import partial

from redge.ddim import (
    mk_schedule,
    ddim_step,
    cat_denoiser,
    sample_one_hot_gumbel,
)


def ddim_mod(logits, initial_noise, n_steps, t_1):

    detached_denoiser_fn = partial(cat_denoiser, logits=logits.detach())
    denoiser_fn = partial(cat_denoiser, logits=logits)

    alphas, sigmas = mk_schedule(n_steps=n_steps, t_1=t_1)

    x_t = initial_noise

    for t in range(n_steps - 1, 0, -1):
        x0_hat = denoiser_fn(x_t, t, alphas, sigmas)
        if t > 1:
            x_t = ddim_step(x_t, x0_hat, t - 1, t, alphas, sigmas)
        else:
            x0hat_xdetached = denoiser_fn(
                x_t=x_t.detach(), t=t, alphas=alphas, sigmas=sigmas
            )
            x0hat_pdetached = detached_denoiser_fn(
                x_t=x_t, t=t, alphas=alphas, sigmas=sigmas
            )
            x0 = sample_one_hot_gumbel(x0_hat)
            x0_xdetached = (x0 - x0hat_xdetached).detach() + x0hat_xdetached
            x0_pdetached = (x0 - x0hat_pdetached).detach() + x0hat_pdetached
            return x0_xdetached, x0hat_xdetached, x0_pdetached

    return x_t


def reindge(logits, reindge_fn, n_steps, t_1, hard=True, **kwargs):
    if not hard:
        raise NotImplementedError("Soft sampling not defined for reindge as it is based on reinmax, use hard=True.")
    initial_noise = torch.randn(*logits.shape, requires_grad=True, device=logits.device)
    x0_xdetached, x0hat_xdetached, x0_pdetached = reindge_fn(
        logits=logits,
        initial_noise=initial_noise,
        n_steps=n_steps,
        t_1=t_1,
    )
    diff = (x0_xdetached - x0hat_xdetached).detach()

    grad_y_storage = {"value": None}
    called_on_logits = {"done": False}
    handles = {}

    def save_grad_y(grad):
        grad_y_storage["value"] = grad.detach()
        return grad

    def add_projection_to_logits(grad_logits):
        gy = grad_y_storage["value"]

        if not called_on_logits["done"]:
            called_on_logits["done"] = True
            grad_st = torch.autograd.grad(
                outputs=x0hat_xdetached,
                inputs=logits,
                grad_outputs=gy,
                retain_graph=True,
            )[0]
            dot = (gy * diff).sum(dim=-1, keepdim=True)
            proj = dot * diff
            grad_reinmax = 0.5 * (grad_st + proj.squeeze())
            handles["h_y"].remove()
            handles["h_logits"].remove()
            return grad_logits + grad_reinmax

        return grad_logits

    handles["h_y"] = x0_pdetached.register_hook(save_grad_y)
    handles["h_logits"] = logits.register_hook(add_projection_to_logits)

    return x0_pdetached
