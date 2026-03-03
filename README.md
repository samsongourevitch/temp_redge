# ReDGE

Code for *Categorical Reparameterization using Denoising Diffusion models* ([arXiv:2601.00781](https://arxiv.org/abs/2601.00781)).

## Implemented samplers

- `redge`
- `redge_cov`
- `reindge`
- `st`
- `gumbel`
- `reinmax`

## Public API

We provide a simple API for sampling from ReDGE-style samplers:

```python
from redge import redge, SAMPLERS

x = redge(logits, n_steps=5, t_1=0.5, hard=True)
y = SAMPLERS["redge"](logits, n_steps=5, t_1=0.5, hard=True)
```

## Hyperparameters and tuning

ReDGE-style samplers expose two main knobs:

- `n_steps` (int): number of DDIM reverse steps
- `t_1` (float): diffusion endpoint / relaxation strength (eqiuvalent to the temperature in Gumbel-based samplers)

Practical defaults:

- `n_steps`: `3-9` (start with `5`)
- `t_1`: `0.3-0.9` (start with `0.5`)

Guidelines:

- If optimization is unstable/noisy, increase `t_1`
- If samples are too soft/biased, decrease `t_1` or increase `n_steps` slightly

## Citation

If you use this repository, please cite:

```bibtex
@article{gourevitch2026redge,
  title={Categorical Reparameterization with Denoising Diffusion models},
  author={Samson Gourevitch and Alain Durmus and Eric Moulines and Jimmy Olsson and Yazid Janati},
  year={2026},
  eprint={2601.00781},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.00781}
}
```
