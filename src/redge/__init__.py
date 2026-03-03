from functools import partial

from .ddim import redge
from .gumbel_sampling import gumbel_softmax
from .redge_cov import redge_cov
from .reindge import ddim_mod, reindge
from .reinmax import reinmax
from .st import straight_through

SAMPLERS = {
    "st": straight_through,
    "redge": redge,
    "reinmax": reinmax,
    "redge_cov": redge_cov,
    "gumbel": gumbel_softmax,
    "reindge": partial(reindge, reindge_fn=ddim_mod),
}

__all__ = [
    "SAMPLERS",
    "gumbel_softmax",
    "redge",
    "redge_cov",
    "reindge",
    "reinmax",
    "straight_through",
]
