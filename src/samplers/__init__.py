from samplers.ddim import redge
from samplers.gumbel_sampling import gumbel_softmax
from samplers.redge_cov import redge_cov
from samplers.reinmax import reinmax
from samplers.st import straight_through
from samplers.reindge import ddim_mod, reindge
from functools import partial


SAMPLERS = {
    "st": straight_through,
    "redge": redge,
    "reinmax": reinmax,
    "redge_cov": redge_cov,
    "gumbel": gumbel_softmax,
    "reindge": partial(reindge, reindge_fn=ddim_mod),
}
