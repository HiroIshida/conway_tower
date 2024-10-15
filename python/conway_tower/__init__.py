import numpy as np
from typing import Optional
from ._conway_tower import evolve_conway as _evolve_conway


def evolve_conway(init_state: np.ndarray,
                  n_steps: int,
                  perturb: Optional[np.ndarray] = None,
                  z_as_time: bool = False) -> np.ndarray:
    assert init_state.ndim == 2
    assert init_state.dtype == bool
    if perturb is not None:
        assert perturb.shape == (n_steps, 2)
        assert perturb.dtype == int
    out = _evolve_conway(init_state, n_steps, perturb)
    if(z_as_time):
        out = np.moveaxis(out, 0, -1)
    return out
