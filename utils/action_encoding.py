import numpy as np
from envs.env_config import n_power_levels

ACTION_SHAPE = (n_power_levels,n_power_levels,48)
N_ACTIONS = n_power_levels * n_power_levels * 48

def encode(dispatch_idx:int , plan_idx:int , plan_slot:int) -> int:
    """Map a (dispatch_idx, plan_idx, plan_slot) tuple to a flat integer."""
    a_flat = np.ravel_multi_index((dispatch_idx, plan_idx, plan_slot),(11,11,48))
    return int(a_flat)

def decode(a_flat:int) -> tuple[int, int, int]:
    """Map a flat integer back to (dispatch_idx, plan_idx, plan_slot)."""
    d, p, s = np.unravel_index(int(a_flat), ACTION_SHAPE)
    return int(d), int(p), int(s)
