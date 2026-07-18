import numpy as np

from .reservoir import Reservoir


def step(reservoir: Reservoir):
    """
    Perform one timestep of recurrent propagation.

    V(t+1) = C @ V(t)
    """

    reservoir.V = reservoir.C @ reservoir.V

    activity_index = np.sum(reservoir.V)

    return activity_index, reservoir.V
