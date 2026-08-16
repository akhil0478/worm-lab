import numpy as np

from .reservoir import Reservoir


def step(reservoir: Reservoir):
  
    reservoir.v = (reservoir.retention * reservoir.V + reservoir.W @ reservoir.S)
       

    reservoir.V = (reservoir.retention * reservoir.V + reservoir.W @ reservoir.S + reservoir.I)
    reservoir.S = (reservoir.V>= reservoir.threshold).astype(float)
    reservoir.V[reservoir.S == 1] = reservoir.reset

    
    return reservoir.V, reservoir.v

 