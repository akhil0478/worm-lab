from dataclasses import dataclass

import numpy as np


@dataclass
class Reservoir:
   

    C: np.ndarray
    V: np.ndarray
    W: np.ndarray
    I: np.ndarray
    S: np.ndarray
    v: np.ndarray

    threshold: float = 1
    reset: float = 0.01
    learning_rate: float = 0.001
    retention: float = 0.97
   