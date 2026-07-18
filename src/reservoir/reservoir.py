from dataclasses import dataclass

import numpy as np


@dataclass
class Reservoir:
    """
    Represents the current state of the recurrent neural reservoir.
    """

    C: np.ndarray
    V: np.ndarray

    @property
    def size(self) -> int:
        return self.C.shape[0]
