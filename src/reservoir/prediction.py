import numpy as np


class Prediction:
    def __init__(self, sensory_mask: np.ndarray):
        self.sensory_mask = sensory_mask.astype(float)

        size = len(sensory_mask)

        self.previous = np.zeros(
            size,
            dtype=float,
        )

        self.current = np.zeros(
            size,
            dtype=float,
        )

    def update(self, v: np.ndarray):
        self.previous = self.current.copy()

        self.current = self.sensory_mask * v

        return self.previous, self.current