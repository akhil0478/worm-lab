import numpy as np


class Prediction:
    def __init__(self, sensory_mask: np.ndarray):
        self.sensory_mask = sensory_mask.astype(float)

        
        self.previous = np.zeros(280, dtype=float)
        self.current = np.zeros(280, dtype=float)

    def update(self, v: np.ndarray):
        
        self.previous = self.current.copy()

       
        self.current = self.sensory_mask * v

        return self.previous, self.current