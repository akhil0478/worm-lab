from dataclasses import dataclass

import numpy as np


@dataclass
class FoodWorld:

    def food_function(self, x: float, y: float) -> float:
        return np.exp(-(x**2 + y**2) / 2)

    def query(self, x: float, y: float) -> float:
        return self.food_function(x, y)