from dataclasses import dataclass
from typing import Callable


@dataclass
class FoodWorld:
    food_function: Callable[[float, float], float]

    def query(self, x: float, y: float) -> float:
        return self.food_function(x, y)