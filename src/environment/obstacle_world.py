from dataclasses import dataclass

from .world import WORLD_L


@dataclass
class ObstacleWorld:

    def query(self, x: float, y: float) -> float:
        if x == 0 or y == 0 or x == WORLD_L or y == WORLD_L:
            return -1.0

        return 0.0