from dataclasses import dataclass

@dataclass
class ObstacleWorld:
    def query(self, x: float, y: float)-> float:
        return 0.0