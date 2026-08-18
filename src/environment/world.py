from dataclasses import dataclass


@dataclass
class World:
    width: float
    height: float
    food: object
    obstacles: object