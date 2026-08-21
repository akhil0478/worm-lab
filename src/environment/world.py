from dataclasses import dataclass

WORLD_L = 100.0


@dataclass
class World:
    food: object
    obstacles: object