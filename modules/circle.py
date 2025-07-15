from dataclasses import dataclass

@dataclass
class Circle:
    coord: tuple[float, float]
    radius: float

@dataclass
class IdentifiedCircle(Circle):
    id: int

@dataclass
class ZIndexedCircle(Circle):
    z: int