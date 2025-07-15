from modules.circle import ZIndexedCircle
import numpy as np

class Particle:
    """粒子情報を保持するクラス
    
    ParticleAnalyzerでparticle_repositoryに登録する情報をまとめたクラス

    """
    def __init__(
            self,
            particle_id: int,
            slice_index: int,
            radius: float,
            coord: tuple[float, float],
            color: tuple[np.uint8, np.uint8, np.uint8],
            dots_per_micron: float):
        
        self.id = particle_id
        self.color = color
        self.slices: list[ZIndexedCircle] = [ZIndexedCircle(coord, radius, slice_index)]
        self.dots_per_micron = dots_per_micron


    def add_slice(self, slice_index: int, radius: float, coord: tuple[float, float]):
        self.slices.append(ZIndexedCircle(coord, radius, slice_index))


    @property
    def max_radius(self):
        return max(c.radius for c in self.slices)


    @property
    def diameter_micron(self):
        return self.max_radius * 2 / self.dots_per_micron