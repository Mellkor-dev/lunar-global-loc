# maps/global_dem.py
import numpy as np
from scipy.ndimage import map_coordinates

class GlobalDEM:
    def __init__(self, elevation: np.ndarray, lxy: float, origin_xy: tuple[float, float] = (0.0, 0.0)):
        self.elevation = elevation
        self.lxy = lxy
        self.origin_xy = origin_xy

    def xy_to_ij(self, x, y):
        i = (y - self.origin_xy[1]) / self.lxy
        j = (x - self.origin_xy[0]) / self.lxy
        return i, j

    def interpolate_z(self, x, y):
        i, j = self.xy_to_ij(x, y)
        return map_coordinates(self.elevation, [[i], [j]], order=1, mode='nearest')[0]

    @classmethod
    def from_omnilrs_terrain_manager(cls, tm) -> "GlobalDEM":
        """
        tm: an already-constructed TerrainManager with randomizeTerrain() already called.
        """
        if tm._DEM is None:
            raise ValueError("Call tm.randomizeTerrain() before extracting the DEM.")
        lxy = tm._grid_size  # 0.025 for lunaryard_20m
        return cls(elevation=tm._DEM.copy(), lxy=lxy, origin_xy=(0.0, 0.0))

    @classmethod
    def from_npy(cls, path, lxy, origin_xy=(0.0, 0.0)):
        return cls(np.load(path), lxy, origin_xy)