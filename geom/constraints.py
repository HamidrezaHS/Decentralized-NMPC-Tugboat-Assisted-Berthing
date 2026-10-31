import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

from .dxf_loader import read_best_closed_polygon_from_dxf
from .waterline_points_loader import load_waterline_points_dxf, scale_points_to_Lpp


class Constraints:

    def __init__(self, dxf_path, kind="auto", Lpp_target=None):
        self.dxf_path = dxf_path
        self.kind = kind
        self.Lpp_target = Lpp_target

        self.points, self.polygon = self._load_geometry()

    def _load_geometry(self):
        if self.kind == "points":
            pts_closed, poly = load_waterline_points_dxf(self.dxf_path)
            if self.Lpp_target is not None:
                pts_closed = scale_points_to_Lpp(pts_closed, self.Lpp_target)
                poly = Polygon(pts_closed)
            return pts_closed, poly

        if self.kind == "poly":
            poly = read_best_closed_polygon_from_dxf(self.dxf_path)
            pts_closed = np.asarray(poly.exterior.coords, dtype=float)
            return pts_closed, poly

        # AUTO MODE
        try:
            pts_closed, poly = load_waterline_points_dxf(self.dxf_path)
            if self.Lpp_target is not None:
                pts_closed = scale_points_to_Lpp(pts_closed, self.Lpp_target)
                poly = Polygon(pts_closed)
            return pts_closed, poly
        except Exception:
            poly = read_best_closed_polygon_from_dxf(self.dxf_path)
            pts_closed = np.asarray(poly.exterior.coords, dtype=float)
            return pts_closed, poly

    def get_y(self, x):
        vertical = LineString([(x, -1e6), (x, 1e6)])
        inter = self.polygon.boundary.intersection(vertical)

        if inter.is_empty:
            return []

        if inter.geom_type == "Point":
            return [(inter.x, inter.y)]

        if inter.geom_type == "MultiPoint":
            return [(p.x, p.y) for p in inter.geoms]

        if inter.geom_type == "LineString":
            coords = list(inter.coords)
            return [(coords[0][0], coords[0][1]),
                    (coords[-1][0], coords[-1][1])]
        
        try:
            return [(p.x, p.y) for p in inter.geoms]
        except:
            return []

    def beam_at_x(self, x):
        pts = self.get_y(x)
        if len(pts) < 2:
            return 0
        ys = sorted([p[1] for p in pts])
        return ys[-1] - ys[0]

    def plot(self, title=None, color="blue"):

        pts = self.points
        poly = self.polygon

        plt.figure(figsize=(8, 6))

        plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)
        plt.scatter(pts[:-1, 0], pts[:-1, 1], color=color, s=8, alpha=0.5)

        # centroid
        c = poly.centroid
        plt.scatter([c.x], [c.y], color=color, marker="x", s=60)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title if title else f"Waterline: {self.dxf_path}")
        plt.tight_layout()
        plt.show()
