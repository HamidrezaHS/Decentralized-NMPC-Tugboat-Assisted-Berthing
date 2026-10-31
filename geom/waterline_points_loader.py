# geom/waterline_points_loader.py
from __future__ import annotations
import numpy as np
import ezdxf
from shapely.geometry import Polygon


def load_waterline_points_dxf(dxf_path: str):
    """
    Read DXF that contains POINT entities for the ship waterline.
    Steps:
      1) read POINTs
      2) center x at midship (midship => x=0)
      3) circular sort points (angle around centroid)
      4) close polygon
    Returns:
      pts_closed: (N,2) array, last==first
      poly: shapely Polygon (mostly for intersections / sanity)
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    pts = []
    for e in msp:
        if e.dxftype() == "POINT":
            x, y, _z = e.dxf.location
            pts.append((x, y))

    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 10:
        raise ValueError("Not enough POINT entities in DXF. Expect waterline_points.dxf with many POINTs.")

    # midship center (x=0)
    x_mid = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    pts[:, 0] -= x_mid

    # circular sort (important)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]

    # close
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts_closed = np.vstack([pts, pts[0]])
    else:
        pts_closed = pts.copy()

    poly = Polygon(pts_closed)

    # fix if invalid (fallback)
    if (not poly.is_valid) or (poly.area <= 0):
        poly = poly.buffer(0)
        if (not poly.is_valid) or (poly.area <= 0):
            poly = Polygon(pts_closed).convex_hull
        pts_closed = np.asarray(poly.exterior.coords, dtype=float)

    return pts_closed, poly


def scale_points_to_Lpp(pts_closed: np.ndarray, Lpp_target: float) -> np.ndarray:
    """
    Uniformly scale points so x-extent becomes Lpp_target.
    Assumes x is already midship-centered (x=0 at midship).
    """
    x = pts_closed[:, 0]
    L_est = float(x.max() - x.min())
    if L_est <= 1e-9:
        raise ValueError("Bad waterline x-extent; cannot scale.")
    s = Lpp_target / L_est
    return pts_closed * s
