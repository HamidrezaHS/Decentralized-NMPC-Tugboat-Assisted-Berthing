# geom/dxf_loader.py
import numpy as np
import ezdxf
from shapely.geometry import Polygon
from shapely.ops import orient

def _close_if_needed(pts: np.ndarray, tol=1e-9) -> np.ndarray:
    if len(pts) < 3:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        pts = np.vstack([pts, pts[0]])
    return pts

def _poly_from_pts(pts: np.ndarray) -> Polygon:
    pts = _close_if_needed(np.asarray(pts, float))
    poly = Polygon(pts)
    poly = orient(poly, sign=1.0)
    poly = poly.buffer(0)  # fix self-intersections
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def read_best_closed_polygon_from_dxf(dxf_path: str, layer=None, unit_scale: float = 1.0, spline_chord=0.05) -> Polygon:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    polys = []

    for e in msp:
        if layer is not None and getattr(e.dxf, "layer", None) != layer:
            continue
        t = e.dxftype()
        try:
            if t == "LWPOLYLINE":
                pts = np.array([(p[0], p[1]) for p in e.get_points()], float) * unit_scale
                polys.append(_poly_from_pts(pts))
            elif t == "POLYLINE":
                verts = e.vertices if isinstance(e.vertices, list) else e.vertices()
                pts = np.array([(v.dxf.location.x, v.dxf.location.y) for v in verts], float) * unit_scale
                polys.append(_poly_from_pts(pts))
            elif t == "SPLINE":
                pts = np.array([(p[0], p[1]) for p in e.flattening(distance=spline_chord)], float) * unit_scale
                polys.append(_poly_from_pts(pts))
        except Exception:
            continue

    if not polys:
        raise ValueError("Failed to build polygon from DXF.")

    polys.sort(key=lambda P: P.area, reverse=True)
    return polys[0]
