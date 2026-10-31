"""Tow-tug operational geometry and a local, *predictive* feasibility filter.

A rope can only pull; it cannot force its tug to remain outside the hull. The
original bilateral constrained plant also allows negative line tension. This
module enforces action-selection limits and supplies *independent* post-step
hull-distance checks. It makes no claim of constraint-invariant closed-loop
control or unilateral rope dynamics; the latter require a complementarity or
slack/taut model.
"""
from __future__ import annotations
import numpy as np
from .types import wrap


def free_side_from_quay(target_heading: float, quay_normal: np.ndarray) -> int:
    """Side opposite the quay, as sign of the ship's BODY port-normal.

    The quay normal points from the target position TOWARDS the quay.
    Return +1 for the port side, -1 for starboard. Near exactly fore/aft
    aligned berths there is no unique side: reject rather than guessing.
    """
    port = np.array([-np.sin(target_heading), np.cos(target_heading)])
    dot = float(port @ (-np.asarray(quay_normal, float)))
    if abs(dot) < 1e-6:
        raise ValueError('Free side undefined: quay is longitudinal to the ship')
    return 1 if dot > 0 else -1


def lead_error(ship_heading: float, tug_heading: float,
               rope_relative_angle: float, side: int) -> float:
    """Zero when rope points inward normal to the free-side ship shell.

    The rope axis in this DAE points from the tug attachment towards the ship.
    Tug center is ship-attachment minus lci*rope_axis minus ati*tug_axis.
    """
    return wrap(tug_heading + rope_relative_angle - ship_heading + side*np.pi/2)


def heading_error(ship_heading: float, tug_heading: float, side: int) -> float:
    """Desired tug bow faces its rope's connection on the ship."""
    return wrap(tug_heading - ship_heading + side*np.pi/2)


def predictive_lead_errors(ship_heading: float, ship_yaw_rate: float,
                           tug_heading: float, tug_yaw_rate: float,
                           theta: float, theta_rate: float,
                           tug_yaw_acc: float, theta_acc: float,
                           ship_yaw_acc: float, side: int,
                           lookahead_s: float) -> tuple[float, float]:
    lead = lead_error(ship_heading,tug_heading,theta,side)
    yaw = heading_error(ship_heading,tug_heading,side)
    dw = tug_yaw_rate-ship_yaw_rate
    da = tug_yaw_acc-ship_yaw_acc
    lead = wrap(lead+lookahead_s*(dw+theta_rate)+.5*lookahead_s**2*(da+theta_acc))
    yaw = wrap(yaw+lookahead_s*dw+.5*lookahead_s**2*da)
    return lead,yaw


def tug_corners(center, heading, length=22.943, beam=8.5):
    """Conservative rectangular tug footprint in world coordinates."""
    verts=np.array([[-length/2,-beam/2],[length/2,-beam/2],
                    [length/2,beam/2],[-length/2,beam/2]])
    c,s=np.cos(heading),np.sin(heading)
    return verts@np.array([[c,s],[-s,c]])+np.asarray(center)


def hull_clearance(ship_outline_world, tug_center_world, tug_heading):
    """Nonnegative hull clearance; overlap is detected separately."""
    from shapely.geometry import Polygon
    ship=Polygon(ship_outline_world).buffer(0)
    tug=Polygon(tug_corners(tug_center_world,tug_heading))
    return float(ship.distance(tug)),bool(ship.intersects(tug))
