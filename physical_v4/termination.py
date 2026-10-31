"""Online berthing success monitor; independent of game optimiser and dynamics.

The exact same thresholds are used online and in the final report.
An interval contributes dt seconds ONLY if all four terminal requirements hold
AND there has been no unsafe sample in the mission so far.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BerthingCriteria:
    position_m: float = 2.5
    heading_deg: float = 2.0
    speed_mps: float = 0.045
    yaw_rate_deg_s: float = 0.035
    hold_s: float = 5.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('All berthing thresholds and hold time must be finite positive numbers')


class BerthingMonitor:
    """Deterministic timer for consecutive admissible POST-STEP intervals."""

    def __init__(self, criteria: BerthingCriteria | None = None):
        self.criteria = criteria or BerthingCriteria()
        self.hold_elapsed_s = 0.0
        self.first_success_time_s: float | None = None
        self.ever_unsafe = False
        self.last_terminal_conditions_met = False
        # Audit of every physical integration substep AFTER the initial safe-berthing event.
        self.post_success_elapsed_s = 0.0
        self.post_success_in_limits_s = 0.0
        self.post_success_violations = 0
        self.post_success_samples = 0
        self.post_success_max_position_error_m = 0.0
        self.post_success_max_heading_error_deg = 0.0
        self.post_success_max_speed_mps = 0.0
        self.post_success_max_yaw_rate_deg_s = 0.0
        self.post_success_min_hull_gap_m = float('inf')

    def update(self, *, t: float, dt: float, position_error_m: float,
               heading_error_deg: float, speed_mps: float, yaw_rate_deg_s: float,
               safe: bool = True, hull_gap_m: float | None = None) -> bool:
        """Returns True once a 5-second admissible hold has FIRST completed.

        Time points represent ends of intervals; each good step contributes dt.
        Past violations disqualify a *safe* success for this mission.
        """
        if dt <= 0 or not math.isfinite(t) or not math.isfinite(dt):
            raise ValueError('t and dt must be finite, and dt positive')
        self.ever_unsafe |= not safe
        c = self.criteria
        terminal = (math.isfinite(position_error_m) and position_error_m <= c.position_m
                    and math.isfinite(heading_error_deg) and abs(heading_error_deg) <= c.heading_deg
                    and math.isfinite(speed_mps) and speed_mps <= c.speed_mps
                    and math.isfinite(yaw_rate_deg_s) and abs(yaw_rate_deg_s) <= c.yaw_rate_deg_s)
        self.last_terminal_conditions_met = bool(terminal)
        if self.first_success_time_s is not None:
            # The success-triggering substep is NOT part of the subsequent hold.
            self.post_success_elapsed_s += dt
            self.post_success_samples += 1
            self.post_success_max_position_error_m = max(self.post_success_max_position_error_m, abs(position_error_m))
            self.post_success_max_heading_error_deg = max(self.post_success_max_heading_error_deg, abs(heading_error_deg))
            self.post_success_max_speed_mps = max(self.post_success_max_speed_mps, abs(speed_mps))
            self.post_success_max_yaw_rate_deg_s = max(self.post_success_max_yaw_rate_deg_s, abs(yaw_rate_deg_s))
            if hull_gap_m is not None:
                self.post_success_min_hull_gap_m = min(self.post_success_min_hull_gap_m, float(hull_gap_m))
            if terminal and safe:
                self.post_success_in_limits_s += dt
            else:
                self.post_success_violations += 1
        if terminal and not self.ever_unsafe:
            self.hold_elapsed_s += dt
        else:
            self.hold_elapsed_s = 0.0
        # eps only addresses representational error in summing time increments.
        arrived = self.hold_elapsed_s + 1e-9 >= c.hold_s
        if arrived and self.first_success_time_s is None:
            self.first_success_time_s = float(t)
            return True
        return False
