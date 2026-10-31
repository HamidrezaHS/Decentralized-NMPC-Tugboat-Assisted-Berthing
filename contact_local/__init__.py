"""Strict contact-point-only information interface and baseline NMPC.

No component in this package needs access to the vessel pose, velocities,
other tugs, or environmental truth. The simulator is the only truth holder.
"""
from .sensors import ContactObservation, contact_observation, contact_goal
from .controller import ContactNMPC, ContactNMPCConfig
from .actuation import ContactOnlyHelm
