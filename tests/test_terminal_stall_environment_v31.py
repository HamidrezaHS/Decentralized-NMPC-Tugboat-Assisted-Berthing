import numpy as np
from decentralized_nmpc.controller import LocalNMPC, NMPCConfig


def controller():
    return LocalNMPC(2, np.array([50.,-100.,0.]),
                     np.array([[-160.,-30.],[160.,30.]]),
                     np.array([-160.,30.]),1,882598.,NMPCConfig())


def test_low_progress_near_goal_detected_without_peer_states():
    c=controller()
    c._update_stall_estimate(np.array([50.,-90.,0.,0.,-.001,0.]),0.)
    c._update_stall_estimate(np.array([50.,-90.4,0.,0.,-.001,0.]),200.)
    assert c.stall_factor>=.39
    gate,needed=c._terminal_counterforce_gate(np.array([50.,-90.4,0.,0.,-.001,0.]))
    assert gate>0
    assert needed[1]<0


def test_legitimate_near_goal_progress_does_not_trigger_stall():
    c=controller()
    c._update_stall_estimate(np.array([50.,-90.,0.,0.,-.01,0.]),0.)
    c._update_stall_estimate(np.array([50.,-92.,0.,0.,-.01,0.]),200.)
    assert c.stall_factor==0.
