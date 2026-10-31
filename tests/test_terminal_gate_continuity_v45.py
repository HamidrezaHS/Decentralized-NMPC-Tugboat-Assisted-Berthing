"""Regression for the v4.4 3-m dead zone and terminal chatter.

All tests use only public ship pose, own-ship motion and one controller.
These tests prove activation and smoothness, not mission-level success.
"""
import unittest
import numpy as np
from decentralized_nmpc.controller import LocalNMPC,NMPCConfig

class TerminalContinuity(unittest.TestCase):
    def setUp(self):
        self.ctrl=LocalNMPC(2,np.array([50.,-100.,0.]),
            np.array([[0.,-29.],[0.,29.]]),
            np.array([-160.,29.]),1,882000.,
            NMPCConfig(mission_guidance=True))
        self.ctrl.stall_factor=1.
    def state(self,d,vy=0.,psi=0.):
        return np.array([50.,-100.+d,psi,0.,vy,0.])
    def gate(self,d,vy=0.,psi=0.):
        return self.ctrl._terminal_counterforce_gate(self.state(d,vy,psi))[0]
    def test_active_at_previously_unprotected_2712_m(self):
        self.assertGreater(self.gate(2.712),0.5)
    def test_active_through_success_tolerance(self):
        self.assertGreater(self.gate(2.5),0.5)
        self.assertGreater(self.gate(2.1),0.)
        self.assertEqual(self.gate(0.),0.)
    def test_no_step_at_old_three_m_switch(self):
        self.assertLess(abs(self.gate(3.00001)-self.gate(2.99999)),1e-4)
    def test_smooth_near_zero(self):
        values=[self.gate(x) for x in np.linspace(0.,3.,301)]
        self.assertTrue(all(a<=b+1.e-12 for a,b in zip(values,values[1:])))
        self.assertLess(max(abs(b-a) for a,b in zip(values,values[1:])),.04)
    def test_no_penalty_when_moving_toward_fast_or_heading_wrong(self):
        self.assertEqual(self.gate(2.712,vy=-.03),0.)
        self.assertEqual(self.gate(2.712,psi=np.deg2rad(4.)),0.)
    def test_opposition_direction_matches_needed_motion(self):
        gate,unit=self.ctrl._terminal_counterforce_gate(self.state(2.712))
        self.assertGreater(gate,.5)
        self.assertAlmostEqual(unit[1],-1.)
        tow_n=self.ctrl._force_signature(.60,0.)[:2]
        self.assertLess(float(tow_n@unit),0.)

if __name__=='__main__':unittest.main()
