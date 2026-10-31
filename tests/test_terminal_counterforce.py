"""Regression tests for terminal anti-opposition objective (v4.4)."""
import unittest
import numpy as np
from decentralized_nmpc.controller import LocalNMPC, NMPCConfig

class TestTerminalCounterforce(unittest.TestCase):
    @staticmethod
    def controller(i=2):
        return LocalNMPC(i, np.array([50.,-100.,0.]),
                         np.array([[0.,-29.],[0.,29.]]),
                         np.array([(-.5 if i==2 else .5)*320.,29.]),
                         1, 8.82e5, NMPCConfig(mission_guidance=True))

    def test_only_stalled_near_berth(self):
        c=self.controller()
        ship=np.array([50.,-91.3,0.,0.012,0.,0.])
        self.assertEqual(c._terminal_counterforce_gate(ship)[0],0.)
        c.stall_factor=1.
        self.assertGreater(c._terminal_counterforce_gate(ship)[0],.5)
        self.assertEqual(c._terminal_counterforce_gate(np.array([50.,-100.,0.,0.,0.,0.]))[0],0.)
        self.assertEqual(c._terminal_counterforce_gate(np.array([50.,-91.3,.2,0.,0.,0.]))[0],0.)
        self.assertEqual(c._terminal_counterforce_gate(np.array([50.,-50.,0.,0.,0.,0.]))[0],0.)

    def test_own_force_opposition_only(self):
        ship=np.array([50.,-91.3,0.,0.012,0.,0.])
        tow=self.controller(2); push=self.controller(0)
        tow.stall_factor=push.stall_factor=1.
        gate,unit=tow._terminal_counterforce_gate(ship)
        self.assertGreater(gate,.5)
        tow_force=tow._force_signature(.60,20.)[:2]
        push_force=push._force_signature(.52,-20.)[:2]
        self.assertLess(float(tow_force@unit),0.)
        self.assertGreater(float(push_force@unit),0.)

    def test_no_private_peer_info_in_gate(self):
        import inspect
        text=inspect.getsource(LocalNMPC._terminal_counterforce_gate)
        self.assertNotIn('peer',text)
        self.assertNotIn('env.',text)

if __name__=='__main__': unittest.main()
