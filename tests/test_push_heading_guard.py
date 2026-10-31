"""Verification of the independent 0.2-s heading protection layer."""
import numpy as np
import pytest
from physical_v4.push_heading_guard import PushHeadingGuard
from marl_game.env import EnvConfig,BerthingGameEnv


def _guard():
    return PushHeadingGuard(side=1,max_thrust=900000.,stern_thruster_x_m=-11.47,max_yaw_moment_Nm=5e6)


def _poses(error_deg=0.,diverge_deg_s=0.):
    ship=np.zeros(6,float)
    tug=np.zeros(6,float)
    tug[2]=-np.pi/2-np.deg2rad(error_deg)
    tug[5]=-np.deg2rad(diverge_deg_s)
    return ship,tug


def test_unchanged_in_nominal_heading_and_yaw_rate():
    guard=_guard()
    ship,tug=_poses(3.,0.)
    out=guard.filter(ship,tug,220000.,0.10,200000.)
    assert out == (220000.,0.10,200000.,False)
    assert guard.n_interventions == 0


def test_early_intervention_uses_relative_yaw_rate_prediction():
    guard=_guard()
    ship,tug=_poses(7.,3.)  # predicted error 16 deg in 3 sec
    thrust,steer,yaw,active=guard.filter(ship,tug,400000.,0.,0.)
    assert active and guard.n_interventions==1
    assert yaw>0.
    assert steer<0.  # stern thruster assists positive yaw moment
    assert 0<=thrust<=400000.


def test_high_risk_guard_is_bounded_and_does_not_flip_push_thrust():
    guard=_guard()
    ship,tug=_poses(25.,1.)
    thrust,steer,yaw,active=guard.filter(ship,tug,850000.,0.,0.)
    assert active
    assert 0. <= thrust <= .30*900000.
    assert abs(steer)<=np.deg2rad(35.)
    assert abs(yaw)<=5e6
    assert guard.max_abs_error_deg>=24.99


def test_guard_is_reset_each_run_and_shared_across_methods():
    # The guard sits in the common physical environment, not inside an NMPC.
    env=BerthingGameEnv(EnvConfig(train=False,observation_noise=0.))
    assert len(env.heading_guards)==2
    assert env.cfg.push_heading_guard
    env.heading_guards[0].n_interventions=12
    env.reset()
    assert env.heading_guards[0].n_interventions==0
    assert all(g.side==env.side for g in env.heading_guards)


def test_legacy_ablation_disables_guard_but_preserves_monitor():
    env=BerthingGameEnv(EnvConfig(train=False,push_heading_guard=False))
    assert not env.cfg.push_heading_guard
    assert env.cfg.stop_on_unsafe
