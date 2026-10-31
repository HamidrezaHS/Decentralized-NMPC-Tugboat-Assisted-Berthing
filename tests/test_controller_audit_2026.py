"""Targeted regressions for end-position convergence and fair joint guidance."""
import numpy as np
import pytest
from marl_game.env import BerthingGameEnv, EnvConfig
from decentralized_nmpc import NMPCConfig, LocalNMPC
from decentralized_nmpc.coordination_baselines import JointNMPC,CoordinationSettings


def setup():
    env=BerthingGameEnv(EnvConfig(train=False, initial_psi_deg=20.))
    cfg=NMPCConfig(mission_guidance=True, terminal_position_weight=.45,max_iterations=2)
    c=LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,env.plant.max_force,cfg)
    c.mission_start=np.array([0.,0.])
    return env,c,cfg


def test_braking_cost_detects_both_directions():
    _,c,_=setup()
    for x,v_sign in [(49.,1.),(51.,-1.)]:
        slow=np.array([x,-100.,0.,.002*v_sign,0.,0.])
        fast=np.array([x,-100.,0.,.08*v_sign,0.,0.])
        assert c._mission_cost(fast,None,1.,1.)>c._mission_cost(slow,None,1.,1.)


def test_outward_motion_detected_on_both_sides_of_target():
    _,c,_=setup()
    for x,v_sign in [(49.,-1.),(51.,1.)]:
        calm=np.array([x,-100.,0.,0.,0.,0.])
        outward=np.array([x,-100.,0.,.08*v_sign,0.,0.])
        assert c._mission_cost(outward,None,1.,1.)>c._mission_cost(calm,None,1.,1.)


def test_terminal_position_has_nontrivial_near_berth_penalty():
    _,c,_=setup()
    assert c._terminal_position_cost(8.)>c._terminal_position_cost(2.5)*5.
    assert c._terminal_position_cost(8.)>.1
    assert c._terminal_position_cost(100.)<c._terminal_position_cost(8.)


def test_local_stall_detection_operates_inside_38m():
    _,c,_=setup()
    ship=np.array([50.,-94.,0.,0.,0.,0.])
    c._update_stall_estimate(ship,0.)
    c._update_stall_estimate(ship,201.)
    assert c.stall_factor>0.


def test_joint_initializes_and_shares_same_guidance_model():
    env,_,cfg=setup()
    joint=JointNMPC(env,cfg,CoordinationSettings(method='central'))
    ship,hydro,inv=joint._update_model(env)
    assert all(x.mission_start is not None for x in joint.local)
    x=np.zeros(16)
    assert np.isfinite(joint._cost(x,ship,hydro,inv))
    assert joint.local[0].cfg.mission_guidance


def test_legacy_terminal_ablation_is_explicit():
    env,_,cfg=setup()
    a=LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,
                env.plant.max_force,NMPCConfig(terminal_position_weight=0))
    assert a._terminal_position_cost(10.)==0.
