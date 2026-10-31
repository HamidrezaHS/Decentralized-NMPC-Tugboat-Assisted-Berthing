"""Distributed finite exact-potential game, recomputed at each sampling instant.

What is proved (for a FROZEN stage game): fixed finite individual action sets,
common potential P, and serialized STRICT better responses imply finite
termination at a pure Nash equilibrium of the DISCRETE SURROGATE GAME.

What is NOT proved: a Nash equilibrium of the original continuous dynamic game,
closed-loop stability, robust safety or unilateral rope/contact feasibility.

Communication: Each player broadcasts ONLY its *currently selected* contribution
(delta ship accelerations, own local cost and an opaque action index), not the
private ship/tug state, hidden forces or full candidate action catalog.
All four agents use the same public ship pose, ship velocity, target and measured
ship acceleration. No force allocator and no central objective optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .types import TugAction, TugObservation, wrap


@dataclass(frozen=True)
class ActionBid:
    action: TugAction
    marginal_acc: tuple[float, float, float]
    local_cost: float
    feasible: bool = True

    def __post_init__(self):
        if len(self.marginal_acc) != 3: raise ValueError('Acceleration must be 3-vector')
        if not np.isfinite([*self.marginal_acc, self.local_cost, self.action.thrust,
                            self.action.azimuth]).all():
            raise ValueError('Non-finite game bid')


@dataclass(frozen=True)
class GameMessage:
    agent_id: int
    action_index: int  # local opaque id, not a command to another tug
    marginal_acc: tuple[float, float, float]
    local_cost: float


def ship_cost(obs: TugObservation, public_acc: np.ndarray, effective_acc: np.ndarray,
              lookahead_s: float, desired_speed_mps: float, max_near_speed_mps: float,
              safety_near_gap_m: float) -> float:
    """Identical shared payoff model for ALL players in a given stage game.

    Constant-acceleration forecast is intentionally a *surrogate* and should
    not be mistaken for a full 4-tug constrained DAE rollout.
    """
    H=float(lookahead_s)
    x,y,psi=obs.ship_pose
    ue,ve=obs.ship_earth_velocity
    r=obs.ship_body_velocity[2]
    goal=np.asarray(obs.target_pose[:2],float)
    err=goal-np.array([x,y]);dist=float(np.linalg.norm(err))
    # Local predictions relative to observed physical aggregate acceleration.
    a=np.asarray(public_acc,float)+np.asarray(effective_acc,float)
    pred_v=np.array([ue,ve])+H*a[:2]
    pred_xy=np.array([x,y])+H*np.array([ue,ve])+.5*H*H*a[:2]
    pred_r=r+H*a[2]
    pred_heading=wrap(psi+H*r+.5*H*H*a[2])
    desired_v=err/max(180.,dist/max(desired_speed_mps,1e-7))
    if dist<20.:desired_v*=min(1.,dist/30.)
    desired_r=float(np.clip(wrap(obs.target_pose[2]-psi)/180.,
                            -np.deg2rad(.1),np.deg2rad(.1)))
    cost=(np.sum(((pred_xy-goal)/55.)**2)
          +4.*np.sum(((pred_v-desired_v)/.13)**2)
          +2.*(wrap(pred_heading-obs.target_pose[2])/np.deg2rad(8.))**2
          +3.*((pred_r-desired_r)/np.deg2rad(.13))**2)
    normal=np.array([-np.sin(obs.target_pose[2]),np.cos(obs.target_pose[2])])
    if float(err@normal)<0.:normal=-normal
    approach=float(pred_v@normal)
    allowed=max_near_speed_mps if obs.near_wall_gap_m<safety_near_gap_m else desired_speed_mps
    cost+=35.*(max(0.,approach-allowed)/.08)**2
    if obs.near_wall_gap_m<25.:
        cost+=30.*(max(0.,-obs.near_wall_gap_m)/5.)**2
    return float(cost)


class DistributedPotentialPlayer:
    """One player's private catalog plus public peer messages.

    No references to peer private models, other states or peers' catalogs.
    """
    def __init__(self, agent_id: int, bids: Sequence[ActionBid], observation: TugObservation,
                 baseline_acc, lookahead_s=90., desired_speed_mps=.17,
                 max_near_speed_mps=.095, safety_near_gap_m=45.):
        self.id=int(agent_id)
        self.bids=tuple(bids)
        self.feasible_ids=tuple(i for i,b in enumerate(self.bids) if b.feasible)
        if not self.feasible_ids:
            raise ValueError(f'Agent {agent_id} has no feasible action: equilibrium undefined')
        self.obs=observation
        self.baseline=np.asarray(baseline_acc,float)
        self.params=(lookahead_s,desired_speed_mps,max_near_speed_mps,safety_near_gap_m)
        self.selected=min(self.feasible_ids,key=lambda j:
            ((self.bids[j].action.thrust-observation.own_applied_thrust)/1e6)**2
            +(wrap(self.bids[j].action.azimuth-observation.own_applied_azimuth))**2)
        self.messages:dict[int,GameMessage]={}
        self.local_eval_count=0

    def message(self)->GameMessage:
        b=self.bids[self.selected]
        return GameMessage(self.id,self.selected,b.marginal_acc,b.local_cost)

    def receive(self, message:GameMessage):
        if message.agent_id != self.id:self.messages[message.agent_id]=message

    def _stage_cost(self, chosen: int)->float:
        if len(self.messages)!=3:raise ValueError('Missing peer announcements')
        b=self.bids[chosen]
        delta=np.asarray(b.marginal_acc,float).copy()
        local=b.local_cost
        for m in self.messages.values():
            delta+=np.asarray(m.marginal_acc,float)
            local+=m.local_cost
        self.local_eval_count+=1
        return ship_cost(self.obs,self.baseline,delta,*self.params)+local

    def best_response(self,tolerance=1e-7)->tuple[bool,float]:
        """Serialized *strict* improvement. Immutable catalog each game epoch."""
        current=self._stage_cost(self.selected)
        best=self.selected;best_cost=current
        for j in self.feasible_ids:
            if j==self.selected:continue
            val=self._stage_cost(j)
            if val<best_cost-tolerance:
                best=j;best_cost=val
        moved=(best!=self.selected)
        if moved:self.selected=best
        return moved,best_cost

    def exploitability(self)->float:
        current=self._stage_cost(self.selected)
        return max(0.,current-min(self._stage_cost(j) for j in self.feasible_ids))


class DistributedGameSession:
    """Only broadcasts public messages and schedules player turns.

    Selection of actions takes place INSIDE each DistributedPotentialPlayer.
    This object has no ship/tug plant, private candidate models, or force map.
    """
    def __init__(self, players:Sequence[DistributedPotentialPlayer]):
        if len(players)!=4 or {p.id for p in players}!={0,1,2,3}:
            raise ValueError('Exactly four players with IDs 0..3 are required')
        self.players=tuple(players)
        # Initialization can use one message broadcast for every player.
        for sender in self.players:
            for receiver in self.players:receiver.receive(sender.message())

    def solve(self,max_sweeps=30,tol=1e-7)->dict:
        if max_sweeps<1: raise ValueError('max_sweeps must be positive')
        moves=0; sweeps=0; history=[]
        for sweep in range(max_sweeps):
            sweeps+=1; changed=False
            for player in self.players:
                moved,_=player.best_response(tol)
                if moved:
                    moves+=1;changed=True
                    # Protocol conveys ONLY the currently chosen marginal message.
                    msg=player.message()
                    for peer in self.players:peer.receive(msg)
            history.append(self.players[0]._stage_cost(self.players[0].selected))
            if not changed:break
        residuals=[p.exploitability() for p in self.players]
        common=[p._stage_cost(p.selected) for p in self.players]
        if max(common)-min(common)>1e-5:
            raise RuntimeError('Payoff disagreement: public observations/models out of sync')
        return dict(equilibrium=bool(max(residuals)<=tol+1e-10),
                    max_unilateral_gain=float(max(residuals)),
                    per_agent_unilateral_gain=[float(v) for v in residuals],
                    sweeps=sweeps,moves=moves,potential=float(common[0]),
                    potential_by_sweep=history,
                    selected_action_ids=[int(p.selected) for p in self.players],
                    local_cost_evaluations=[int(p.local_eval_count) for p in self.players])
