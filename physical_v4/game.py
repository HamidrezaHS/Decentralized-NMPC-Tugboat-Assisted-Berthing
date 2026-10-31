"""Serializable, finite identical-interest game with an exact-potential certificate.

The stage potential is defined from a PUBLIC ship state and the sum of the
predicted force signatures broadcast by each independently acting tug.
Every player computes its own bids using its own tug mode and station geometry.
Serialized strict improvement converges on a pure equilibrium for a fixed
finite catalog. This does not imply finite-horizon optimality or global safety.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from physical_v4.plant import rot,wrap


def stopping_distance_shortfall(closing_speed_mps:float, free_gap_m:float,
                                stopping_horizon_s:float=65., reserve_m:float=6.) -> float:
    """Nonnegative shortage of braking room in metres.

    A safe, slower approach must NEVER accrue a penalty simply because
    it uses less than the available distance. This fixes a signed-square
    objective error in v5.1. The horizon is a heuristic, not a proven
    physical stopping distance or safety certificate.
    """
    return max(0., max(0.,float(closing_speed_mps))*stopping_horizon_s
               - max(0.,float(free_gap_m)-reserve_m))

@dataclass(frozen=True)
class Action:
    fraction:float
    offset_deg:float

class TugPlayer:
    def __init__(self,i,side,max_force,anchor_x,ship_length=320.):
        self.i,self.side,self.max_force=int(i),int(side),float(max_force)
        self.x=float(anchor_x)
        self.last=Action(0.,0.)
        self.selected=0
        self.messages={}
        self.bids=[]

    def offers(self,ship,own,remaining):
        """Only public ship and own tug/anchor parameters are used."""
        # Candidate contribution acts at this agent's physical ship attachment.
        fractions=([0.,.17,.35,.55,.78,1.] if self.i<2 else [0.,.2,.4,.65,.9])
        offsets=([-15.,0.,15.] if self.i<2 else [-12.,0.,12.])
        actions=[Action(float(f),float(o)) for f in fractions for o in offsets]
        actions.append(self.last)
        seen=set();self.bids=[]
        for a in actions:
            key=(a.fraction,a.offset_deg)
            if key in seen:continue
            seen.add(key)
            # Effective force exerted on the SHIP, not the tug hull heading.
            # Push tug faces INBOARD with positive thrust; tow tug ALSO faces
            # INBOARD, but uses negative (astern) thrust, pulling the ship
            # OUTBOARD through its bow-attached rope.
            angle=(-self.side*np.pi/2 if self.i<2 else self.side*np.pi/2)+np.deg2rad(a.offset_deg)
            direction=np.array([np.cos(angle),np.sin(angle)])
            f=a.fraction*self.max_force*direction
            # Tangential thrust partly supports station-keeping; reduce presumed
            # ship transmitted tangential component to avoid exaggeration.
            # Push reacts opposite to its outward-facing tug drive force? A
            # push heading INWARD applies force INWARD to ship, so same sign.
            # Tow ASTERN propulsion pulls the ship OUTWARD via a bow rope.
            f[0]*=.65
            yaw=self.x*f[1]
            # agent-local costs include effort and reluctance to change commands
            local=(.08*a.fraction*a.fraction+
                   .030*(a.fraction-self.last.fraction)**2+
                   .010*(a.offset_deg/20.)**2+
                   .015*((a.offset_deg-self.last.offset_deg)/30.)**2)
            # Some directional heading corrections via offset are allowed.
            self.bids.append((a,np.array([f[0],f[1],yaw]),float(local)))
        # Warm-start nearest previous executed command.
        self.selected=min(range(len(self.bids)),key=lambda j:
                 (self.bids[j][0].fraction-self.last.fraction)**2+
                 ((self.bids[j][0].offset_deg-self.last.offset_deg)/30.)**2)
        return self.bids

    def message(self):
        # Only the currently selected predicted ship-force signature and cost
        # are broadcast, NEVER own state, future candidate catalog or controller.
        a,f,c=self.bids[self.selected]
        return (self.i,self.selected,tuple(map(float,f)),float(c))

    def receive_message(self,message):
        sender,selection,force,cost=message
        if sender!=self.i:self.messages[int(sender)]=(int(selection),np.array(force,float),float(cost))

    def _local_score(self,public,choice):
        if len(self.messages)!=3:raise ValueError('Missing messages from other players')
        _,force,cost=self.bids[choice]
        total=force.copy();total_cost=cost
        for _,peer_force,peer_cost in self.messages.values():
            total+=peer_force;total_cost+=peer_cost
        return public.potential_for(total,total_cost)

    def best_response(self,public,tol=1e-9):
        current=self._local_score(public,self.selected)
        best=self.selected;best_score=current
        for j in range(len(self.bids)):
            val=self._local_score(public,j)
            if val<best_score-tol:best_score=val;best=j
        changed=best!=self.selected
        self.selected=best
        return changed

    def equilibrium_gap(self,public):
        current=self._local_score(public,self.selected)
        best=min(self._local_score(public,j) for j in range(len(self.bids)))
        return max(0.,current-best)

class ExactPotentialGame:
    def __init__(self,ship,goal,ship_model,players,tug_states,accel_scale=1.,horizon=75.):
        self.ship=np.asarray(ship,float)
        self.goal=np.asarray(goal,float)
        self.ship_model=ship_model
        self.players=list(players)
        self.H=float(horizon)
        self.scale=float(accel_scale)
        # Precompute ship's hydro acceleration AND mass operator. Game player
        # messages contain force signatures only, no private tug states.
        ship_model.external_tau=np.zeros(3)
        self.A,self.C=ship_model.ship_matrices(ship[3],ship[4],ship[5],ship[2])
        self.A3=self.A[:3,:3]
        self.A3inv=np.linalg.inv(self.A3)
        self.d0=np.linalg.solve(self.A,self.C.flatten())
        remain=np.linalg.norm(goal[:2]-ship[:2])
        for p in self.players:p.offers(ship,tug_states[p.i],remain)
        self.start_cost=None

    def potential_for(self,force,regularizer):
        """PUBLIC stage cost based only on shared ship state and announced totals."""
        body_delta=self.A3inv@np.asarray(force)
        body_acc=self.d0[:3]+self.scale*body_delta
        psi=self.ship[2];u,v,r=self.ship[3:]
        world_vel=rot(psi)@np.array([u,v])
        # earth acceleration includes Coriolis term from heading rotation
        a_world=rot(psi)@(body_acc[:2]+np.array([-r*v,r*u]))
        H=self.H
        pred_v=world_vel+H*a_world
        pred_p=self.ship[:2]+H*world_vel+.5*H*H*a_world
        pred_r=r+H*body_acc[2]
        pred_psi=wrap(psi+H*r+.5*H*H*body_acc[2])
        e_p=np.linalg.norm(self.goal[:2]-self.ship[:2])
        max_v=.080 if e_p>45 else (.065 if e_p>16 else .040)
        desired_v=(self.goal[:2]-self.ship[:2]) / max(e_p,1e-5)*max_v if e_p>1e-8 else np.zeros(2)
        heading_err=wrap(pred_psi-self.goal[2])
        # Strong target-progress signal at long distance, and terminal speed/yaw.
        cost=(2.8*np.sum(((pred_p-self.goal[:2])/75.)**2)
              +1.7*np.sum(((pred_v-desired_v)/.15)**2)
              +1.8*(heading_err/np.deg2rad(16.))**2
              +1.4*(pred_r/np.deg2rad(.09))**2
              +regularizer)
        # Physical quay feasibility: ship bounding box must not sweep through
        # quay even when centre-of-mass remains far from destination.
        quay_y=self.goal[1]-29.-6.
        hull_low=pred_p[1]-160*abs(np.sin(pred_psi))-29*abs(np.cos(pred_psi))
        gap_pred=hull_low-quay_y
        if gap_pred<7.:
            cost+=300.*(max(0.,7.-gap_pred)/7.)**2
        # A vessel can have zero centre-position error yet strike the berth
        # because it retains lateral velocity. Penalize predicted normal speed
        # in the safety-critical approach region; this can make outward tow
        # actions attractive to the individual game players.
        current_gap=self.ship[1]-160*abs(np.sin(psi))-29*abs(np.cos(psi))-quay_y
        if current_gap<52. or self.ship[1]<self.goal[1]+62.:
            speed_limit=.052 if current_gap<25 else .080
            closing=max(0.,-pred_v[1])
            cost+=170.*(max(0.,closing-speed_limit)/.10)**2
            # Protect a stopping horizon from contact even if the heading
            # changes while the vessel continues drifting sideways.
            # One-sided stopping constraint: only insufficient stopping room is
            # penalized. Squaring a signed residual also penalizes slow, safe
            # approaches, causing the stage game to prefer unsafe speed.
            stopping_deficit=stopping_distance_shortfall(-pred_v[1], current_gap)
            cost+=80.*(stopping_deficit/20.)**2
        # Near-quay and near-goal braking: avoid crossing the terminal location
        # at high speed. Actual collision is evaluated separately in the plant.
        if e_p<55:
            cost+=2.5*(max(0.,np.linalg.norm(pred_v)-max(.04,e_p/250.))/.09)**2
        return float(cost)

    def potential(self,indices):
        force=np.zeros(3);regularizer=0.
        for p,j in zip(self.players,indices):
            _,f,c=p.bids[j]
            force+=f;regularizer+=c
        return self.potential_for(force,regularizer)

    def solve(self,max_sweeps=50,tol=1e-9):
        ids=[p.selected for p in self.players]
        before=self.potential(ids);history=[before];moves=0
        for s in range(max_sweeps):
            changed=False
            for i,p in enumerate(self.players):
                cost=self.potential(ids);best=ids[i];bestcost=cost
                for j in range(len(p.bids)):
                    if j==ids[i]:continue
                    test=list(ids);test[i]=j
                    q=self.potential(test)
                    if q<bestcost-tol:
                        bestcost=q;best=j
                if best!=ids[i]:
                    ids[i]=best;p.selected=best;moves+=1;changed=True
            history.append(self.potential(ids))
            if not changed:break
        gains=[]
        current=self.potential(ids)
        for i,p in enumerate(self.players):
            lowest=min(self.potential([*ids[:i],j,*ids[i+1:]]) for j in range(len(p.bids)))
            gains.append(max(0.,current-lowest))
        for p,j in zip(self.players,ids):p.selected=j;p.last=p.bids[j][0]
        return {'actions':[p.bids[j][0] for p,j in zip(self.players,ids)],
                'equilibrium':bool(max(gains)<tol+1e-9),'unilateral_gains':gains,
                'potential':current,'potential_before':before,'history':history,
                'sweeps':len(history)-1,'moves':moves}

class DistributedEquilibrium:
    """Message relay plus deterministic serial turn schedule; NOT an optimizer.

    Players themselves evaluate their local candidate catalogs; the scheduler
    never inspects bids, private states, or individual candidate scores.
    """
    def __init__(self,players,public):
        if len(players)!=4:raise ValueError('Expected four autonomous controllers')
        self.players=tuple(players)
        self.public=public
        for sender in self.players:
            self._broadcast(sender.message())

    def _broadcast(self,msg):
        for p in self.players:p.receive_message(msg)

    def solve(self,max_sweeps=50,tol=1e-9):
        self.public.players=[]  # Public evaluator never holds private bids.
        initial=self.players[0]._local_score(self.public,self.players[0].selected)
        history=[initial];moves=0
        for _ in range(max_sweeps):
            changed=False
            for p in self.players:
                moved=p.best_response(self.public,tol)
                if moved:
                    moves+=1;changed=True
                    self._broadcast(p.message())
            history.append(self.players[0]._local_score(self.public,self.players[0].selected))
            if not changed:break
        gaps=[p.equilibrium_gap(self.public) for p in self.players]
        score=self.players[0]._local_score(self.public,self.players[0].selected)
        for p in self.players:p.last=p.bids[p.selected][0]
        return dict(actions=[p.last for p in self.players],
            equilibrium=bool(max(gaps)<=tol+1e-9),unilateral_gains=gaps,
            potential=score,potential_before=initial,history=history,
            sweeps=len(history)-1,moves=moves)

class TugHelmsman:
    """Local low-level heading control from own heading/rate and public ship heading.

    The equilibrium game sets the thrust fraction and desired offset. The
    azimuth command is a bounded feedback steering command, not an external
    force allocator, teleportation or direct change of vessel state.
    """
    def __init__(self,i,side):self.i=int(i);self.side=int(side)
    def steer(self,ship,tug,action):
        outward=ship[2]+self.side*np.pi/2
        # Both bow headings remain directed inward. Tow propulsion is astern.
        base=outward+np.pi
        desired=wrap(base+np.deg2rad(action.offset_deg))
        angle_error=wrap(desired-tug[2]);yaw_rate_rel=tug[5]-ship[5]
        # Stern azimuth transverse force creates negative moment for +delta.
        delta=-2.0*angle_error+5.5*yaw_rate_rel
        # Steering yaw sign reverses when the aft thruster operates astern.
        if self.i>=2:delta=-delta
        return float(np.clip(delta,-np.deg2rad(35),np.deg2rad(35)))

    def yaw_moment(self,ship,tug,action):
        base=ship[2]-self.side*np.pi/2
        desired=wrap(base+np.deg2rad(action.offset_deg))
        error=wrap(desired-tug[2]);rate=tug[5]-ship[5]
        return float(np.clip(8e6*error-18e6*rate,-4e6,4e6))
