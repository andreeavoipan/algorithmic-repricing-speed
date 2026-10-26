"""Uncertainty-aware belief rule: the agent knows its information about the rival price is stale.
Rival price ~ Normal(last_observed, belief_sd^2 * age); decisions use expected profit
(5-node Gauss-Hermite quadrature). With age 0 it reduces to the benchmark myopic rule."""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from .agents import MyopicBestResponseAgent
from .demand import profit_of_firm
_GH_X,_GH_W=np.polynomial.hermite_e.hermegauss(5); _GH_W=_GH_W/_GH_W.sum()

@dataclass
class UncertainBeliefAgent(MyopicBestResponseAgent):
    belief_sd: float = 0.01

    def _nodes(self,state):
        sd=self.belief_sd*np.sqrt(max(state.obs_age,0))
        if sd<=0: return np.array([state.last_observed_rival_price]),np.array([1.])
        r=np.clip(state.last_observed_rival_price+sd*_GH_X,self.price_floor,self.price_ceiling)
        return r,_GH_W

    def _exp_profit(self,p,nodes,w,costs):
        return sum(wi*self._profit_against_belief(p,ri,costs) for ri,wi in zip(nodes,w))

    def decision_details(self,state):
        if not state.can_recompute:
            return {"candidate_price":state.own_price,"expected_gain":0.0,"threshold":self.cost_threshold}
        costs=self._cost_vector(state); nodes,w=self._nodes(state)
        res=minimize_scalar(lambda p:-self._exp_profit(p,nodes,w,costs),bounds=(self.price_floor,self.price_ceiling),method="bounded")
        cand=float(res.x)
        gain=self._exp_profit(cand,nodes,w,costs)-self._exp_profit(state.own_price,nodes,w,costs)
        return {"candidate_price":cand,"expected_gain":gain,"threshold":self.cost_threshold}
