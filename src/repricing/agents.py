"""Repricing agents used by the simulation.

The simulator deliberately separates three objects:

* technical opportunity: whether the agent can recompute this period;
* information: the last rival price the agent has actually observed;
* economic action: whether the candidate price is worth implementing.

Importantly, an agent with imperfect observability never receives an oracle copy of
its rival's current price. Realised market outcomes are computed by the simulator,
not by the decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .demand import profit_of_firm


@dataclass
class AgentState:
    """Information available to an agent when it makes a pricing decision."""

    own_price: float
    last_observed_rival_price: float
    can_recompute: bool
    own_cost: Optional[float] = None
    market_costs: Optional[np.ndarray] = None
    obs_age: int = 0  # periods since the rival price was last observed (0 = observed this period)


@dataclass
class FixedRuleAgent:
    """Simple non-strategic repricing rule used as a baseline."""

    firm_index: int
    price_floor: float
    price_ceiling: float
    undercut: float = 0.02
    markup_step: float = 0.01
    tolerance: float = 0.01

    def decide(self, state: AgentState):
        if not state.can_recompute:
            return False, state.own_price

        rival = state.last_observed_rival_price
        gap = rival - state.own_price
        if gap < -self.tolerance:
            return True, max(self.price_floor, rival - self.undercut)
        if gap > self.tolerance:
            return True, min(self.price_ceiling, state.own_price + self.markup_step)
        return False, state.own_price


@dataclass
class MyopicBestResponseAgent:
    """Myopic best-response agent with an implementation/adjustment threshold.

    ``cost_threshold`` is intentionally interpreted as an exogenous repricing
    friction (menu/adjustment/implementation cost), not as an endogenous strategic
    cost. Both the proposed price and the gain used to decide whether to implement
    it are evaluated using the agent's *belief* about the rival price.
    """

    firm_index: int
    rival_index: int
    costs: np.ndarray
    a: np.ndarray
    mu: float
    a0: float
    price_floor: float
    price_ceiling: float
    cost_threshold: float = 0.0

    def _cost_vector(self, state: Optional[AgentState] = None) -> np.ndarray:
        if state is not None and state.market_costs is not None:
            return np.asarray(state.market_costs, dtype=float)
        return np.asarray(self.costs, dtype=float)

    def _best_response(self, rival_price: float, costs: Optional[np.ndarray] = None) -> float:
        use_costs = np.asarray(self.costs if costs is None else costs, dtype=float)
        rival_prices = np.zeros(len(self.a))
        rival_prices[self.rival_index] = rival_price
        result = minimize_scalar(
            lambda p: -profit_of_firm(
                p, self.firm_index, rival_prices, use_costs, self.a, self.mu, self.a0
            ),
            bounds=(self.price_floor, self.price_ceiling),
            method="bounded",
        )
        return float(result.x)

    def _profit_against_belief(self, own_price: float, rival_price: float, costs: np.ndarray) -> float:
        rival_prices = np.zeros(len(self.a))
        rival_prices[self.rival_index] = rival_price
        return profit_of_firm(
            own_price, self.firm_index, rival_prices, costs, self.a, self.mu, self.a0
        )

    def decision_details(self, state: AgentState) -> dict:
        if not state.can_recompute:
            return {
                "candidate_price": state.own_price,
                "expected_gain": 0.0,
                "threshold": self.cost_threshold,
            }
        costs = self._cost_vector(state)
        believed_rival = state.last_observed_rival_price
        current_profit = self._profit_against_belief(state.own_price, believed_rival, costs)
        candidate = self._best_response(believed_rival, costs)
        candidate_profit = self._profit_against_belief(candidate, believed_rival, costs)
        return {
            "candidate_price": candidate,
            "expected_gain": candidate_profit - current_profit,
            "threshold": self.cost_threshold,
        }

    def decide(self, state: AgentState):
        if not state.can_recompute:
            return False, state.own_price
        details = self.decision_details(state)
        if details["expected_gain"] > self.cost_threshold:
            return True, float(details["candidate_price"])
        return False, state.own_price


@dataclass
class AnticipatoryBestResponseAgent(MyopicBestResponseAgent):
    """Best-response agent that internalises an anticipated rival reaction.

    The agent first computes a myopic candidate using its observed rival price.
    If it changes price, it believes the rival reacts with probability
    ``reaction_probability`` by choosing its own one-period best response to the
    candidate. The implementation decision therefore depends on the *expected*
    post-reaction payoff rather than on an exogenous label of "strategic cost".

    ``cost_threshold`` remains available as a separate implementation friction, so
    endogenous strategic anticipation and ordinary adjustment costs can be varied
    independently.
    """

    reaction_probability: float = 1.0

    def _rival_best_response(self, own_candidate: float, costs: np.ndarray) -> float:
        rival_prices = np.zeros(len(self.a))
        rival_prices[self.firm_index] = own_candidate
        result = minimize_scalar(
            lambda p: -profit_of_firm(
                p, self.rival_index, rival_prices, costs, self.a, self.mu, self.a0
            ),
            bounds=(self.price_floor, self.price_ceiling),
            method="bounded",
        )
        return float(result.x)

    def decision_details(self, state: AgentState) -> dict:
        if not state.can_recompute:
            return {
                "candidate_price": state.own_price,
                "expected_gain": 0.0,
                "threshold": self.cost_threshold,
                "anticipated_rival_price": state.last_observed_rival_price,
            }

        costs = self._cost_vector(state)
        believed_rival = state.last_observed_rival_price
        candidate = self._best_response(believed_rival, costs)

        current_profit = self._profit_against_belief(state.own_price, believed_rival, costs)
        no_response_profit = self._profit_against_belief(candidate, believed_rival, costs)

        reacted_rival = self._rival_best_response(candidate, costs)
        response_profit = self._profit_against_belief(candidate, reacted_rival, costs)
        rho = float(np.clip(self.reaction_probability, 0.0, 1.0))
        expected_candidate_profit = (1.0 - rho) * no_response_profit + rho * response_profit

        return {
            "candidate_price": candidate,
            "expected_gain": expected_candidate_profit - current_profit,
            "threshold": self.cost_threshold,
            "anticipated_rival_price": reacted_rival,
            "no_response_profit": no_response_profit,
            "response_profit": response_profit,
        }
