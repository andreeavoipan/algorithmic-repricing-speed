"""Unified duopoly repricing simulator.

All analytical and learning agents use the same period timing:

1. the market cost state evolves;
2. each firm may observe the rival and may receive a technical recomputation opportunity;
3. all agents decide from start-of-period information;
4. decisions are applied simultaneously;
5. realised profits are computed from the true market state;
6. learning agents receive the realised reward and post-action next state.

The market, rather than an agent object, is the source of truth for prices and costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .agents import AgentState
from .demand import profits


@dataclass
class MarketState:
    prices: np.ndarray
    costs: np.ndarray


@dataclass
class DuopolyConfig:
    T: int = 2000
    burn_in: int = 0
    tech_freq_0: float = 1.0
    tech_freq_1: float = 1.0
    obs_freq_0: float = 1.0
    obs_freq_1: float = 1.0
    init_price_0: float = 1.5
    init_price_1: float = 1.5
    init_cost_0: float = 1.0
    init_cost_1: float = 1.0
    seed: int = 0
    cost_shock_std: float = 0.0
    cost_shock_std_1: float = 0.0
    cost_bounds: tuple = (0.5, 1.5)
    cost_mean: tuple = (1.0, 1.0)
    cost_rho: float = 0.98
    record_decision_log: bool = True


@dataclass
class DuopolyResult:
    price_history: np.ndarray
    cost_history: np.ndarray
    reprice_events: list
    profit_history: np.ndarray
    decision_log: list
    burn_in: int = 0

    @property
    def T(self) -> int:
        return int(self.price_history.shape[0])

    def _active_events(self, firm_index: int):
        return [t for (t, f) in self.reprice_events if f == firm_index and t >= self.burn_in]

    def effective_speed(self, firm_index: int) -> float:
        """Primary speed metric: reprices per post-burn-in period."""
        active_periods = max(self.T - self.burn_in, 1)
        return len(self._active_events(firm_index)) / active_periods

    def inter_reprice_speed(self, firm_index: int) -> float:
        """Legacy metric: reciprocal mean interval between reprices."""
        times = self._active_events(firm_index)
        if len(times) < 2:
            return 0.0
        return float(1.0 / np.diff(times).mean())

    def mean_reprice_interval(self, firm_index: int) -> float:
        times = self._active_events(firm_index)
        if len(times) < 2:
            return float("nan")
        return float(np.diff(times).mean())

    def median_reprice_interval(self, firm_index: int) -> float:
        times = self._active_events(firm_index)
        if len(times) < 2:
            return float("nan")
        return float(np.median(np.diff(times)))

    def conditional_repricing_rate(self, firm_index: int) -> float:
        rows = [
            r for r in self.decision_log
            if r["firm"] == firm_index and r["period"] >= self.burn_in and r["can_recompute"]
        ]
        if not rows:
            return 0.0
        return float(np.mean([bool(r["reprice"]) for r in rows]))


def _evolve_cost(current: float, mean: float, rho: float, std: float, bounds, rng) -> float:
    if std <= 0:
        return current
    value = mean + rho * (current - mean) + rng.normal(0, std)
    return float(np.clip(value, *bounds))


def _decision_details(agent, state: AgentState) -> dict:
    if hasattr(agent, "decision_details"):
        try:
            return dict(agent.decision_details(state))
        except Exception:
            pass
    return {}


def run_duopoly(agent0, agent1, config: DuopolyConfig) -> DuopolyResult:
    if config.burn_in < 0 or config.burn_in >= config.T:
        raise ValueError("burn_in must satisfy 0 <= burn_in < T")

    rng = np.random.default_rng(config.seed)
    market = MarketState(
        prices=np.array([config.init_price_0, config.init_price_1], dtype=float),
        costs=np.array([config.init_cost_0, config.init_cost_1], dtype=float),
    )
    price_history = np.zeros((config.T, 2))
    cost_history = np.zeros((config.T, 2))
    profit_history = np.zeros((config.T, 2))
    reprice_events = []
    decision_log = []
    last_observed = np.array([market.prices[1], market.prices[0]], dtype=float)
    last_obs_t = np.array([0, 0])

    # Market parameters are common in this project. Keep them out of market-state
    # mutation but obtain them from an agent that defines the demand model.
    model_agent = agent0 if hasattr(agent0, "a") else agent1
    a = np.asarray(model_agent.a, dtype=float)
    mu = float(model_agent.mu)
    a0 = float(model_agent.a0)

    for t in range(config.T):
        market.costs[0] = _evolve_cost(
            market.costs[0], config.cost_mean[0], config.cost_rho,
            config.cost_shock_std, config.cost_bounds, rng,
        )
        market.costs[1] = _evolve_cost(
            market.costs[1], config.cost_mean[1], config.cost_rho,
            config.cost_shock_std_1, config.cost_bounds, rng,
        )

        start_prices = market.prices.copy()

        can0 = rng.random() < config.tech_freq_0
        can1 = rng.random() < config.tech_freq_1
        observed0 = rng.random() < config.obs_freq_0
        observed1 = rng.random() < config.obs_freq_1
        if observed0:
            last_observed[0] = start_prices[1]
            last_obs_t[0] = t
        if observed1:
            last_observed[1] = start_prices[0]
            last_obs_t[1] = t

        state0 = AgentState(
            own_price=start_prices[0],
            last_observed_rival_price=last_observed[0],
            can_recompute=can0,
            own_cost=market.costs[0],
            market_costs=market.costs.copy(),
            obs_age=int(t - last_obs_t[0]),
        )
        state1 = AgentState(
            own_price=start_prices[1],
            last_observed_rival_price=last_observed[1],
            can_recompute=can1,
            own_cost=market.costs[1],
            market_costs=market.costs.copy(),
            obs_age=int(t - last_obs_t[1]),
        )

        details0 = _decision_details(agent0, state0)
        details1 = _decision_details(agent1, state1)
        reprice0, new0 = agent0.decide(state0)
        reprice1, new1 = agent1.decide(state1)

        if reprice0:
            market.prices[0] = new0
            reprice_events.append((t, 0))
        if reprice1:
            market.prices[1] = new1
            reprice_events.append((t, 1))

        realized = profits(market.prices, market.costs, a, mu, a0)
        price_history[t] = market.prices
        cost_history[t] = market.costs
        profit_history[t] = realized

        if config.record_decision_log:
            decision_log.extend([
            {
                "period": t, "firm": 0, "can_recompute": bool(can0),
                "observed_rival": bool(observed0),
                "believed_rival_price": float(last_observed[0]),
                "own_cost": float(market.costs[0]), "old_price": float(start_prices[0]),
                "candidate_price": float(details0.get("candidate_price", new0)),
                "expected_gain": details0.get("expected_gain", np.nan),
                "threshold": details0.get("threshold", np.nan),
                "reprice": bool(reprice0), "new_price": float(market.prices[0]),
                "realized_profit": float(realized[0]),
            },
            {
                "period": t, "firm": 1, "can_recompute": bool(can1),
                "observed_rival": bool(observed1),
                "believed_rival_price": float(last_observed[1]),
                "own_cost": float(market.costs[1]), "old_price": float(start_prices[1]),
                "candidate_price": float(details1.get("candidate_price", new1)),
                "expected_gain": details1.get("expected_gain", np.nan),
                "threshold": details1.get("threshold", np.nan),
                "reprice": bool(reprice1), "new_price": float(market.prices[1]),
                "realized_profit": float(realized[1]),
            },
            ])

        # Post-action state for a same-timing Q-learning transition. It uses the
        # observation already available this period; no future information leaks in.
        next0 = AgentState(
            own_price=market.prices[0], last_observed_rival_price=last_observed[0],
            can_recompute=True, own_cost=market.costs[0], market_costs=market.costs.copy(),
        )
        next1 = AgentState(
            own_price=market.prices[1], last_observed_rival_price=last_observed[1],
            can_recompute=True, own_cost=market.costs[1], market_costs=market.costs.copy(),
        )
        if hasattr(agent0, "learn"):
            reward0 = (
                agent0.reward_from_realized_profit(realized[0], reprice0)
                if hasattr(agent0, "reward_from_realized_profit") else float(realized[0])
            )
            agent0.learn(reward0, next0)
        if hasattr(agent1, "learn"):
            reward1 = (
                agent1.reward_from_realized_profit(realized[1], reprice1)
                if hasattr(agent1, "reward_from_realized_profit") else float(realized[1])
            )
            agent1.learn(reward1, next1)

    return DuopolyResult(
        price_history=price_history,
        cost_history=cost_history,
        reprice_events=reprice_events,
        profit_history=profit_history,
        decision_log=decision_log,
        burn_in=config.burn_in,
    )
