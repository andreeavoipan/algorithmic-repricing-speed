"""Tabular Q-learning repricing agent.

State = (markup_bin, gap_bin, staleness_bin), where markup = own_price -
own_cost, gap = own_price - last_observed_rival_price, and staleness = the
number of consecutive periods the firm has held its CURRENT price (0 right
after a reprice, growing for as long as it keeps holding). The class
implements the same ``decide(state)`` interface as the analytical agents,
allowing training and evaluation to use the exact same simulator and timing
convention.

Why staleness is part of the state (not an afterthought)
----------------------------------------------------------
Without it, HOLD is not a well-defined stationary (state, action) pair: the
reward earned by "holding" depends on how long the firm has already been
holding, because the market cost keeps drifting (AR(1)) while a held price
stays fixed. Two visits to the same (markup_bin, gap_bin) can have very
different expected rewards depending on whether the firm just repriced
(price still fresh) or has been holding for 50 periods (price now stale,
profit degraded) -- information the (markup_bin, gap_bin) state alone
cannot distinguish. Averaging reward over this hidden mixture makes
Q(state, HOLD) estimate a quantity that depends on each training run's own
path statistics (how long its holding streaks happened to run) rather than
on a single, well-defined stationary value -- so different random seeds can
converge to opposite HOLD-vs-MOVE rankings even with unlimited visits,
independent of the learning-rate schedule. Folding staleness into the state
restores the Markov property for the discretised problem and removes this
seed-dependent bistability at its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import AgentState


@dataclass
class QLearningAgent:
    firm_index: int
    rival_index: int
    costs: np.ndarray
    a: np.ndarray
    mu: float
    a0: float
    price_floor: float
    price_ceiling: float
    switching_cost: float = 0.0
    n_markup_bins: int = 13
    markup_range: tuple = (-0.3, 2.0)
    n_gap_bins: int = 15
    gap_range: float = 1.0
    n_staleness_bins: int = 1  # 1 = disabled (paper configuration); 4 = diagnostic variant
    alpha: float = 0.1
    gamma: float = 0.0
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 0
    training: bool = False
    use_decaying_lr: bool = True
    min_alpha: float = 0.01
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self):
        self.markup_grid = np.linspace(self.markup_range[0], self.markup_range[1], self.n_markup_bins)
        self.gap_grid = np.linspace(-self.gap_range, self.gap_range, self.n_gap_bins)
        self.HOLD = self.n_markup_bins
        self.n_actions = self.n_markup_bins + 1
        self.Q = np.zeros((self.n_markup_bins, self.n_gap_bins, self.n_staleness_bins, self.n_actions))
        self.visit_counts = np.zeros(
            (self.n_markup_bins, self.n_gap_bins, self.n_staleness_bins, self.n_actions), dtype=np.int64
        )
        self._last_transition = None
        self._decision_count = 0
        self._periods_held = 0  # periods since this firm last changed its price

    def markup_to_bin(self, markup: float) -> int:
        markup = np.clip(markup, self.markup_range[0], self.markup_range[1])
        return int(np.argmin(np.abs(self.markup_grid - markup)))

    def gap_to_bin(self, gap: float) -> int:
        gap = np.clip(gap, -self.gap_range, self.gap_range)
        return int(np.argmin(np.abs(self.gap_grid - gap)))

    def staleness_to_bin(self, periods_held: int) -> int:
        """Log-scale bucketing of periods-since-last-reprice: 0, 1, 2-3, 4-7,
        8-15, ..., capped at n_staleness_bins-1. Log-scale because the
        marginal information content of "held for 40 vs 41 periods" is far
        lower than "held for 0 vs 1 period" (the latter is exactly the
        fresh-price/stale-price distinction that matters for the reward)."""
        periods_held = max(int(periods_held), 0)
        if periods_held == 0:
            return 0
        b = int(np.log2(periods_held + 1))
        return int(min(b, self.n_staleness_bins - 1))

    def state_bins(self, own_price: float, own_cost: float, rival_price: float, periods_held: int = 0):
        return (
            self.markup_to_bin(own_price - own_cost),
            self.gap_to_bin(own_price - rival_price),
            self.staleness_to_bin(periods_held),
        )

    def choose_action(self, markup_bin: int, gap_bin: int, staleness_bin: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[markup_bin, gap_bin, staleness_bin]))

    def action_to_price(self, action: int, own_price: float, own_cost: float) -> float:
        if action == self.HOLD:
            return own_price
        target = own_cost + self.markup_grid[action]
        return float(np.clip(target, self.price_floor, self.price_ceiling))

    def decide(self, state: AgentState):
        if not state.can_recompute:
            # No technical opportunity to act this period: nothing was
            # decided, so no learning transition is recorded. The held
            # price still ages regardless of whether this firm got a
            # chance to act on it.
            self._periods_held += 1
            self._last_transition = None
            return False, state.own_price
        own_cost = (
            float(state.own_cost)
            if state.own_cost is not None
            else float(self.costs[self.firm_index])
        )
        mb, gb, sb = self.state_bins(
            state.own_price, own_cost, state.last_observed_rival_price, self._periods_held
        )
        if self.training and self.epsilon_decay_steps > 0:
            frac = min(self._decision_count / self.epsilon_decay_steps, 1.0)
            self.epsilon = max(self.epsilon_min, 1.0 - frac * (1.0 - self.epsilon_min))
        action = self.choose_action(mb, gb, sb, explore=self.training)
        self._decision_count += 1
        new_price = self.action_to_price(action, state.own_price, own_cost)
        reprice = abs(new_price - state.own_price) > 1e-9
        self._last_transition = (mb, gb, sb, action, state.own_price)
        # Advance staleness for whatever state comes next: reset on reprice,
        # otherwise the hold streak continues.
        self._periods_held = 0 if reprice else self._periods_held + 1
        return reprice, new_price

    def reward_from_realized_profit(self, realized_profit: float, switched: bool) -> float:
        return float(realized_profit - (self.switching_cost if switched else 0.0))

    def update(self, markup_bin, gap_bin, staleness_bin, action, r, next_markup_bin, next_gap_bin, next_staleness_bin):
        best_next = self.Q[next_markup_bin, next_gap_bin, next_staleness_bin].max()
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[markup_bin, gap_bin, staleness_bin, action]
        if self.use_decaying_lr:
            # Sample-average (Robbins-Monro, 1/N) step size instead of a fixed
            # alpha. With gamma=0, Q(s,a) is estimating a stationary mean
            # reward per (state, action) -- a per-cell multi-armed-bandit
            # problem. A FIXED step size never converges: it keeps a permanent
            # steady-state estimation variance of order alpha/(2-alpha) times
            # the reward variance, however many visits occur. Decaying the
            # step size as 1/N(s,a) removes this residual variance as visits
            # accumulate, while min_alpha keeps early updates from being too
            # slow to start learning. (This alone fixed the large-switching-
            # cost instability; it does NOT by itself fix the small-cost
            # instability, which comes from the staleness confound described
            # in the module docstring -- that is why staleness is now part
            # of the state key being updated here.)
            self.visit_counts[markup_bin, gap_bin, staleness_bin, action] += 1
            n = self.visit_counts[markup_bin, gap_bin, staleness_bin, action]
            step = max(self.min_alpha, 1.0 / n)
        else:
            step = self.alpha
        self.Q[markup_bin, gap_bin, staleness_bin, action] += step * td_error

    def learn(self, reward: float, next_state: AgentState):
        if not self.training or self._last_transition is None:
            return
        mb, gb, sb, action, _ = self._last_transition
        own_cost = (
            float(next_state.own_cost)
            if next_state.own_cost is not None
            else float(self.costs[self.firm_index])
        )
        # self._periods_held was already advanced inside decide() to reflect
        # the staleness that will be in effect at the START of next period,
        # i.e. exactly the staleness component of next_state.
        nmb, ngb = self.state_bins(
            next_state.own_price, own_cost, next_state.last_observed_rival_price
        )[:2]
        nsb = self.staleness_to_bin(self._periods_held)
        self.update(mb, gb, sb, action, reward, nmb, ngb, nsb)

    # Backwards-compatible helper retained for simple unit tests / notebooks.
    def reward(self, new_own_price: float, old_own_price: float, rival_price: float) -> float:
        from .demand import profit_of_firm

        rival_prices = np.zeros(len(self.a))
        rival_prices[self.rival_index] = rival_price
        profit = profit_of_firm(
            new_own_price, self.firm_index, rival_prices, self.costs, self.a, self.mu, self.a0
        )
        return self.reward_from_realized_profit(
            profit, abs(new_own_price - old_own_price) > 1e-9
        )
