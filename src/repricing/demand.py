"""
Logit demand system, as used in Calvano et al. (2020) and related
algorithmic-collusion literature.

Each firm i sets price p_i. Demand for firm i is:

    q_i(p) = exp((a_i - p_i) / mu) / [ sum_j exp((a_j - p_j) / mu) + exp(a0 / mu) ]

a0 is the "outside option" (the consumer buying nothing / from a fringe).
mu controls horizontal differentiation: mu -> 0 recovers near-Bertrand
competition (very price sensitive), larger mu means products are more
differentiated and price competition is softer.
"""

from __future__ import annotations

import numpy as np


def logit_quantities(
    prices: np.ndarray,
    a: np.ndarray,
    mu: float,
    a0: float = 0.0,
) -> np.ndarray:
    """
    Compute logit demand quantities for N firms.

    Parameters
    ----------
    prices : array of shape (N,)
    a : array of shape (N,), quality/vertical differentiation index per firm
    mu : horizontal differentiation / price sensitivity parameter (mu > 0)
    a0 : outside-option utility (higher a0 -> more consumers buy nothing)

    Returns
    -------
    quantities : array of shape (N,), each in (0, 1), summing to < 1
    """
    prices = np.asarray(prices, dtype=float)
    a = np.asarray(a, dtype=float)

    utilities = (a - prices) / mu
    exp_u = np.exp(utilities)
    denom = exp_u.sum() + np.exp(a0 / mu)
    return exp_u / denom


def profits(
    prices: np.ndarray,
    costs: np.ndarray,
    a: np.ndarray,
    mu: float,
    a0: float = 0.0,
) -> np.ndarray:
    """Per-firm profit = (price - marginal cost) * quantity."""
    q = logit_quantities(prices, a, mu, a0)
    return (np.asarray(prices, dtype=float) - np.asarray(costs, dtype=float)) * q


def profit_of_firm(
    own_price: float,
    firm_index: int,
    rival_prices: np.ndarray,
    costs: np.ndarray,
    a: np.ndarray,
    mu: float,
    a0: float = 0.0,
) -> float:
    """
    Profit of a single firm as a function of its own price, holding all
    rival prices fixed. Used by best-response agents to optimize their
    own price via a 1-D search.
    """
    prices = np.array(rival_prices, dtype=float, copy=True)
    prices[firm_index] = own_price
    return float(profits(prices, costs, a, mu, a0)[firm_index])
