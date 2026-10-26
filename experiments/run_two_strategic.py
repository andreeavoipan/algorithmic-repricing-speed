"""Core experiment: technical opportunity versus realised repricing rate.

The primary metric is now reprices / post-burn-in periods.  The reciprocal
mean inter-event interval is still saved as a secondary/legacy metric.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import MyopicBestResponseAgent
from repricing.simulation import DuopolyConfig, run_duopoly

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
A = np.array([2.0, 2.0]); MU = 0.25; A0 = 0.0; FLOOR, CEIL = 1.0, 3.0
TECH_FREQS = np.linspace(0.05, 1.0, 12)
COST_THRESHOLDS = [0.0, 0.0005, 0.002, 0.008]
COST_SHOCK_STD = 0.01
N_SEEDS = 40
T = 5000
BURN_IN = 500


def make_agents(cost_threshold: float):
    common = dict(costs=np.array([1.0, 1.0]), a=A, mu=MU, a0=A0,
                  price_floor=FLOOR, price_ceiling=CEIL)
    return (
        MyopicBestResponseAgent(firm_index=0, rival_index=1,
                                cost_threshold=cost_threshold, **common),
        MyopicBestResponseAgent(firm_index=1, rival_index=0,
                                cost_threshold=0.0, **{**common, "costs": np.array([1.0, 1.0])}),
    )


def main():
    rows = []
    for threshold in COST_THRESHOLDS:
        for tech in TECH_FREQS:
            rates, legacy, conditional = [], [], []
            for seed in range(N_SEEDS):
                a0, a1 = make_agents(threshold)
                result = run_duopoly(a0, a1, DuopolyConfig(
                    T=T, burn_in=BURN_IN, tech_freq_0=tech, tech_freq_1=1.0,
                    obs_freq_0=1.0, obs_freq_1=1.0, seed=seed,
                    cost_shock_std=COST_SHOCK_STD, cost_shock_std_1=COST_SHOCK_STD,
                    cost_rho=0.98,
                ))
                rates.append(result.effective_speed(0))
                legacy.append(result.inter_reprice_speed(0))
                conditional.append(result.conditional_repricing_rate(0))
            rows.append({
                "cost_threshold": threshold, "tech_freq": tech,
                "reprice_rate_mean": np.mean(rates), "reprice_rate_std": np.std(rates),
                "legacy_inter_event_speed_mean": np.mean(legacy),
                "conditional_repricing_rate_mean": np.mean(conditional),
            })
            print(f"threshold={threshold:.4f} tech={tech:.2f} rate={np.mean(rates):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "two_strategic_grid.csv"), index=False)
    plt.figure(figsize=(7, 5))
    for threshold in COST_THRESHOLDS:
        sub = df[df.cost_threshold == threshold]
        ci = 1.96 * sub.reprice_rate_std / np.sqrt(N_SEEDS)
        plt.errorbar(sub.tech_freq, sub.reprice_rate_mean, yerr=ci,
                     marker="o", capsize=3, label=f"adjustment threshold = {threshold}")
    plt.plot(TECH_FREQS, TECH_FREQS, "k--", alpha=0.4, label="technical opportunity rate")
    plt.xlabel("Technical recomputation probability")
    plt.ylabel("Realised repricing rate (reprices / period)")
    plt.title("Technical capacity vs realised repricing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "two_strategic_effective_speed.png"), dpi=150)


if __name__ == "__main__":
    main()
