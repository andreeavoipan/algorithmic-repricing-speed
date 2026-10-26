"""Simple baseline with a fixed-rule rival.

Kept mainly as a sanity check.  The paper-facing core experiment is
``run_two_strategic.py``.  ``cost_threshold`` is described here as an adjustment
threshold, not as an endogenous strategic cost.
"""
from __future__ import annotations
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
from repricing.agents import FixedRuleAgent,MyopicBestResponseAgent
from repricing.simulation import DuopolyConfig,run_duopoly
RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results"); os.makedirs(RESULTS_DIR,exist_ok=True)
A=np.array([2.,2.]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
TECH_FREQS=np.linspace(.05,1.,12); THRESHOLDS=[0.,.002,.006,.015]; N_SEEDS=15; T=3000; BURN_IN=300

def main():
    rows=[]
    for th in THRESHOLDS:
        for tech in TECH_FREQS:
            vals=[]
            for seed in range(N_SEEDS):
                a0=MyopicBestResponseAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,th)
                a1=FixedRuleAgent(1,FLOOR,CEIL)
                r=run_duopoly(a0,a1,DuopolyConfig(T=T,burn_in=BURN_IN,tech_freq_0=tech,seed=seed))
                vals.append(r.effective_speed(0))
            rows.append(dict(cost_threshold=th,tech_freq=tech,reprice_rate_mean=np.mean(vals),reprice_rate_std=np.std(vals)))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS_DIR,"baseline_grid.csv"),index=False)
    plt.figure(figsize=(7,5))
    for th in THRESHOLDS:
        s=df[df.cost_threshold==th]; plt.plot(s.tech_freq,s.reprice_rate_mean,marker="o",label=f"adjustment threshold={th}")
    plt.plot(TECH_FREQS,TECH_FREQS,"k--",alpha=.4,label="technical opportunity rate")
    plt.xlabel("Technical recomputation probability"); plt.ylabel("Reprices / period"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"effective_speed_vs_tech_speed.png"),dpi=150)

if __name__=="__main__": main()
