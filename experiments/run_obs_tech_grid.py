"""Observability x technical capacity x adjustment-threshold experiment.

Agents decide only from the rival price they actually observed.  No true current
rival price is passed into their decision rule; reality affects realised payoff only.
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
A=np.array([2.0,2.0]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
TECH_FREQS=np.linspace(.1,1.,6); OBS_FREQS=np.linspace(.1,1.,6)
COST_THRESHOLDS=[0.,.0005,.002,.008]; N_SEEDS=10; T=5000; BURN_IN=500


def make_agents(threshold):
    return (
        MyopicBestResponseAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,threshold),
        MyopicBestResponseAgent(1,0,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,0.),
    )


def main():
    rows=[]
    for threshold in COST_THRESHOLDS:
        for tech in TECH_FREQS:
            for obs in OBS_FREQS:
                rates=[]
                for seed in range(N_SEEDS):
                    x,y=make_agents(threshold)
                    r=run_duopoly(x,y,DuopolyConfig(
                        T=T,burn_in=BURN_IN,tech_freq_0=tech,tech_freq_1=1.,
                        obs_freq_0=obs,obs_freq_1=1.,seed=seed,
                        cost_shock_std=.01,cost_shock_std_1=.01,cost_rho=.98))
                    rates.append(r.effective_speed(0))
                rows.append(dict(cost_threshold=threshold,tech_freq=tech,obs_freq=obs,
                                 reprice_rate_mean=np.mean(rates),reprice_rate_std=np.std(rates)))
                print(threshold, tech, obs, np.mean(rates))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS_DIR,"obs_tech_grid.csv"),index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    vmin=max(df.reprice_rate_mean.min(),1e-5); vmax=df.reprice_rate_mean.max()
    from matplotlib.colors import LogNorm
    for ax,threshold in zip(axes.flat,COST_THRESHOLDS):
        sub=df[df.cost_threshold==threshold]
        p=sub.pivot(index="obs_freq",columns="tech_freq",values="reprice_rate_mean")
        im=ax.imshow(np.clip(p.values,vmin,None),origin="lower",aspect="auto",
                     norm=LogNorm(vmin=vmin,vmax=vmax),extent=[.1,1.,.1,1.])
        ax.set_title(f"adjustment threshold = {threshold}"); ax.set_xlabel("tech_freq"); ax.set_ylabel("obs_freq")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Reprices / period",shrink=.8)
    fig.suptitle("Technical opportunity, observability, and adjustment friction")
    plt.savefig(os.path.join(RESULTS_DIR,"obs_tech_heatmaps.png"),dpi=150,bbox_inches="tight")


if __name__=="__main__": main()
