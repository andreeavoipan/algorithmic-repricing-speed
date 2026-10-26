from common import *
N=10
specs=[]
for sd in [.01,.05]:
    for C in [.0005,.002]:
        for tau in [.3,1.]:
            for om in [.1,.5,1.]:
                for s in range(N): specs.append(dict(kind='belief',belief_sd=sd,C=C,tau=tau,omega=om,seed=s))
# benchmark rule on the same grid for direct comparison
for C in [.0005,.002]:
    for tau in [.3,1.]:
        for om in [.1,.5,1.]:
            for s in range(N): specs.append(dict(kind='myopic',C=C,tau=tau,omega=om,seed=s))
if __name__=='__main__': run_all(specs,'exp4')
