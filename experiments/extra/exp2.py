from common import *
N=16
specs=[]
for C in [0.,.0005,.002,.008]:
    for tau in [.05,.1,.25,.5,1.]:
        for s in range(N): specs.append(dict(kind='myopic',C=C,tau=tau,seed=s))
for rho in [0.,.5,1.]:
    for tau in [.1,.5,1.]:
        for s in range(N): specs.append(dict(kind='anticip',C=0.,rho=rho,tau=tau,seed=s))
if __name__=='__main__': run_all(specs,'exp2')
