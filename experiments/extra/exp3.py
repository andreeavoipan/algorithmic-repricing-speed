from common import *
N=10
DEM={'baseline':dict(),'mu0.1':dict(mu=.1),'mu0.5':dict(mu=.5),'asym':dict(a=[2.,2.5]),'a0_1':dict(a0=1.)}
specs=[]
for name,d in DEM.items():
    for C in [.0005,.002,.008]:
        for tau in [.05,.1,.25,.5,1.]:
            for s in range(N): specs.append(dict(kind='myopic',dem=name,C=C,tau=tau,seed=s,**d))
    for tau in [.1,.5,1.]:
        for s in range(N): specs.append(dict(kind='anticip',dem=name,C=0.,rho=1.,tau=tau,seed=s,**d))
if __name__=='__main__': run_all(specs,'exp3')
