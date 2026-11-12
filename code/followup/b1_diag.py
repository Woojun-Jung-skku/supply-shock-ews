"""B1 diagnostic (Table S14, continued): why the negative-binomial design does not restore the EWMA run length.
Realised ARL0 of the NB-designed charts in the four stable cells of Study B with phase-I estimates, the oracle
in-control mean, oracle mean and dispersion, and data without indicator effects (beta = 0). Prints four lines;
usage: python followup/b1_diag.py  (writes nothing; copy the output into B1_nbcharts_diag.csv)."""
import os, sys, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); CODE=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,CODE); sys.path.insert(0,os.path.join(CODE,'tools'))
import config as C; C.apply_profile("t4_min")
import dgp, charts as CH, metrics as MT, arl as ARL
import b1_nbcharts as B1
cells=[c for c in B1.cells() if c['regime']=='stable']
months=np.arange(C.ORIGINS[0], C.DECISION_END+1)
def run(variant):
    rows=[]
    for c in cells:
        for r in range(5):
            beta_save=C.BETA
            if variant=='beta0': C.BETA=(0.0,0.0,0.0)
            s=dgp.simulate(c['I'],c['T'],float(c['mu']),float(c['alpha']),c['regime'],seed=dgp.seed_of('B',c['id'],r),miss=0.0)
            C.BETA=beta_save
            N,group,tr=s['N'],s['group'],s['truth']
            a,b=C.CHART_PHASE1
            mu0,alpha=B1.nb_params(N,group)
            if variant in('oracle_mu','oracle_both'):
                X=s['X']; Xlag=X[:,:c['T']]
                mu_true=np.exp(tr['m'][group]+np.einsum('itk,k->it',Xlag,tr['beta'] if variant!='beta0' else np.zeros(3))+tr['logC'][:,None])
                mu0=mu_true[:,a:b].mean(1)
            if variant=='oracle_both': alpha=np.full(N.shape[0],float(c['alpha']))
            rng=np.random.default_rng(dgp.seed_of('nbchart',c['id'],r))
            sim=B1.nb_sim(rng,mu0,alpha,CH._CAL_MONTHS)
            h_e=CH._limit(CH._ewma(sim,C.EWMA_LAMBDA,mu0)[:,100:],C.ARL0_TARGET)
            k=B1.nb_k(mu0,alpha,C.RHO_SHIFT); h_c=CH._limit(CH._cusum_k(sim,k)[:,100:],C.ARL0_TARGET)
            run_=N[:,b:].astype(float); e=CH._ewma(run_,C.EWMA_LAMBDA,mu0); cc=CH._cusum_k(run_,k)
            idx=months-1-b
            ea=(e[:,idx]>h_e[:,None]).T; ca=(cc[:,idx]>h_c[:,None]).T
            for key,A in (('ewma',ea),('cusum',ca)):
                rl=MT.run_length(A,months,tr['t0'],'stable')
                rows.append(dict(cell=c['id'],rep=r,regime='stable',key=key,arl0=rl['arl0'],units=rl['stable_alarm_units']))
    d=pd.DataFrame(rows); out={}
    for key in ('ewma','cusum'):
        dd=d[d.key==key].rename(columns={'arl0':f'{key}_arl0','units':f'{key}_stable_alarm_units'})
        pr=ARL.pooled(dd,f'{key}_arl0',f'{key}_stable_alarm_units',ARL.UNITS['item']); out[key]=f"{pr['est']:.1f} ({pr['lo']:.1f}-{pr['hi']:.1f})"
    return out
for v in ('estimated','oracle_mu','oracle_both','beta0'):
    print(v, run(v))
