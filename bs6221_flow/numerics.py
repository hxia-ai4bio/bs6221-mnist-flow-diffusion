"""Small, paired Flow solver experiments for interactive exploration.

Time O(samples * (reference NFE + all tested NFE * repeats) * model cost).
Reference solutions are CPU float64 approximations; refinement is a diagnostic.
"""
import copy,math,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from .sampling import sample,STAGES,_initial_state


def compare_numerics(net,digit=3,seed=42,samples=2,steps=(4,8,16,32),
                     nfe_budgets=(4,8,20,40),reference_steps=(128,256),repeats=3,tolerance=1e-5):
    for name,values in [('steps',steps),('nfe_budgets',nfe_budgets),('reference_steps',reference_steps)]:
        if not values or any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in values):raise ValueError(f'{name}: positive integers required')
        if len(set(values))!=len(values):raise ValueError(f'{name}: duplicate values')
    if any(n%4 for n in nfe_budgets):raise ValueError('NFE budgets must be multiples of four')
    if len(reference_steps)<2 or tuple(sorted(reference_steps))!=tuple(reference_steps):raise ValueError('Reference levels must be strictly increasing')
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<1 or not math.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid repeats/tolerance')
    z,y=_initial_state(digit,samples,seed,1.)
    reference_model=copy.deepcopy(net).cpu().double().eval().requires_grad_(False)
    references=[sample(reference_model,z,y,n,'RK4')[0] for n in reference_steps]
    reference=references[-1]
    gap=(references[-2]-reference).square().flatten(1).mean(1).sqrt().numpy()
    cases={(method,n) for method in STAGES for n in steps}
    cases.update((method,b//stages) for method,stages in STAGES.items() for b in nfe_budgets)
    rng=np.random.default_rng(seed);order=sorted(cases);rng.shuffle(order)
    rows=[];endpoints={}
    for method,n in order:
        sample(net,z,y,min(n,2),method)
        durations=[]
        for repeat in range(repeats):
            endpoint,seconds,calls,_=sample(net,z,y,n,method)
            durations.append(seconds)
        error=(endpoint.double()-reference).square().flatten(1).mean(1).sqrt().numpy()
        rows.append({'method':method,'steps':n,'h':1/n,'nfe':calls,'samples':samples,
                     'batch_seconds_median':float(np.median(durations)),
                     'rmse_median':float(np.median(error)),'rmse_max':float(error.max())})
        endpoints[(method,n)]=endpoint
    table=pd.DataFrame(rows).sort_values(['method','steps'])
    convergence=table[table.steps.isin(steps)].copy()
    same_nfe=table[table.nfe.isin(nfe_budgets)].copy()
    floor=max(float(gap.max()),1e-12)
    orders=[]
    for method in STAGES:
        part=convergence[(convergence.method==method)&(convergence.rmse_median>10*floor)&(convergence.steps>=20)].sort_values('steps').tail(3)
        slope=float(np.polyfit(np.log(part.h),np.log(part.rmse_median),1)[0]) if len(part)>=3 and gap.max()<=tolerance else float('nan')
        orders.append({'method':method,'observed_slope':slope,'fit_steps':part.steps.tolist(),
                       'interpretation':'diagnostic only; verify asymptotic regime and reference refinement'})
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for method in STAGES:
        part=convergence[convergence.method==method].sort_values('h')
        axes[0].loglog(part.h,part.rmse_median,'o-',label=method)
        part=same_nfe[same_nfe.method==method].sort_values('nfe')
        axes[1].loglog(part.nfe,part.rmse_median,'o-',label=method)
        axes[2].plot(part.nfe,part.batch_seconds_median,'o-',label=method)
    axes[0].set(xlabel='Step size h',ylabel='RMSE to approximate reference',title='Step convergence')
    axes[1].set(xlabel='NFE',ylabel='RMSE to approximate reference',title='Matched NFE')
    axes[2].set(xlabel='NFE',ylabel=f'Batch seconds ({samples} images)',title=f'Median of {repeats} repeats')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();plt.show()
    report={'reference_steps':list(reference_steps),'max_refinement_gap':float(gap.max()),
            'reference_diagnostic_passed':bool(gap.max()<=tolerance),'reference_tolerance':tolerance,
            'reference_is_exact':False,'device':str(next(net.parameters()).device),'seed':seed,
            'labels':y.tolist(),'weight_sha256':getattr(net,'flow_weight_sha256',None),
            'timing_excludes_loading_plotting_and_reference':True}
    print(report);display(convergence);display(same_nfe);display(pd.DataFrame(orders))
    return {'convergence':convergence,'same_nfe':same_nfe,'orders':pd.DataFrame(orders),
            'reference_report':report,'endpoints':endpoints,'noise':z,'labels':y}
