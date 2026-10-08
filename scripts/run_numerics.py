"""Recompute frozen-model ODE experiments independently of Jupyter."""
from pathlib import Path
import argparse
import flow_numerics as nm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--recompute',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    nm.analytic_experiment(a.out);nm.stability_experiment(a.out)
    e=nm.FlowExperiment(a.out,per_class=2 if a.pilot else 10,recompute=a.recompute)
    e.reference((128,256,512) if a.pilot else (1280,2560,5120),max_difference=1e-5)
    e.convergence((20,40,80) if a.pilot else (5,10,20,40,80,160,320,640))
    e.efficiency((20,40,80) if a.pilot else (20,40,80,160,320),repeats=3)
    e.finish()

if __name__=='__main__':main()
