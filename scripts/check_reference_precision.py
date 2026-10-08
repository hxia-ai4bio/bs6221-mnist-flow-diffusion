"""Reproduce the small MPS/CPU precision pilot used by the numerical notebook."""
import argparse
import json
import numpy as np
import torch
import flow_numerics as nm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute',action='store_true')
    args=parser.parse_args()
    out=nm.ROOT/'outputs/numerical_analysis'
    out.mkdir(parents=True,exist_ok=True)
    p=out/'pilot.pt'; r=out/'pilot.json'; meta=out/'pilot_provenance.json'
    torch.set_num_threads(4)
    x,y=nm.initial_conditions()
    ids=torch.arange(0,100,10)
    x,y=x[ids],y[ids]
    provenance={'snapshot':nm.SNAPSHOT.name,'model_sha256':nm.sha256(nm.SNAPSHOT/'flow_ema.pt'),
                'initial_seed':6221003,'gpu_sample_ids':ids.tolist(),'cpu_sample_id':30,
                'cpu_sample_digit':3,'method':'RK4'}
    if p.exists() and r.exists() and not args.recompute:
        data=torch.load(p,map_location='cpu',weights_only=True)
        assert torch.equal(data['initial_noise'],x) and torch.equal(data['labels'],y)
        if meta.exists() and json.loads(meta.read_text())!=provenance:
            raise RuntimeError('Precision pilot provenance changed; use --recompute')
        nm.save_json(provenance,meta)
        print('Existing precision pilot verified; --recompute reruns all integrations.')
        return
    model,_,_=nm.load_snapshot()
    outputs={};rows=[];previous=None
    for steps in [160,320,640,1280,2560,5120]:
        z,seconds,nfe=nm.solve_model(model,x,y,steps,'RK4')
        outputs[steps]=z
        row={'device':'mps','dtype':'float32','steps':steps,'seconds':seconds,'nfe':nfe}
        if previous is not None:
            errors=nm.rmse(z,previous)
            row.update(previous_difference_median=float(np.median(errors)),previous_difference_max=float(errors.max()))
        previous=z;rows.append(row);print(row,flush=True)
    model64,_,_=nm.load_snapshot(device='cpu',dtype=torch.float64)
    previous=None
    for steps in [320,640,1280,2560,5120]:
        z,seconds,nfe=nm.solve_model(model64,x[3:4],y[3:4],steps,'RK4')
        row={'device':'cpu','dtype':'float64','steps':steps,'seconds':seconds,
             'mps_difference':float(nm.rmse(z,outputs[steps][3:4])[0])}
        if previous is not None: row['previous_difference']=float(nm.rmse(z,previous)[0])
        previous=z;rows.append(row);print(row,flush=True)
    nm.save_tensor({'initial_noise':x,'labels':y,'mps_endpoints':outputs,'cpu64_endpoint':previous},p)
    nm.save_json(rows,r);nm.save_json(provenance,meta)


if __name__=='__main__':
    main()
