"""Optional conditional Flow/DDPM training; immutable bundled inference weights.

Time O(steps * batch * network cost); GPU memory holds one batch. Raw MNIST stays
uint8 on CPU. An epoch shuffles all 55,000 training indices without replacement.
"""
from pathlib import Path
import copy,json,math,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .models import ConditionalTinyUNet,cosine_schedule
from .data import training_data
from .runtime import ROOT,choose_device,synchronize


def pair(clean,kind,rng,schedule):
    device=clean.device
    noise=torch.randn(clean.shape,generator=rng).to(device)
    if kind=='flow':
        t=torch.rand(len(clean),generator=rng).to(device)
        return (1-t[:,None,None,None])*noise+t[:,None,None,None]*clean,t,clean-noise
    ids=torch.randint(len(schedule['abar']),(len(clean),),generator=rng).to(device)
    a=schedule['abar'][ids][:,None,None,None]
    return a.sqrt()*clean+(1-a).sqrt()*noise,ids.float()/(len(schedule['abar'])-1),noise


def train_model(kind='flow',epochs=None,steps=None,batch_size=64,learning_rate=3e-4,
                ema_decay=.995,seed=42,validation_every=250,validation_samples=256,
                device_preference='auto',run_dir=None):
    if kind not in ('flow','diffusion'): raise ValueError('kind must be flow or diffusion')
    for name,value in [('batch_size',batch_size),('validation_every',validation_every),('validation_samples',validation_samples)]:
        if not isinstance(value,int) or isinstance(value,bool) or value<1: raise ValueError(f'{name} must be positive integer')
    if epochs is not None and steps is not None: raise ValueError('Choose epochs or steps, not both')
    for name,value in [('epochs',epochs),('steps',steps)]:
        if value is not None and (not isinstance(value,int) or isinstance(value,bool) or value<1): raise ValueError(f'{name} must be positive integer')
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**63-1: raise ValueError('Invalid seed')
    if not math.isfinite(learning_rate) or learning_rate<=0 or not math.isfinite(ema_decay) or not 0<=ema_decay<1: raise ValueError('Invalid learning rate/EMA')
    device=choose_device(device_preference);torch.set_num_threads(4);torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False
    raw,labels,train,valid,split_hash=training_data()
    per_epoch=math.ceil(len(train)/batch_size)
    total_steps=epochs*per_epoch if epochs is not None else (steps or 1000)
    out=Path(run_dir) if run_dir is not None else ROOT/'outputs/training'/f'{kind}_{time.time_ns()}'
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use an empty run directory; training does not overwrite previous runs.')
    out.mkdir(parents=True,exist_ok=True)
    config={'architecture':'conditional_tiny_unet_v1','channels':24,'num_classes':10,
            'normalization':'pixel / 127.5 - 1','kind':kind,'batch_size':batch_size,
            'learning_rate':learning_rate,'ema_decay':ema_decay,'seed':seed,'diffusion_steps':100,
            'split_sha256':split_hash,'validation_samples':min(validation_samples,5000),
            'validation_every':validation_every,'sampling':'shuffled epochs without replacement',
            'steps_per_epoch':per_epoch,'target_steps':total_steps,'device':str(device)}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    schedule={k:v.to(device) for k,v in cosine_schedule(100).items()}
    net=ConditionalTinyUNet(24).to(device);ema=copy.deepcopy(net).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    rng=torch.Generator().manual_seed(seed+1)
    ids=valid[:config['validation_samples']]
    vx=raw[ids].float()/127.5-1;vy=labels[ids]
    # Fixed validation pairs, held-out images only; no test data loaded.
    val_schedule={k:v.cpu() for k,v in schedule.items()}
    vs,vt,vu=pair(vx,kind,torch.Generator().manual_seed(1701),val_schedule)
    records=[];best=float('inf');samples_seen=0;step=0;window_loss=0.;window_samples=0
    synchronize(device);start=time.perf_counter()
    while step<total_steps:
        permutation=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)]
        for ids in permutation.split(batch_size):
            if step>=total_steps: break
            clean=(raw[ids].float()/127.5-1).to(device);y=labels[ids].to(device)
            state,t,target=pair(clean,kind,rng,schedule)
            net.train();optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(net(state,t,y),target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for averaged,live in zip(ema.parameters(),net.parameters()):averaged.lerp_(live,1-ema_decay)
            step+=1;samples_seen+=len(ids);window_loss+=loss.item()*len(ids);window_samples+=len(ids)
            if step==1 or step%validation_every==0 or step==total_steps:
                total=0.
                with torch.inference_mode():
                    for begin in range(0,len(vs),batch_size):
                        sl=slice(begin,begin+batch_size)
                        error=F.mse_loss(ema(vs[sl].to(device),vt[sl].to(device),vy[sl].to(device)),vu[sl].to(device))
                        total+=error.item()*len(vs[sl])
                val=total/len(vs)
                if not math.isfinite(val):raise RuntimeError('Nonfinite validation loss')
                synchronize(device)
                row={'step':step,'epoch':samples_seen/55000,'training_mse':window_loss/window_samples,
                     'validation_mse':val,'learning_rate':learning_rate,'gradient_norm_before_clip':norm.item(),
                     'elapsed_seconds':time.perf_counter()-start}
                records.append(row);pd.DataFrame(records).to_csv(out/'history.csv',index=False)
                print(f"{kind}: step={step}/{total_steps}, epoch={row['epoch']:.3f}, train MSE={row['training_mse']:.5f}, validation MSE={val:.5f}",flush=True)
                window_loss=0.;window_samples=0
                if val<best:
                    best=val
                    torch.save({'kind':kind,'state_dict':{k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
                                'config':config,'completed_steps':step,'weight_type':'best_validation_EMA','history':records},out/'best.pt')
    # Latest checkpoint supports explicit future resume implementations; never claimed to resume now.
    torch.save({'kind':kind,'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'generator_state':rng.get_state(),'config':config,'completed_steps':step,'history':records},out/'latest.pt')
    report={'kind':kind,'completed_steps':step,'epochs_seen':samples_seen/55000,'best_validation_mse':best,
            'test_data_used':False,'wall_seconds':time.perf_counter()-start,'checkpoint':str(out/'best.pt'),
            'note':'Short runs validate the pipeline; do not establish generation quality.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    return out,records
