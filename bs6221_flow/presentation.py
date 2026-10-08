"""Figures and teaching demonstrations; numerical/model code lives in Python.

PCA is fitted on training images only. 2D clouds are projections of empirical
samples, not a recovered 784D density or exact cell/sample trajectories.
"""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from .output import finish_figure, print_table
import matplotlib.pyplot as plt
import torch
from .data import examples,training_data
from .runtime import ROOT,choose_device
from .models import ConditionalTinyUNet,cosine_schedule
from .sampling import generate_images,show_trajectory,load_flow


def show_training(history=None,kind='flow'):
    if history is None:
        path=ROOT/'results/training'/f'{kind}_history.csv'
        table=pd.read_csv(path)
        config=json.loads((ROOT/'results/training'/f'{kind}_config.json').read_text())
        print_table(pd.Series(config,name='Original training configuration').to_frame())
    else: table=pd.DataFrame(history)
    if table.empty: raise ValueError('No training history')
    fig,axes=plt.subplots(1,2,figsize=(11,3.3))
    # Different validation protocols remain separate series; never joined silently.
    groups=table.groupby('phase',sort=False) if 'phase' in table else [('new_run',table)]
    for phase,part in groups:
        axes[0].plot(part.step,part.training_mse,'o-',label=f'{phase}: training')
        axes[0].plot(part.step,part.validation_mse,'s--',label=f'{phase}: validation')
        axes[1].plot(part.epoch,part.validation_mse,'o-',label=str(phase))
    axes[0].set(xlabel='Optimizer step',ylabel='MSE',title=f'{kind}: own prediction target')
    axes[1].set(xlabel='Epoch / equivalent data passes',ylabel='Validation MSE',title='Fixed held-out validation')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f"{kind} training history")
    fig.tight_layout();finish_figure()
    print_table(table.tail(8));return table


def image_sequence(states,times,title):
    fig,axes=plt.subplots(1,len(states),figsize=(2*len(states),2.5),squeeze=False)
    for ax,state,t in zip(axes[0],states,times):
        ax.imshow(state[0,0],cmap='gray',vmin=-1,vmax=1);ax.axis('off');ax.set_title(f'{t:.2f}')
    fig.suptitle(title);fig.tight_layout();finish_figure();return fig


def forward_demo(digit=3,seed=42,diffusion_steps=100):
    clean,_,_=examples(digit,count=1)
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    levels=np.linspace(0,1,6)
    flow=[(1-s)*clean+s*noise for s in levels]
    image_sequence(flow,levels,'Flow construction: image -> noise (s increases)')
    schedule=cosine_schedule(diffusion_steps);rng=torch.Generator().manual_seed(seed)
    state=clean.clone();states=[state.clone()];marks=set(np.linspace(0,diffusion_steps,6).round().astype(int)[1:]);times=[0.]
    for j in range(diffusion_steps):
        a=schedule['alphas'][j]
        state=a.sqrt()*state+(1-a).sqrt()*torch.randn(state.shape,generator=rng)
        if j+1 in marks:states.append(state.clone());times.append((j+1)/diffusion_steps)
    image_sequence(states,times,'DDPM forward: fresh noise at each step')
    return {'flow':flow,'diffusion':states,'noise':noise}


def probability_paths(seed=42,samples=300):
    if not isinstance(samples,int) or isinstance(samples,bool) or not 20<=samples<=3000:raise ValueError('samples must be 20..3000')
    raw,labels,train,_,_=training_data();rng=torch.Generator().manual_seed(seed)
    ids=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)[:samples]]
    flat=(raw[ids].float()/127.5-1).flatten(1);center=flat.mean(0,keepdim=True)
    # Exact SVD on the small teaching subset; axes fitted once, then fixed.
    _,singular,vt=torch.linalg.svd(flat-center,full_matrices=False);basis=vt[:2].T
    variance=float(singular[:2].square().sum()/singular.square().sum())
    noise=torch.randn(flat.shape,generator=rng)
    levels=[0.,.25,.5,.75,1.]
    clouds=[((1-t)*noise+t*flat-center)@basis for t in levels]
    fig,axes=plt.subplots(1,len(levels),figsize=(14,3),sharex=True,sharey=True)
    colors=labels[ids].numpy()
    for ax,cloud,t in zip(axes,clouds,levels):
        ax.scatter(cloud[:,0],cloud[:,1],c=colors,cmap='tab10',s=8,vmin=0,vmax=9,alpha=.65)
        # Paired paths of the first 8 samples, selected before any model inference.
        for i in range(8):
            curve=torch.stack([c[i] for c in clouds[:levels.index(t)+1]])
            ax.plot(curve[:,0],curve[:,1],color='gray',alpha=.25,lw=.8)
        ax.set_title(f't={t:.2f}');ax.set_xlabel('Fixed train PCA 1')
    axes[0].set_ylabel('Fixed train PCA 2')
    fig.suptitle(f'Constructed Flow probability path: noise -> training images; 2D variance {variance:.1%}')
    fig.tight_layout();finish_figure()
    print('This is an empirical projected path, not learned generation or a 784D density. Time t=1-s of the image-to-noise view.')
    return {'clouds':clouds,'times':levels,'indices':ids,'variance_ratio':variance}


@torch.inference_mode()
def diffusion_demo(digit=3,seed=42,step_seed=43,weights=None,device_preference='auto'):
    path=Path(weights) if weights is not None else ROOT/'checkpoints/frozen/20261002_flow15000_ddpm20000/diffusion_ema.pt'
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('kind')!='diffusion':raise ValueError('Expected DDPM checkpoint')
    device=choose_device(device_preference)
    model=ConditionalTinyUNet(payload['config']['channels']).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['state_dict'],strict=True)
    schedule={k:v.to(device) for k,v in cosine_schedule(payload['config'].get('diffusion_steps',100)).items()}
    state=torch.randn(1,1,28,28,generator=torch.Generator().manual_seed(seed)).to(device)
    y=torch.tensor([digit],device=device);rng=torch.Generator().manual_seed(step_seed)
    total=len(schedule['abar']);marks=set(np.linspace(0,total,6).round().astype(int)[1:]);states=[state.cpu().clone()];times=[0.]
    for k in reversed(range(total)):
        t=torch.full((1,),k/(total-1),device=device)
        eps=model(state,t,y);a=schedule['abar'][k]
        clean=((state-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
        state=schedule['coef_x0'][k]*clean+schedule['coef_xt'][k]*state
        if k:state+=schedule['posterior_var'][k].sqrt()*torch.randn(state.shape,generator=rng).to(device)
        if total-k in marks:states.append(state.cpu().clone());times.append((total-k)/total)
    image_sequence(states,times,'Learned DDPM generation: reverse sampling progress (not Flow time)')
    return {'states':states,'times':times,'noise_seed':seed,'step_seed':step_seed}


def generation_demo(net=None,digit=3,seed=42,steps=40):
    return generate_images(digit=digit,n=4,seed=seed,steps=steps,trajectory=True,net=net)


def noise_preview(seed=42,scale=1.):
    if not math.isfinite(scale) or scale<=0:raise ValueError('Scale must be finite and positive')
    noise=torch.randn(1,1,28,28,generator=torch.Generator().manual_seed(seed))*scale
    image_sequence([noise],[0.],f'Chosen initial Gaussian noise, seed={seed}, scale={scale}')
    return noise


def show_saved_numerics():
    folder=ROOT/'results/numerical_analysis'
    metadata=json.loads((folder/'provenance.json').read_text())
    print_table(pd.Series(metadata,name='Original experiment provenance').to_frame())
    for name in ['analytic_orders.csv','flow_observed_orders.csv','equal_nfe_comparison.csv']:
        print_table(pd.read_csv(folder/name))
    import shutil
    from .output import _FOLDER
    _FOLDER.mkdir(parents=True,exist_ok=True)
    for path in sorted(folder.glob('*.png')):
        target=_FOLDER/path.name
        if path.resolve()!=target.resolve():shutil.copy2(path,target)
        print('Saved historical figure:',target)
