"""Python CLI for training curves, noise paths, generation and solver analysis."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bs6221_flow.output import configure_output
from bs6221_flow.runtime import ROOT
import numpy as np
import torch
from bs6221_flow.presentation import show_training,forward_demo,probability_paths,diffusion_demo,noise_preview,show_saved_numerics,image_sequence
from bs6221_flow.sampling import load_flow,generate_images,compare_solvers
from bs6221_flow.numerics import compare_numerics


def integers(value):
    try:values=tuple(int(x) for x in value.split(','))
    except ValueError:raise argparse.ArgumentTypeError('Use comma-separated integers')
    if not values or any(x<1 for x in values):raise argparse.ArgumentTypeError('Use positive integers')
    return values


def save_generation(result,folder):
    torch.save({k:v for k,v in result.items() if k!='params'},folder/'generation_states.pt')
    np.save(folder/'initial_noise.npy',result['noise'].numpy())
    (folder/'generation_params.json').write_text(json.dumps(result['params'],indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','training-curves','forward','paths','generate','diffusion','compare','numerics','saved'])
    p.add_argument('--out',type=Path)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--digit',type=int,choices=range(10),default=3)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--step-seed',type=int,default=43)
    p.add_argument('--count',type=int,default=4);p.add_argument('--noise-scale',type=float,default=1.)
    p.add_argument('--noise-file',type=Path,help='Saved .npy initial noise, shape [count,1,28,28], generate stage only')
    p.add_argument('--weights',type=Path,help='Flow frozen EMA or newly trained best.pt')
    p.add_argument('--diffusion-weights',type=Path)
    p.add_argument('--method',choices=['Euler','Heun','RK4'],default='Heun')
    p.add_argument('--steps',type=int,default=40);p.add_argument('--nfe',type=int,help='Equal-NFE comparison; multiple of four')
    p.add_argument('--path-samples',type=int,default=300)
    p.add_argument('--history',type=Path,help='Training run directory containing config.json/history.csv')
    p.add_argument('--samples',type=int,default=2,help='Number of independent noises for numerical exploration')
    p.add_argument('--step-list',type=integers,default=(4,8,16,32))
    p.add_argument('--nfe-list',type=integers,default=(4,8,20,40))
    p.add_argument('--reference-steps',type=integers,default=(128,256))
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--tolerance',type=float,default=1e-5)
    a=p.parse_args()
    if a.noise_file is not None and a.stage!='generate':p.error('--noise-file is only supported for generate')
    if a.noise_file is not None and a.noise_scale!=1.:p.error('Use --noise-scale 1 with --noise-file')
    if a.history is not None and a.stage!='training-curves':p.error('--history requires training-curves')
    if a.seed<0 or a.step_seed<0 or a.count<1 or a.steps<1:p.error('Invalid seed/count/steps')
    torch.set_num_threads(4)
    folder=configure_output(a.out or ROOT/'outputs/demo'/f'{a.stage}_{time.time_ns()}')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (folder/'command.json').write_text(json.dumps(config,indent=2)+'\n')
    stages=['training-curves','forward','paths','generate','diffusion','compare','saved'] if a.stage=='all' else [a.stage]
    net=load_flow(a.weights,a.device) if any(s in stages for s in ['generate','compare','numerics']) else None
    for stage in stages:
        print('\nRunning:',stage,flush=True)
        if stage=='training-curves':
            if a.history:
                import pandas as pd
                cfg=json.loads((a.history/'config.json').read_text());print(cfg)
                table=show_training(history=pd.read_csv(a.history/'history.csv'),kind=cfg['kind'])
                table.to_csv(folder/'training_history.csv',index=False)
            else:
                for kind in ['flow','diffusion']:
                    show_training(kind=kind).to_csv(folder/f'{kind}_history.csv',index=False)
        elif stage=='forward':forward_demo(a.digit,a.seed)
        elif stage=='paths':
            result=probability_paths(a.seed,a.path_samples)
            torch.save(result,folder/'probability_paths.pt')
        elif stage=='generate':
            noise=torch.from_numpy(np.load(a.noise_file,allow_pickle=False)) if a.noise_file else None
            if noise is None:noise_preview(a.seed,a.noise_scale)
            else:image_sequence([noise],[0.],'Chosen custom initial noise')
            result=generate_images(a.digit,a.count,a.method,a.steps,a.seed,a.noise_scale,trajectory=True,net=net,noise=noise)
            save_generation(result,folder);print(result['params'])
        elif stage=='diffusion':
            result=diffusion_demo(a.digit,a.seed,a.step_seed,a.diffusion_weights,a.device)
            torch.save(result,folder/'diffusion_states.pt')
        elif stage=='compare':
            result=compare_solvers(a.digit,a.count,a.steps,a.nfe,a.seed,a.noise_scale,net=net)
            result['table'].to_csv(folder/'solver_comparison.csv',index=False)
            torch.save({k:result[k] for k in ['noise','labels','results']},folder/'solver_states.pt')
            (folder/'comparison.json').write_text(json.dumps({'mode':result['comparison_mode'],'weight_sha256':net.flow_weight_sha256,'seed':a.seed,'noise_scale':a.noise_scale},indent=2)+'\n')
        elif stage=='numerics':
            result=compare_numerics(net,a.digit,a.seed,a.samples,a.step_list,a.nfe_list,a.reference_steps,a.repeats,a.tolerance)
            for key in ['convergence','same_nfe','orders']:result[key].to_csv(folder/f'{key}.csv',index=False)
            (folder/'reference_report.json').write_text(json.dumps(result['reference_report'],indent=2)+'\n')
            torch.save({k:result[k] for k in ['noise','labels','endpoints']},folder/'numerical_states.pt')
        elif stage=='saved':show_saved_numerics()
    print('\nOutputs:',folder)

if __name__=='__main__':main()
