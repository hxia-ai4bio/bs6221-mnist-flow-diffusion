"""Reproducible numerical experiments for the frozen conditional Flow ODE.

An integration with N steps costs stages*N network calls and O(B*784) state
memory, in addition to the model and temporary network activations. No training.
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import NullFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT/'checkpoints/frozen/20261002_flow15000_ddpm20000'
STAGES = {'Euler': 1, 'Heun': 2, 'RK4': 4}
COLORS = {'Euler': '#D55E00', 'Heun': '#0072B2', 'RK4': '#009E73'}


def ode_step(f, t, x, h, method):
    """One explicit step; x may be a scalar, a vector, or an image batch."""
    if method not in STAGES:
        raise ValueError(f'Unknown solver: {method}')
    k1 = f(t, x)
    if method == 'Euler':
        return x+h*k1
    if method == 'Heun':
        k2 = f(t+h, x+h*k1)
        return x+h*(k1+k2)/2
    k2 = f(t+h/2, x+h*k1/2)
    k3 = f(t+h/2, x+h*k2/2)
    k4 = f(t+h, x+h*k3)
    return x+h*(k1+2*k2+2*k3+k4)/6


def integrate(f, initial, steps, method, end=1.):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1 or end <= 0:
        raise ValueError('steps must be a positive integer, end must be positive')
    x = initial.clone() if torch.is_tensor(initial) else np.array(initial, copy=True)
    h = end/steps
    for j in range(steps):
        x = ode_step(f, j*h, x, h, method)
    return x


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(device='mps', dtype=torch.float32):
    manifest = json.loads((SNAPSHOT/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if sha256(SNAPSHOT/name) != digest:
            raise RuntimeError(f'Frozen snapshot was changed: {name}')
    spec = importlib.util.spec_from_file_location('analysis_frozen_models', SNAPSHOT/'model_definitions.py')
    definitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definitions)
    payload = torch.load(SNAPSHOT/'flow_ema.pt', map_location='cpu', weights_only=True)
    model = definitions.ConditionalTinyUNet(payload['config']['channels'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    classifier = definitions.DigitClassifier()
    classifier.load_state_dict(torch.load(SNAPSHOT/'digit_classifier.pt', map_location='cpu', weights_only=True)['state_dict'])
    classifier = classifier.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, classifier, manifest


def initial_conditions(per_class=10, seed=6221003):
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class < 2:
        raise ValueError('Use at least two independent samples per class')
    labels = torch.arange(10).repeat_interleave(per_class)
    noise = torch.randn(len(labels), 1, 28, 28, generator=torch.Generator().manual_seed(seed))
    return noise, labels


def synchronize(device):
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.inference_mode()
def solve_model(model, initial, labels, steps, method):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    x, y = initial.to(device=device, dtype=dtype), labels.to(device)
    calls = 0
    def field(t, state):
        nonlocal calls
        calls += 1
        return model(state, torch.full((len(state),), t, device=device, dtype=dtype), y)
    synchronize(device)
    start = time.perf_counter()
    result = integrate(field, x, steps, method)
    synchronize(device)
    elapsed = time.perf_counter()-start
    assert calls == STAGES[method]*steps
    if not torch.isfinite(result).all():
        raise RuntimeError(f'Nonfinite result: {method} N={steps}')
    return result.cpu(), elapsed, calls


def rmse(a, b):
    return (a.double()-b.double()).square().flatten(1).mean(1).sqrt().numpy()


def save_json(data, path):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def save_tensor(data, path):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def figure_file(fig, out, stem):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/f'{stem}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    return out/f'{stem}.png'


def analytic_experiment(out):
    """Exact-solution checks of consistency and local/global error orders."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, orders = [], []
    for method in STAGES:
        for steps in [5, 10, 20, 40, 80, 160]:
            h = 1/steps
            local = abs(float(ode_step(lambda t, u: -u, 0., 1., h, method))-np.exp(-h))
            global_error = abs(float(integrate(lambda t, u: -u, 1., steps, method))-np.exp(-1))
            rows.append(dict(method=method, steps=steps, h=h, raw_local_defect=local,
                             normalized_defect=local/h, global_error=global_error))
        # Nonautonomous test detects an incorrect time at intermediate RK stages.
        nonauto = np.array([abs(float(integrate(lambda t, u: t*u, 1., n, method))-np.exp(.5))
                            for n in [20, 40, 80]])
        slope = np.polyfit(np.log(1/np.array([20, 40, 80])), np.log(nonauto), 1)[0]
        expected = {'Euler': 1, 'Heun': 2, 'RK4': 4}[method]
        assert abs(slope-expected) < .2, (method, slope)
        vector = integrate(lambda t, u: np.array([-1., -2.])*u, np.ones(2), 100, method)
        assert vector.shape == (2,) and np.isfinite(vector).all()
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for method in STAGES:
        part = table[table.method == method]
        record = {'method': method}
        for ax, metric, title in zip(axes,
                ['raw_local_defect', 'normalized_defect', 'global_error'],
                ['One-step defect |d(h)|', 'Consistency: |d(h)| / h', 'Global error at t = 1']):
            valid = part[(part.steps <= 40) & (part.steps >= 10) & (part[metric] > 100*np.finfo(float).eps)]
            slope = float(np.polyfit(np.log(valid.h), np.log(valid[metric]), 1)[0])
            record[metric+'_slope'] = slope
            ax.loglog(part.h, part[metric], 'o-', color=COLORS[method], label=f'{method}: p={slope:.2f}')
            ax.set(title=title, xlabel='Step size h', ylabel='Absolute error')
            ax.grid(alpha=.2, which='both'); ax.legend(fontsize=8)
        orders.append(record)
    fig.suptitle("Exact benchmark: u' = -u, u(0) = 1; exact solution exp(-t)")
    fig.tight_layout()
    table.to_csv(out/'analytic_errors.csv', index=False)
    order_table = pd.DataFrame(orders)
    order_table.to_csv(out/'analytic_orders.csv', index=False)
    return order_table, figure_file(fig, out, '01_consistency_and_order')


def stability_function(z, method):
    if method == 'Implicit Euler':
        return 1/(1-z)
    result = 1+z
    if method in ('Heun', 'RK4'):
        result = result+z*z/2
    if method == 'RK4':
        result = result+z**3/6+z**4/24
    return result


def stability_experiment(out):
    """Absolute stability of a two-time-scale linear test problem, not the U-Net."""
    out = Path(out)
    methods = [*STAGES, 'Implicit Euler']
    hs, rates = np.logspace(-4, 0, 140), np.logspace(0, 3, 140)
    H, L = np.meshgrid(hs, rates)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, method in zip(axes, methods):
        amplitude = np.abs(stability_function(-L*H, method))
        colors = np.log10(np.maximum(amplitude, 1e-12))
        mesh = ax.pcolormesh(H, L, colors, shading='auto', cmap='coolwarm',
                             norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2))
        if method != 'Implicit Euler':
            ax.contour(H, L, amplitude, levels=[1.], colors='black', linewidths=1.)
        ax.set(xscale='log', yscale='log', xlabel='Step size h', title=method)
    axes[0].set_ylabel('Fast decay rate lambda')
    fig.subplots_adjust(top=.78, bottom=.2, left=.065, right=.86, wspace=.18)
    color_axis = fig.add_axes([.89, .2, .013, .58])
    fig.colorbar(mesh, cax=color_axis, label='log10 |R(-lambda h)|', extend='both')
    fig.suptitle("Linear test: u'=-u, w'=-lambda*w; black boundary |R|=1\nBlue: contraction; red: amplification", fontsize=13)
    heatmap = figure_file(fig, out, '02_absolute_stability')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    rows = []
    for ax, count in zip(axes, [4, 10, 20]):
        end, rate = .2, 100.
        h = end/count
        fine = np.linspace(0, end, 400)
        ax.plot(fine, np.exp(-rate*fine), 'k--', label='Exact fast component')
        for method in methods:
            values = stability_function(-rate*h, method)**np.arange(count+1)
            ax.plot(np.linspace(0, end, count+1), values, '.-',
                    color=COLORS.get(method, '#CC79A7'), label=method, markersize=3)
            rows.append({'method': method, 'h': h, 'lambda': rate,
                         'amplification': float(abs(stability_function(-rate*h, method))),
                         'endpoint_error': float(abs(values[-1]-np.exp(-rate*end)))})
        ax.set(title=f'h={h:g}; lambda*h={rate*h:g}', xlabel='Time', ylabel='Fast component w(t)',
               yscale='symlog', ylim=None)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle('Stable does not mean accurate: oscillation, decay and divergence')
    fig.tight_layout()
    table = pd.DataFrame(rows); table.to_csv(out/'linear_stability_cases.csv', index=False)
    return table, [heatmap, figure_file(fig, out, '03_stability_trajectories')]


def precision_experiment(out):
    """Same single image on both devices; no mixing of medians and individual errors."""
    root=ROOT/'outputs/numerical_analysis'
    meta=json.loads((root/'pilot_provenance.json').read_text())
    assert meta['model_sha256']==sha256(SNAPSHOT/'flow_ema.pt')
    pilot=pd.read_json(root/'pilot.json')
    data=torch.load(root/'pilot.pt',map_location='cpu',weights_only=True)
    table=pilot[pilot.device=='cpu'][['steps','previous_difference','mps_difference']].copy()
    table=table.rename(columns={'previous_difference':'cpu64_refinement_rmse','mps_difference':'cross_precision_rmse'})
    values=[]
    for n in table.steps:
        states=data['mps_endpoints']
        values.append(float(rmse(states[n][3:4],states[n//2][3:4])[0]) if n//2 in states else np.nan)
    table['mps32_refinement_rmse']=values
    table['cpu64_self_order']=np.log2(table.cpu64_refinement_rmse.shift(1)/table.cpu64_refinement_rmse)
    table.to_csv(Path(out)/'precision_pilot.csv',index=False)
    selected=table.dropna(subset=['cpu64_refinement_rmse'])
    fig,ax=plt.subplots(figsize=(8,4))
    ax.loglog(selected.steps,selected.cpu64_refinement_rmse,'o-',label='CPU float64: same image')
    ax.loglog(selected.steps,selected.mps32_refinement_rmse,'s-',label='MPS float32: same image')
    n=selected.steps.to_numpy()[-3:]
    anchor=selected.cpu64_refinement_rmse.iloc[-1]
    ax.loglog(n,anchor*(n[-1]/n)**4,'k:',label='Slope -4 guide')
    ax.set_xticks(selected.steps,[str(v) for v in selected.steps]);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel='Refined RK4 steps N',ylabel='RMSE between N and N/2 steps',
           title='Precision diagnostic: one fixed digit-3 sample')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=9);fig.tight_layout()
    return table,figure_file(fig,out,'12_precision_diagnostic')


class FlowExperiment:
    def __init__(self, out, per_class=10, seed=6221003, recompute=False):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out/'cache'; self.cache.mkdir(exist_ok=True)
        self.recompute, self.memo = recompute, {}
        torch.set_num_threads(4)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model, self.classifier, manifest = load_snapshot(self.device)
        self.initial, self.labels = initial_conditions(per_class, seed)
        code = '\n'.join(inspect.getsource(f) for f in (ode_step, integrate, solve_model))
        config = {'frozen_snapshot': SNAPSHOT.name, 'model_sha256': manifest['files_sha256']['flow_ema.pt'],
                  'architecture_sha256': manifest['files_sha256']['model_definitions.py'],
                  'solver_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                  'initial_seed': seed, 'samples_per_class': per_class, 'batch_size': len(self.labels),
                  'dtype': 'float32', 'device': str(self.device), 'torch': str(torch.__version__),
                  'normalization': '[-1,1] real images; no clipping of ODE states',
                  'test_set_used': False}
        self.config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        path = self.out/'configuration.json'
        if path.exists() and json.loads(path.read_text()) != config and not recompute:
            raise RuntimeError('Experiment settings changed. Use a new output directory or RECOMPUTE=True.')
        save_json(config, path)
        save_tensor({'noise': self.initial, 'labels': self.labels, 'seed': seed}, self.out/'initial_conditions.pt')
        self.uncertainty = None

    def case(self, method, steps):
        key = (method, steps)
        path = self.cache/f'{method}_{steps:05d}.pt'
        if key in self.memo:
            return self.memo[key]
        if path.exists() and not self.recompute:
            result = torch.load(path, map_location='cpu', weights_only=True)
            if result['config_id'] != self.config_id:
                raise RuntimeError(f'Incompatible cache: {path}')
        else:
            images, sec, nfe = solve_model(self.model, self.initial, self.labels, steps, method)
            result = {'images': images, 'seconds': sec, 'nfe': nfe, 'config_id': self.config_id}
            save_tensor(result, path)
            print(f'{method:5s} N={steps:5d}, NFE={nfe:5d}, batch time={sec:.2f}s', flush=True)
        self.memo[key] = result
        return result

    def reference(self, levels=(1280, 2560, 5120), max_difference=1e-5):
        if len(levels) < 2 or sorted(set(levels)) != list(levels):
            raise ValueError('Use at least two strictly increasing reference resolutions')
        rows, previous = [], None
        for steps in levels:
            images = self.case('RK4', steps)['images']
            if previous is not None:
                errors = rmse(images, previous)
                rows.append({'coarse_steps': previous_steps, 'fine_steps': steps,
                             'median_difference': float(np.median(errors)), 'max_difference': float(errors.max())})
            previous, previous_steps = images, steps
        self.reference_images, self.reference_steps = images, levels[-1]
        self.uncertainty = np.maximum(errors, 1e-6)
        # This is a diagnostic resolution, not a rigorous reference-error bound.
        accepted = bool(errors.max() < max_difference)
        self.reference_diagnostics = {'levels': list(levels), 'refinement': rows,
                                     'acceptance_max_rmse': max_difference, 'refinement_check_passed': accepted,
                                     'precision_floor_used': 1e-6, 'uncertainty_is_rigorous_bound': False}
        save_json(self.reference_diagnostics, self.out/'reference_check.json')
        pd.DataFrame(rows).to_csv(self.out/'reference_refinement.csv', index=False)
        if not accepted:
            raise RuntimeError(f'Reference refinement max RMSE {errors.max():.3g} > {max_difference:g}; refine further.')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].loglog([r['fine_steps'] for r in rows], [r['median_difference'] for r in rows], 'o-', label='Median')
        axes[0].loglog([r['fine_steps'] for r in rows], [r['max_difference'] for r in rows], 's--', label='Maximum')
        axes[0].axhline(max_difference, color='gray', linestyle=':', label='Maximum acceptance threshold')
        axes[0].set(xlabel='Refined RK4 steps', ylabel='RMSE between successive refinements', title='Reference refinement check')
        axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
        axes[1].hist(errors, bins=20, color='#0072B2')
        axes[1].set(xlabel='Last refinement RMSE per image', ylabel='Number of images', title=f'All {len(errors)} paired images')
        fig.tight_layout()
        return pd.DataFrame(rows), figure_file(fig, self.out, '04_reference_validation')

    @torch.inference_mode()
    def agreement(self, images):
        predicted = self.classifier(images.to(self.device).clamp(-1, 1)).argmax(1).cpu()
        return predicted == self.labels

    def convergence(self, steps=(1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640)):
        if self.uncertainty is None:
            raise RuntimeError('Validate the reference first')
        rows = []
        for method in STAGES:
            for count in steps:
                result = self.case(method, count)
                errors = rmse(result['images'], self.reference_images)
                max_error = (result['images'].double()-self.reference_images.double()).abs().flatten(1).max(1).values.numpy()
                correct = self.agreement(result['images']).numpy()
                for i, (error, linf, hit) in enumerate(zip(errors, max_error, correct)):
                    rows.append({'method': method, 'steps': count, 'h': 1/count, 'nfe': result['nfe'],
                                 'sample': i, 'label': int(self.labels[i]), 'rmse': error, 'max_pixel_error': linf,
                                 'label_correct': bool(hit), 'reference_resolution': self.uncertainty[i]})
        frame = pd.DataFrame(rows)
        frame.to_csv(self.out/'flow_errors_per_image.csv', index=False)
        table = frame.groupby(['method', 'steps', 'h', 'nfe'], sort=False).agg(
            median_rmse=('rmse', 'median'), mean_rmse=('rmse', 'mean'), q25=('rmse', lambda x: x.quantile(.25)),
            q75=('rmse', lambda x: x.quantile(.75)), worst_rmse=('rmse', 'max'),
            label_agreement=('label_correct', 'mean')).reset_index()
        table.to_csv(self.out/'flow_convergence_summary.csv', index=False)
        self.frame, self.table, self.steps = frame, table, list(steps)
        orders = []
        floor = float(np.median(self.uncertainty))
        for method in STAGES:
            part = table[(table.method == method) & (table.steps >= 20) & (table.median_rmse > 10*floor)]
            fit = part.sort_values('steps').tail(3)
            record = {'method': method, 'theoretical_order': {'Euler': 1, 'Heun': 2, 'RK4': 4}[method],
                      'fit_steps': ','.join(map(str, fit.steps)), 'observed_slope': None, 'r_squared': None}
            if len(fit) >= 3:
                x, y = np.log(fit.h), np.log(fit.median_rmse)
                p = np.polyfit(x, y, 1)
                record.update(observed_slope=float(p[0]), r_squared=float(1-np.square(y-np.polyval(p,x)).sum()/np.square(y-y.mean()).sum()))
            orders.append(record)
        order_table = pd.DataFrame(orders)
        order_table.to_csv(self.out/'flow_observed_orders.csv', index=False)
        save_json({'fit_rule': 'last three N>=20 with median error >10x median reference resolution; not proof of asymptotic regime',
                   'orders': orders}, self.out/'order_fit_details.json')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for method in STAGES:
            part = table[table.method == method]
            axes[0].loglog(part.h, part.median_rmse, 'o-', color=COLORS[method], label=method)
            axes[0].fill_between(part.h, part.q25, part.q75, color=COLORS[method], alpha=.15)
            axes[1].semilogx(part.steps, part.label_agreement, 'o-', color=COLORS[method], label=method)
        axes[0].axhline(floor, color='gray', linestyle='--', label='Reference resolution (diagnostic)')
        axes[0].set(xlabel='Step size h = 1/N', ylabel='Unclipped 784D pixel RMSE', title='Numerical convergence: median and IQR')
        axes[1].axhline(float(self.agreement(self.reference_images).float().mean()), color='gray', linestyle='--', label='Reference agreement')
        axes[1].set(xlabel='Number of steps', ylabel='Classifier label agreement', title='Auxiliary conditioning proxy', ylim=(0.,1.0))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.,decimals=0))
        for ax in axes:
            ax.grid(alpha=.2, which='both'); ax.legend(fontsize=8)
        fig.tight_layout()
        paths = [figure_file(fig, self.out, '05_flow_convergence')]
        # Actual cross-solver differences use identical initial states and labels.
        fig, ax = plt.subplots(figsize=(7, 4))
        cross_rows = []
        for a, b in [('Euler','Heun'), ('Heun','RK4'), ('Euler','RK4')]:
            values = []
            for count in steps:
                error = rmse(self.case(a,count)['images'], self.case(b,count)['images'])
                values.append(float(np.median(error)))
                cross_rows.append({'solver_a':a,'solver_b':b,'steps':count,'median_pairwise_rmse':values[-1]})
            ax.loglog(steps, values, 'o-', label=f'{a} vs {b}')
        ax.set(xlabel='Same number of steps N', ylabel='Median endpoint RMSE', title='Agreement across solvers (not a consistency proof)')
        ax.legend(); ax.grid(alpha=.2, which='both'); fig.tight_layout()
        pd.DataFrame(cross_rows).to_csv(self.out/'cross_solver_differences.csv',index=False)
        paths.append(figure_file(fig,self.out,'06_cross_solver_agreement'))
        return table, order_table, paths

    def image_panels(self, digit=3, counts=(1, 2, 3, 5, 10)):
        # First predeclared noise for the requested class; never choose by quality.
        counts = list(counts)
        if not counts or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in counts):
            raise ValueError('Image panel steps must be positive integers')
        ix = int(torch.where(self.labels == digit)[0][0])
        fig, axes = plt.subplots(3, len(counts)+1, figsize=(1.65*(len(counts)+1), 5))
        for row, method in enumerate(STAGES):
            for col, count in enumerate([*counts, self.reference_steps]):
                image = self.reference_images[ix] if col == len(counts) else self.case(method,count)['images'][ix]
                ax = axes[row,col]; ax.imshow(image[0],cmap='gray',vmin=-1,vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0: ax.set_title('Reference' if col == len(counts) else f'N={count}',fontsize=10)
                if col == 0: ax.set_ylabel(method)
        fig.suptitle(f'Same noise + label {digit}: very few integration steps (not equal NFE)')
        fig.tight_layout()
        images = figure_file(fig,self.out,'07_same_noise_image_grid')
        diffs = [[(self.case(m,n)['images'][ix]-self.reference_images[ix]).abs()[0].numpy() for n in counts] for m in STAGES]
        vmax = max(float(a.max()) for row in diffs for a in row)
        fig, axes = plt.subplots(3,len(counts),figsize=(1.65*len(counts)+1.3,5),squeeze=False)
        for row, method in enumerate(STAGES):
            for col,count in enumerate(counts):
                ax=axes[row,col]
                im=ax.imshow(np.maximum(diffs[row][col],1e-6),cmap='magma',norm=LogNorm(vmin=1e-6,vmax=vmax))
                ax.set_xticks([]);ax.set_yticks([])
                if row==0: ax.set_title(f'N={count}',fontsize=10)
                if col==0: ax.set_ylabel(method)
        fig.suptitle('Error maps against the same refined reference')
        fig.subplots_adjust(left=.065,right=.84,top=.86,bottom=.05,wspace=.13,hspace=.15)
        color_axis = fig.add_axes([.88,.12,.015,.67])
        fig.colorbar(im,cax=color_axis,label='Absolute pixel error (log scale; floor 1e-6)')
        errors = figure_file(fig,self.out,'08_pixel_error_maps')
        fig,axes=plt.subplots(2,10,figsize=(13,3.4))
        for row in range(2):
            for digit in range(10):
                ix=int(torch.where(self.labels==digit)[0][row])
                axes[row,digit].imshow(self.reference_images[ix,0],cmap='gray',vmin=-1,vmax=1)
                axes[row,digit].set_axis_off()
                if row==0: axes[row,digit].set_title(f'y={digit}')
        fig.suptitle('Reference samples: first two noises for every label, no selection')
        fig.tight_layout()
        return [images,errors,figure_file(fig,self.out,'09_reference_samples')]

    def efficiency(self, budgets=(20,40,80,160,320), repeats=3):
        path=self.out/'timing_runs.csv'
        cases=[(m,b//STAGES[m],b) for m in STAGES for b in budgets]
        if path.exists() and not self.recompute:
            timing=pd.read_csv(path)
            print('Reusing saved timing runs; RECOMPUTE=True measures runtime again.',flush=True)
            if set(zip(timing.method,timing.nfe)) != {(m,b) for m,_,b in cases} or timing['repeat'].nunique()!=repeats:
                raise RuntimeError('Timing settings changed; use RECOMPUTE=True or another output folder')
        else:
            # Warm up all stages before measuring. Shuffle cases within each round.
            for method in STAGES:
                solve_model(self.model,self.initial,self.labels,2,method)
            rng=np.random.default_rng(622104)
            rows=[]
            for repeat in range(repeats):
                for order,k in enumerate(rng.permutation(len(cases))):
                    method,count,budget=cases[k]
                    result,seconds,calls=solve_model(self.model,self.initial,self.labels,count,method)
                    assert calls==budget
                    assert torch.allclose(result,self.case(method,count)['images'],atol=2e-5,rtol=2e-5)
                    rows.append({'repeat':repeat,'run_order':order,'method':method,'steps':count,
                                 'nfe':budget,'seconds_per_batch':seconds,'batch_size':len(self.labels)})
                print(f'Timing round {repeat+1}/{repeats} completed (randomized order, synchronized GPU).',flush=True)
            timing=pd.DataFrame(rows);timing.to_csv(path,index=False)
        times=timing.groupby(['method','steps','nfe']).agg(
            median_seconds=('seconds_per_batch','median'), min_seconds=('seconds_per_batch','min'),
            max_seconds=('seconds_per_batch','max')).reset_index()
        table=self.table.merge(times,on=['method','steps','nfe'],how='inner')
        table.to_csv(self.out/'equal_nfe_comparison.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for method in STAGES:
            part=table[table.method==method].sort_values('nfe')
            axes[0].loglog(part.nfe,part.median_rmse,'o-',color=COLORS[method],label=method)
            axes[1].loglog(part.median_seconds,part.median_rmse,'o-',color=COLORS[method],label=method)
            axes[2].errorbar(part.nfe,part.median_seconds,
                            yerr=[part.median_seconds-part.min_seconds,part.max_seconds-part.median_seconds],
                            fmt='o-',color=COLORS[method],label=method)
        axes[0].set(xlabel='Network evaluations (NFE)',ylabel='Median raw pixel RMSE',title='Error at equal NFE')
        axes[1].set(xlabel=f'Seconds per batch of {len(self.labels)} images',ylabel='Median raw pixel RMSE',title='Measured error versus runtime')
        axes[2].set(xlabel='NFE',ylabel='Batch seconds: median and min/max',title=f'{repeats} timing repetitions')
        axes[0].set_xticks(budgets,[str(v) for v in budgets])
        axes[0].xaxis.set_minor_formatter(NullFormatter())
        axes[1].set_xticks([.2,.5,1.,2.],['0.2','0.5','1','2'])
        axes[1].xaxis.set_minor_formatter(NullFormatter())
        for ax in axes: ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout()
        plot=figure_file(fig,self.out,'10_accuracy_cost_tradeoff')
        ix=int(torch.where(self.labels==3)[0][0])
        fig,axes=plt.subplots(3,len(budgets),figsize=(9,5.2))
        for row,method in enumerate(STAGES):
            for col,budget in enumerate(budgets):
                count=budget//STAGES[method]
                ax=axes[row,col];ax.imshow(self.case(method,count)['images'][ix,0],cmap='gray',vmin=-1,vmax=1)
                ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f'{count} steps',fontsize=9)
                if col==0: ax.set_ylabel(method)
                if row==0: ax.set_title(f'NFE={budget}\n{count} steps',fontsize=10)
        fig.suptitle('Equal network-call budgets; same noise and label 3')
        fig.tight_layout()
        return table,[plot,figure_file(fig,self.out,'11_equal_budget_images')]

    def low_nfe_comparison(self, budgets=(4, 8, 12, 20, 40), digit=3):
        """Paired low-budget comparison; reuse the validated reference and caches."""
        if self.uncertainty is None:
            raise RuntimeError('Validate the reference first')
        budgets = list(budgets)
        if not budgets or len(set(budgets)) != len(budgets) or any(
                not isinstance(b, int) or isinstance(b, bool) or b <= 0 or b % 4 for b in budgets):
            raise ValueError('Equal budgets for all three methods must be positive multiples of 4')
        if digit not in self.labels.tolist():
            raise ValueError('Requested digit is absent from the fixed noise bank')
        rows, spreads = [], []
        for budget in budgets:
            for method, stages in STAGES.items():
                count = budget//stages
                result = self.case(method, count)
                assert result['nfe'] == budget
                errors = rmse(result['images'], self.reference_images)
                hits = self.agreement(result['images']).numpy()
                for i, (error, hit) in enumerate(zip(errors, hits)):
                    rows.append({'method': method, 'nfe': budget, 'steps': count,
                                 'sample': i, 'label': int(self.labels[i]),
                                 'rmse': float(error), 'label_correct': bool(hit)})
                for label in self.labels.unique():
                    mask = self.labels == label
                    value = result['images'][mask].double().var(0, unbiased=False).mean().sqrt()
                    ref = self.reference_images[mask].double().var(0, unbiased=False).mean().sqrt()
                    spreads.append({'method': method, 'nfe': budget, 'label': int(label),
                                    'rms_pixel_std': float(value), 'reference_rms_pixel_std': float(ref),
                                    'spread_ratio': float(value/ref) if ref > 0 else np.nan})
        frame = pd.DataFrame(rows)
        table = frame.groupby(['nfe', 'method', 'steps'], sort=False).agg(
            median_rmse=('rmse', 'median'), mean_rmse=('rmse', 'mean'),
            q25=('rmse', lambda v: v.quantile(.25)), q75=('rmse', lambda v: v.quantile(.75)),
            worst_rmse=('rmse', 'max'), label_agreement=('label_correct', 'mean')).reset_index()
        frame.to_csv(self.out/'low_nfe_errors_per_image.csv', index=False)
        table.to_csv(self.out/'low_nfe_comparison.csv', index=False)
        pd.DataFrame(spreads).to_csv(self.out/'low_nfe_class_spread.csv', index=False)
        save_json({'config_id': self.config_id, 'budgets': budgets, 'samples': len(self.labels),
                   'reference_steps': self.reference_steps, 'display_digit': digit,
                   'sample_rule': 'first fixed noise for main grid; first six for multi-noise grid',
                   'spread_note': 'Raw pixel variation is descriptive; noise also increases it.'},
                  self.out/'low_nfe_configuration.json')
        indices = torch.where(self.labels == digit)[0]
        ix = int(indices[0])
        fig, axes = plt.subplots(3, len(budgets)+1, figsize=(1.8*(len(budgets)+1), 5.4))
        for row, (method, stages) in enumerate(STAGES.items()):
            for col in range(len(budgets)+1):
                is_ref = col == len(budgets)
                count = None if is_ref else budgets[col]//stages
                image = self.reference_images[ix] if is_ref else self.case(method, count)['images'][ix]
                ax = axes[row, col]
                ax.imshow(image[0], cmap='gray', vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title('Reference' if is_ref else f'NFE={budgets[col]} / {count} steps', fontsize=9)
                if col == 0:
                    ax.set_ylabel(method)
        fig.suptitle(f'Equal NFE: same noise and label {digit}')
        fig.tight_layout()
        paths = [figure_file(fig, self.out, '14_low_nfe_images')]
        # Show the first six noises at the smallest budget, without quality-based selection.
        budget, indices = min(budgets), indices[:6]
        fig, axes = plt.subplots(4, len(indices), figsize=(1.35*len(indices), 5.8), squeeze=False)
        for row, method in enumerate([*STAGES, 'Reference']):
            images = self.reference_images if method == 'Reference' else self.case(method, budget//STAGES[method])['images']
            for col, index in enumerate(indices):
                ax = axes[row, col]
                ax.imshow(images[index, 0], cmap='gray', vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f'Noise {col+1}', fontsize=10)
                if col == 0:
                    ax.set_ylabel(method if method == 'Reference' else f'{method}\n{budget//STAGES[method]} steps')
        fig.suptitle(f'Equal NFE={budget}: first six fixed noises, label {digit}')
        fig.tight_layout()
        paths.append(figure_file(fig, self.out, '15_low_nfe_multiple_noises'))
        return table, paths

    def finish(self):
        table=pd.read_csv(self.out/'equal_nfe_comparison.csv')
        winners=[]
        for nfe,part in table.groupby('nfe'):
            best=part.loc[part.median_rmse.idxmin()]
            winners.append({'nfe':int(nfe),'smallest_median_error_method':best.method,
                            'median_rmse':float(best.median_rmse)})
        summary={'samples':len(self.labels),'samples_per_class':len(self.labels)//10,
                 'frozen_snapshot':SNAPSHOT.name,'reference_steps':self.reference_steps,
                 'reference_check':self.reference_diagnostics,
                 'empirical_best_at_each_budget':winners,'device':str(self.device),
                 'precision':'float32','test_data_used':False,'training_performed':False,
                 'limitations':['Reference refinement is a diagnostic, not a rigorous error bound.',
                                'Classifier agreement is not numerical accuracy or a complete quality metric.',
                                'Linear test stiffness does not establish stiffness of the learned Flow.',
                                'Runtime is batch throughput on this machine; timing repeats are not new image samples.']}
        save_json(summary,self.out/'run_summary.json')
        return summary
