"""Visualize fixed-start conditional Flow and DDPM using static Python APIs.

Reuses architecture definitions and schedules from the static conditional model module. Runs inference only and retains every generated sample.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(extra_names=()):
    """Compatibility namespace sourced from static Python modules, without dynamic code extraction."""
    import sys
    sys.path.insert(0, str(ROOT))
    from bs6221_flow import models, legacy_samplers
    required = {'TimeEmbedding', 'TimeBlock', 'ConditionalTinyUNet', 'cosine_schedule'} | set(extra_names)
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math}
    for name in required:
        module = models if hasattr(models, name) else legacy_samplers
        namespace[name] = getattr(module, name)
    return namespace


@torch.inference_mode()
def flow_path(model, start, labels, steps=80):
    state = start.clone()
    states = [state.cpu().clone()]
    h = 1 / steps
    for j in range(steps):
        t = torch.full((len(state),), j / steps, device=state.device)
        velocity = model(state, t, labels)
        proposal = state + h * velocity
        state = state + .5 * h * (velocity + model(proposal, t + h, labels))
        states.append(state.cpu().clone())
    return torch.stack(states)


@torch.inference_mode()
def ddpm_paths(model, start, labels, schedule, seed):
    state = start.clone()
    states = [state.cpu().clone()]
    rng = torch.Generator().manual_seed(seed)
    steps = len(schedule['abar'])
    for j in reversed(range(steps)):
        t = torch.full((len(state),), j / (steps - 1), device=state.device)
        eps = model(state, t, labels)
        a = schedule['abar'][j]
        clean = ((state - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        mean = schedule['coef_x0'][j] * clean + schedule['coef_xt'][j] * state
        if j:
            # Independent draws for each branch at each DDPM reverse step.
            noise = torch.randn(state.shape, generator=rng).to(state.device)
            state = mean + schedule['posterior_var'][j].sqrt() * noise
        else:
            state = mean
        states.append(state.cpu().clone())
    return torch.stack(states)


def fit_pca():
    """Same train-only subset and PCA recipe as the original notebook."""
    mnist = datasets.MNIST(ROOT / 'data', train=True, download=False)
    indices = np.load(ROOT / 'data/mnist_split_seed42.npz')['train']
    take = torch.randperm(len(indices), generator=torch.Generator().manual_seed(42))[:3000]
    ids = torch.from_numpy(indices)[take]
    flat = (mnist.data[ids].float() / 127.5 - 1).flatten(1).numpy()
    center = flat.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(flat - center, full_matrices=False)
    basis = vt[:2].T
    return center, basis, (flat - center) @ basis, mnist.targets[ids].numpy(), float((singular[:2] ** 2).sum() / (singular ** 2).sum())


def trajectory_axis(ax, paths, cloud, target_mask, color, title, colors=None):
    ax.scatter(*cloud.T, s=5, c='#cbd5e1', alpha=.28, rasterized=True)
    ax.scatter(*cloud[target_mask].T, s=8, c='#94a3b8', alpha=.45, rasterized=True)
    for i in range(paths.shape[1]):
        p = paths[:, i]
        c = colors[i] if colors is not None else color
        ax.plot(*p.T, color=c, lw=2 if colors is not None else 3, alpha=.85)
        for j in ([20, 45, 70] if len(p) == 101 else [16, 40, 64]):
            ax.annotate('', xy=p[j + 1], xytext=p[j],
                        arrowprops={'arrowstyle': '->', 'color': c, 'lw': 2, 'mutation_scale': 13})
        ax.scatter(*p[-1], s=70, color=c, edgecolor='white', linewidth=1.1, zorder=5)
        if colors is not None:
            ax.annotate(str(i + 1), p[-1], xytext=(6, 5), textcoords='offset points',
                        fontsize=11, weight='bold', color=c,
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    start = paths[0, 0]
    ax.scatter(*start, s=130, marker='*', color='#111827', zorder=6)
    ax.annotate('Same initial noise', start, xytext=(0, -25), textcoords='offset points',
                ha='center', fontsize=10, color='#111827',
                bbox={'boxstyle': 'round,pad=.3', 'fc': 'white', 'ec': '#e2e8f0', 'alpha': .95})
    ax.set(title=title, xlabel='PC1', ylabel='PC2')
    ax.set_aspect('equal', adjustable='box')
    ax.spines[['top', 'right']].set_visible(False)


def show_image(ax, image, title, color=None):
    ax.imshow(image[0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=color or '#334155', pad=5)
    for spine in ax.spines.values():
        spine.set_visible(color is not None)
        if color is not None:
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, default=3, choices=range(10))
    parser.add_argument('--seed', type=int, default=2029)
    parser.add_argument('--step-seed', type=int, default=2027)
    args = parser.parse_args()
    out = ROOT / 'outputs/sampling_branches'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    definitions = load_definitions()
    models, budgets, configs, checkpoint_hashes = {}, {}, {}, {}
    for kind in ['flow', 'diffusion']:
        checkpoint_path = ROOT / f'checkpoints/conditional/conditional_{kind}.pt'
        checkpoint_hashes[kind] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        payload = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        assert payload['kind'] == kind
        model = definitions['ConditionalTinyUNet'](payload['config']['channels'])
        model.load_state_dict(payload['ema'], strict=True)
        models[kind] = model.to(device).eval()
        configs[kind] = payload['config']
        budgets[kind] = payload['completed_steps']
    assert configs['flow'] == configs['diffusion']
    schedule = {k: v.to(device) for k, v in definitions['cosine_schedule'](configs['diffusion']['diffusion_steps']).items()}
    count = 6
    initial = torch.randn((1, 1, 28, 28), generator=torch.Generator().manual_seed(args.seed))
    start = initial.repeat(count, 1, 1, 1).to(device)
    labels = torch.full((count,), args.digit, dtype=torch.long, device=device)
    begin = time.perf_counter()
    flow = flow_path(models['flow'], start, labels)
    diffusion = ddpm_paths(models['diffusion'], start, labels, schedule, args.step_seed)
    replay = ddpm_paths(models['diffusion'], start, labels, schedule, args.step_seed)
    if device.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - begin
    assert torch.isfinite(flow).all() and torch.isfinite(diffusion).all()
    flow_spread = float((flow - flow[:, :1]).abs().max())
    assert flow_spread < 1e-5, flow_spread
    assert torch.allclose(diffusion, replay, atol=1e-5, rtol=1e-5)
    assert torch.equal(flow[0], diffusion[0])
    pairwise = torch.pdist(diffusion[-1].flatten(1)).square() / 784
    assert pairwise.min() > 1e-4
    print(f'{device}: Flow identical across {count} repeats (max difference {flow_spread:.2g}); DDPM branches distinct; replay matches. Inference {elapsed:.2f}s.', flush=True)
    center, basis, cloud, data_labels, explained = fit_pca()
    def project(states):
        return ((states.flatten(2).numpy() - center) @ basis)
    fp, dp = project(flow), project(diffusion)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(16, 9.5), facecolor='white')
    grid = fig.add_gridspec(3, 2, height_ratios=[5, .55, 1.4], hspace=.16, wspace=.17)
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    trajectory_axis(axes[0], fp[:, :1], cloud, data_labels == args.digit, '#6D28D9',
                    'FLOW / one deterministic trajectory')
    trajectory_axis(axes[1], dp, cloud, data_labels == args.digit, None,
                    'DIFFUSION (DDPM) / six stochastic trajectories', colors)
    points = np.concatenate([fp.reshape(-1, 2), dp.reshape(-1, 2), cloud[data_labels == args.digit]])
    low, high = points.min(0), points.max(0)
    padding = np.maximum((high - low) * .12, 1.5)
    for ax in axes:
        ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
        ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    flow_box = AnnotationBbox(OffsetImage(flow[-1, 0, 0].numpy(), cmap='gray', norm=matplotlib.colors.Normalize(-1, 1), zoom=1.5),
                              fp[-1, 0], xybox=(58, 0), boxcoords='offset points',
                              arrowprops={'arrowstyle': '-', 'color': '#6D28D9'},
                              bboxprops={'edgecolor': '#6D28D9', 'linewidth': 1.5})
    axes[0].add_artist(flow_box)
    for j, title in enumerate(['Flow: the same route and result on every repeat',
                                f'DDPM: six final images, all requested as digit {args.digit}']):
        ax = fig.add_subplot(grid[1, j]); ax.axis('off')
        ax.text(.5, .25, title, ha='center', va='center', fontsize=12, weight='bold')
    for j in range(2):
        images_grid = grid[2, j].subgridspec(1, 6, wspace=.15)
        for k in range(6):
            ax = fig.add_subplot(images_grid[0, k])
            if j == 0:
                ix = round(k * (len(flow) - 1) / 5)
                show_image(ax, flow[ix, 0], ['Noise', '20%', '40%', '60%', '80%', 'Final'][k])
            else:
                show_image(ax, diffusion[-1, k], f'Branch {k + 1}', colors[k])
    fig.suptitle(f'ONE INITIAL NOISE + ONE CONDITION (digit {args.digit})', fontsize=20, weight='bold', y=.98)
    fig.text(.5, .925, 'Flow: fixed velocity integration     |     DDPM: independent Gaussian draws at reverse steps',
             ha='center', fontsize=12, color='#475569')
    fig.text(.5, .025, f'Actual model samples; no selection. Shared TRAINING-data PCA ({explained:.1%} variance). Gray points: real images.\n'
             'Trajectories are 2D projections of 784D image states. This illustrates sampling randomness, not relative model quality.',
             ha='center', fontsize=10, color='#64748b')
    fig.subplots_adjust(top=.87, bottom=.10, left=.055, right=.96)
    fig.savefig(out / 'flow_vs_diffusion_branches.png', dpi=160, facecolor='white')
    fig.savefig(out / 'flow_vs_diffusion_branches.pdf', facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(count, 6, figsize=(10.5, 10.5), squeeze=False)
    for row in range(count):
        for column in range(6):
            ix = round(column * (len(diffusion) - 1) / 5)
            show_image(axes[row, column], diffusion[ix, row], '' if row else ['Same noise', '20 steps', '40 steps', '60 steps', '80 steps', 'Final'][column], colors[row])
            if column == 0:
                axes[row, column].set_ylabel(f'Branch {row + 1}', color=colors[row], weight='bold')
    fig.suptitle(f'DDPM: one identical noise image becomes six images / requested digit {args.digit}', fontsize=14, weight='bold')
    fig.text(.5, .015, 'Every row starts from the same pixels. Independent step noises produce different intermediate states and final images.', ha='center', fontsize=10)
    fig.tight_layout(rect=(.02, .04, 1, .96))
    fig.savefig(out / 'diffusion_noise_to_six_images.png', dpi=150, facecolor='white')
    plt.close(fig)

    # Synchronized process animation: time indexes are sampling progress, not
    # the same mathematical time parameter for Flow and DDPM.
    fig, axes = plt.subplots(2, 6, figsize=(10, 4), squeeze=False)
    artists = []
    for row in range(2):
        for col in range(6):
            show_image(axes[row, col], initial[0], f'Repeat {col+1}' if row == 0 else f'Branch {col+1}', '#6D28D9' if row == 0 else colors[col])
            artists.append(axes[row, col].images[0])
        axes[row, 0].set_ylabel(['Flow', 'DDPM'][row], fontsize=12, weight='bold')
    heading = fig.suptitle('Same initial noise / sampling progress 0%', fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    progress = np.concatenate([np.zeros(4), np.linspace(0, 1, 41), np.ones(12)])
    def update(frame):
        p = progress[frame]
        for row, states in enumerate([flow, diffusion]):
            ix = round(p * (len(states) - 1))
            for col in range(6):
                artists[row * 6 + col].set_data(states[ix, col, 0].numpy())
        heading.set_text(f'Same initial noise + requested digit {args.digit} / sampling progress {p:.0%}')
        return artists + [heading]
    animation = FuncAnimation(fig, update, frames=len(progress), interval=180, blit=False)
    animation.save(out / 'same_noise_generation.gif', writer=PillowWriter(fps=6), dpi=90)
    plt.close(fig)
    torch.save({'initial_noise': initial, 'target_digit': args.digit, 'flow_states': flow,
                'diffusion_states': diffusion, 'flow_progress': torch.linspace(0, 1, len(flow)),
                'diffusion_progress': torch.linspace(0, 1, len(diffusion)), 'pca_center': torch.from_numpy(center),
                'pca_basis': torch.from_numpy(basis)}, out / 'actual_trajectories.pt')
    summary = {'purpose': 'same-start deterministic versus stochastic sampling illustration',
               'target_digit': args.digit, 'initial_noise_seed': args.seed, 'ddpm_step_noise_seed': args.step_seed,
               'branches': count, 'same_initial_pixels': True, 'samples_selected': False,
               'device': str(device), 'flow_solver': 'Heun', 'flow_steps': len(flow)-1,
               'ddpm_steps': len(diffusion)-1, 'flow_repeated_path_max_difference': flow_spread,
               'ddpm_pairwise_final_mse_min': float(pairwise.min()),
               'ddpm_pairwise_final_mse_mean': float(pairwise.mean()), 'ddpm_fixed_seed_replay_matches': True,
               'pca_real_data_variance_retained': explained, 'training_steps': budgets,
               'checkpoint_sha256': checkpoint_hashes,
               'quality_comparison': False, 'inference_seconds_including_replay': elapsed,
               'note': 'DDPM uses discrete Gaussian posterior sampling, analogous to stochastic SDE increments. Determinism refers to fixed model, initial state, condition and solver.'}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved figures, animation and all trajectories to {out}. Notebooks unchanged.', flush=True)


if __name__ == '__main__':
    main()
