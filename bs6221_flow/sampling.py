"""Inference APIs; importing this module never loads weights or starts training."""
from pathlib import Path
import copy, json, math, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from .models import ConditionalTinyUNet
from .runtime import ROOT, choose_device, synchronize, sha256
OUT = ROOT / 'outputs/demo'
STAGES = {'Euler': 1, 'Heun': 2, 'RK4': 4}
REFERENCE_TOL = 1e-5
TIMING_REPEATS = 2

def default_model():
    return load_flow()

def ode_step(f, t, x, h, method):
    if method not in STAGES:
        raise ValueError(f'未知求解器 {method}')
    k1 = f(t, x)
    if method == 'Euler':
        return x + h * k1
    if method == 'Heun':
        k2 = f(t + h, x + h * k1)
        return x + h * (k1 + k2) / 2
    k2 = f(t + h / 2, x + h * k1 / 2)
    k3 = f(t + h / 2, x + h * k2 / 2)
    k4 = f(t + h, x + h * k3)
    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def integrate(f, initial, steps, method):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('步数必须是正整数')
    if method not in STAGES:
        raise ValueError('未知求解器')
    state = initial.clone() if torch.is_tensor(initial) else np.array(initial, copy=True)
    for n in range(steps):
        state = ode_step(f, n / steps, state, 1 / steps, method)
    return state

@torch.inference_mode()
def sample(net, z, y, steps, method, record=False):
    parameter = next(net.parameters())
    d, dtype = (parameter.device, parameter.dtype)
    state, labels = (z.to(device=d, dtype=dtype), y.to(d))
    calls = 0

    def field(t, x):
        nonlocal calls
        calls += 1
        return net(x, torch.full((len(x),), t, device=d, dtype=dtype), labels)
    states = [state.detach().cpu().clone()] if record else []
    synchronize(d)
    start = time.perf_counter()
    if record:
        marks = set(np.linspace(0, steps, 6).round().astype(int)[1:])
        for n in range(steps):
            state = ode_step(field, n / steps, state, 1 / steps, method)
            if n + 1 in marks:
                states.append(state.detach().cpu().clone())
    else:
        state = integrate(field, state, steps, method)
    synchronize(d)
    seconds = time.perf_counter() - start
    if not torch.isfinite(state).all():
        raise RuntimeError('生成状态非有限')
    assert calls == steps * STAGES[method]
    return (state.detach().cpu(), seconds, calls, states)

def image_grid(images, title):
    n = len(images)
    columns = min(8, n)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, x in zip(axes.flat, images):
        ax.imshow(x[0], cmap='gray', vmin=-1, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} 必须是至少 {minimum} 的整数')
    return int(value)

def _method(value):
    names = {name.lower(): name for name in STAGES}
    if not isinstance(value, str) or value.lower() not in names:
        raise ValueError('method 请选择 Euler、Heun 或 RK4')
    return names[value.lower()]

def load_flow(weights=None, device_preference='auto'):
    """加载附带或自己训练的权重；返回模型，不修改当前全局模型。"""
    path = ROOT/'checkpoints/frozen/20261002_flow15000_ddpm20000/flow_ema.pt' if weights is None else Path(weights)
    if not path.is_absolute():
        path = ROOT / path
    ck = torch.load(path, map_location='cpu', weights_only=True)
    config = ck.get('config', {})
    if ck.get('kind') != 'flow' or config.get('architecture') != 'conditional_tiny_unet_v1' or config.get('channels') != 24:
        raise ValueError('权重与本 notebook 的模型结构不匹配')
    net = ConditionalTinyUNet(24)
    net.load_state_dict(ck['state_dict'], strict=True)
    net = net.to(choose_device(device_preference)).eval().requires_grad_(False)
    net.flow_weight_sha256 = sha256(path)
    return net

def _initial_state(digit, n, seed, noise_scale, noise=None):
    n = _integer(n, 'n')
    seed = _integer(seed, 'seed', 0)
    if seed > 2 ** 63 - 1:
        raise ValueError('seed 必须小于 2**63')
    if not isinstance(noise_scale, (int, float)) or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError('noise_scale 必须是有限正数，标准高斯噪声使用 1.0')
    if isinstance(digit, str) and digit == 'all':
        y = torch.arange(n, dtype=torch.long) % 10
    elif isinstance(digit, (int, np.integer)) and (not isinstance(digit, (bool, np.bool_))):
        if not 0 <= digit <= 9:
            raise ValueError('digit 必须为 0–9')
        y = torch.full((n,), int(digit), dtype=torch.long)
    else:
        values = np.asarray(digit)
        if values.shape != (n,) or values.dtype.kind not in 'iu' or np.any((values < 0) | (values > 9)):
            raise ValueError('digit 也可为长度等于 n、数值为 0–9 的整数列表')
        y = torch.tensor(values.tolist(), dtype=torch.long)
    if noise is None:
        z = torch.randn(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed)) * noise_scale
    else:
        if noise_scale != 1.0:
            raise ValueError('提供 noise 时，请保持 noise_scale=1，直接修改 noise 本身')
        z = torch.as_tensor(noise, device='cpu', dtype=torch.float32).clone()
        if z.shape != (n, 1, 28, 28) or not torch.isfinite(z).all():
            raise ValueError('noise 必须是有限的 [n,1,28,28] 张量')
    return (z, y)

def _generation_figure(result):
    images, labels = (result['images'], result['labels'])
    columns = min(8, len(images))
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, (ax, x) in enumerate(zip(axes.flat, images)):
        ax.imshow(x[0], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'label={int(labels[i])}, #{i}', fontsize=9)
    p = result['params']
    fig.suptitle(f'{p['method']} | steps={p['steps']} | NFE={p['nfe']} | seed={p['seed']}')
    fig.tight_layout()
    return fig

def show_trajectory(result, index=0):
    """显示实际积分状态；即使只有 1–4 步，时间标签也与保存状态一致。"""
    index = _integer(index, 'index', 0)
    if not result['trajectory']:
        raise ValueError('生成时设置 trajectory=True 才会保存轨迹')
    if index >= len(result['images']):
        raise ValueError('index 超过样本数量')
    states, times = (result['trajectory'], result['trajectory_times'])
    fig, axes = plt.subplots(1, len(states), figsize=(2 * len(states), 2.5), squeeze=False)
    for ax, state, t in zip(axes[0], states, times):
        ax.imshow(state[index, 0], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f't={t:.3g}')
        ax.axis('off')
    fig.suptitle('Actual ODE trajectory')
    fig.tight_layout()
    plt.show()
    return fig

def generate_images(digit=3, n=8, method='Euler', steps=40, seed=2026, noise_scale=1.0, trajectory=False, show=True, save=False, net=None, noise=None):
    """返回 images/noise/labels/params/trajectory。只调用模型，不训练。

    digit 可为单个数字、'all'（依次循环 0–9），或长度等于 n 的类别列表。
    noise 可提供自定义初值。积分状态保留原值，只有显示灰度范围被限定。
    """
    steps = _integer(steps, 'steps')
    method = _method(method)
    z, y = _initial_state(digit, n, seed, noise_scale, noise)
    net = default_model() if net is None else net
    images, seconds, calls, states = sample(net, z, y, steps, method, record=trajectory)
    indices = [0] + sorted(set(np.linspace(0, steps, 6).round().astype(int)[1:]) - {0})
    times = [float(i / steps) for i in indices] if trajectory else []
    assert len(times) == len(states)
    result = {'images': images, 'noise': z, 'labels': y, 'trajectory': states, 'trajectory_times': times, 'params': {'method': method, 'steps': steps, 'nfe': calls, 'seed': int(seed), 'samples': len(z), 'labels': y.tolist(), 'noise_scale': float(noise_scale), 'custom_noise': noise is not None, 'device': str(next(net.parameters()).device), 'seconds': seconds, 'timing_includes_trajectory_copy': bool(trajectory), 'weight_sha256': getattr(net, 'flow_weight_sha256', None)}}
    if show or save:
        fig = _generation_figure(result)
        if save:
            OUT.mkdir(parents=True, exist_ok=True)
            folder = OUT / f'generated_{time.time_ns()}'
            folder.mkdir()
            fig.savefig(folder / 'images.png', dpi=140, bbox_inches='tight')
            torch.save({k: v for k, v in result.items() if k != 'params'}, folder / 'states.pt')
            (folder / 'params.json').write_text(json.dumps(result['params'], indent=2), encoding='utf-8')
            result['saved_to'] = folder
        if show:
            plt.show()
        else:
            plt.close(fig)
    if show and trajectory:
        show_trajectory(result)
    return result

def compare_solvers(digit=3, n=4, steps=20, nfe_budget=None, seed=2026, noise_scale=1.0, reference_steps=None, show=True, net=None):
    """相同步数或相同 NFE；共享初值和类别。开启近似参考后再报告相对误差。

    reference_steps=(1024,2048) 可开启 CPU float64 RK4 近似参考；耗时较长。
    """
    steps = _integer(steps, 'steps')
    if nfe_budget is not None:
        nfe_budget = _integer(nfe_budget, 'nfe_budget', 4)
        if nfe_budget % 4:
            raise ValueError('相同 NFE 的预算必须为 4 的倍数')
    net = default_model() if net is None else net
    z, y = _initial_state(digit, n, seed, noise_scale)
    reference = None
    reference_gap = None
    diagnostic_passed = None
    if reference_steps is not None:
        levels = tuple((_integer(v, 'reference_steps') for v in reference_steps))
        if len(levels) < 2 or any((a >= b for a, b in zip(levels, levels[1:]))):
            raise ValueError('reference_steps 至少两个正整数，且严格递增')
        ref_net = copy.deepcopy(net).cpu().double().eval().requires_grad_(False)
        refs = [sample(ref_net, z, y, count, 'RK4')[0] for count in levels]
        reference = refs[-1]
        reference_gap = float((refs[-2] - refs[-1]).square().mean().sqrt())
        diagnostic_passed = reference_gap <= REFERENCE_TOL
        if not diagnostic_passed:
            warnings.warn('参考精度诊断未通过；误差仅相对于当前最细近似参考')
    results = {}
    rows = []
    for name, stages in STAGES.items():
        count = steps if nfe_budget is None else nfe_budget // stages
        sample(net, z, y, min(count, 2), name)
        durations = []
        for _ in range(TIMING_REPEATS):
            result = generate_images(digit, n, name, count, seed, 1.0, show=False, net=net, noise=z)
            result['params']['noise_scale'] = float(noise_scale)
            durations.append(result['params']['seconds'])
        row = {'method': name, 'steps': count, 'nfe': count * stages, 'seconds': float(np.median(durations))}
        if reference is not None:
            row['rmse_to_approx_reference'] = float((result['images'].double() - reference).square().mean().sqrt())
        results[name] = result
        rows.append(row)
    table = pd.DataFrame(rows)
    comparison = {'table': table, 'results': results, 'reference': reference, 'reference_gap': reference_gap, 'reference_diagnostic_passed': diagnostic_passed, 'noise': z, 'labels': y, 'comparison_mode': 'same_steps' if nfe_budget is None else 'same_nfe'}
    if show:
        display(table.round(6))
        columns = min(n, 8)
        fig, axes = plt.subplots(3, columns, figsize=(1.5 * columns, 5), squeeze=False)
        for row, (name, result) in enumerate(results.items()):
            for col in range(columns):
                ax = axes[row, col]
                ax.imshow(result['images'][col, 0], cmap='gray', vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(name)
                ax.set_title(f'label={int(y[col])}', fontsize=9)
        fig.suptitle(f'{comparison['comparison_mode']} | seed={seed} | same initial noise')
        fig.tight_layout()
        plt.show()
        if reference is not None:
            print(f'参考加密差异: {reference_gap:.3g}; 诊断通过: {diagnostic_passed}')
    return comparison

def solve_ode(f, y0, t_span=(0.0, 1.0), steps=100, method='RK4'):
    """任意标量/向量 NumPy ODE，f(t,y) 返回相同形状的导数。

    返回 times、states、nfe；保留全部状态，空间 O(steps × 状态维数)。
    时间为 O(steps × 每步阶段数 × 单次 f 的成本)。
    """
    steps = _integer(steps, 'steps')
    method = _method(method)
    if len(t_span) != 2 or not np.isfinite(t_span).all() or t_span[0] == t_span[1]:
        raise ValueError('t_span 必须为两个不同的有限时间点')
    y = np.array(y0, dtype=float, copy=True)
    if not y.size or not np.isfinite(y).all():
        raise ValueError('y0 必须非空且有限')
    times = np.linspace(float(t_span[0]), float(t_span[1]), steps + 1)
    h = (times[-1] - times[0]) / steps
    states = [y.copy()]

    def field(t, state):
        value = np.asarray(f(t, state), dtype=float)
        if value.shape != y.shape or not np.isfinite(value).all():
            raise ValueError('f 返回的导数必须与 y0 同形状且有限')
        return value
    for t in times[:-1]:
        y = ode_step(field, t, y, h, method)
        if not np.isfinite(y).all():
            raise RuntimeError('积分发散产生非有限值，请检查方程与步长')
        states.append(y.copy())
    return {'times': times, 'states': np.stack(states), 'nfe': steps * STAGES[method], 'method': method}
