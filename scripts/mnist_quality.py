"""Paired MNIST generation-quality benchmark for frozen Flow samplers.

Generation: O(samples * NFE * network cost), in batches of 100. Metrics:
O(classes * samples_per_class**2 * feature_dim) time and O(samples_per_class**2)
working memory. Features and all raw generated endpoints are saved for auditing.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from torchvision import datasets
import flow_numerics as nm
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, PercentFormatter

DEFAULT_OUT = ROOT/'outputs/numerical_analysis/quality_v1'
BUDGETS = (4, 8, 20, 40, 80, 160)
FEATURE_DIM = 64


def mmd_u(kxx, kyy, kxy):
    """Unbiased two-sample MMD squared; finite-sample estimates may be negative."""
    n, m = len(kxx), len(kyy)
    if min(n, m) < 2 or kxy.shape != (n, m):
        raise ValueError('MMD needs at least two samples in each group and matching kernels')
    return float((kxx.sum()-np.trace(kxx))/(n*(n-1))
                 +(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean())


def distribution_statistics(fake, real, bandwidth_sq, k=3):
    """RBF MMD, kNN feature precision/recall, and a descriptive variance ratio."""
    fake, real = np.asarray(fake, dtype=np.float64), np.asarray(real, dtype=np.float64)
    if min(len(fake), len(real)) <= k or bandwidth_sq <= 0:
        raise ValueError('Too few samples or invalid RBF bandwidth')
    if not np.isfinite(fake).all() or not np.isfinite(real).all():
        raise ValueError('Nonfinite features')
    dff, drr, dfr = cdist(fake, fake, 'sqeuclidean'), cdist(real, real, 'sqeuclidean'), cdist(fake, real, 'sqeuclidean')
    value = mmd_u(np.exp(-dff/(2*bandwidth_sq)), np.exp(-drr/(2*bandwidth_sq)),
                  np.exp(-dfr/(2*bandwidth_sq)))
    # Exclude each point itself when defining the k-th-neighbour radius.
    np.fill_diagonal(dff, np.inf); np.fill_diagonal(drr, np.inf)
    rf = np.partition(dff, k-1, axis=1)[:, k-1]
    rr = np.partition(drr, k-1, axis=1)[:, k-1]
    precision = float((dfr <= rr[None, :]).any(axis=1).mean())
    recall = float((dfr <= rf[:, None]).any(axis=0).mean())
    variance_ratio = float(fake.var(axis=0).sum()/real.var(axis=0).sum())
    return {'mmd2': value, 'feature_precision': precision, 'feature_recall': recall,
            'feature_variance_ratio': variance_ratio}


def verify_metric_implementation():
    rng = np.random.default_rng(1904)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(9, 3))
    kernel = lambda a, b: np.exp(-np.square(a-b).sum()/2)
    expected = (sum(kernel(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j)/42
                +sum(kernel(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j)/72
                -2*sum(kernel(a, b) for a in x for b in y)/63)
    actual = distribution_statistics(x, y, 1.)['mmd2']
    assert abs(actual-expected) < 1e-12
    same = distribution_statistics(x, x, 1.)
    assert same['feature_precision'] == same['feature_recall'] == 1.
    collapsed = np.zeros((7, 3))
    assert distribution_statistics(collapsed, y, 1.)['feature_recall'] == 0.
    assert distribution_statistics(x+100, x, 1.)['mmd2'] > .1


@torch.inference_mode()
def extract_features(classifier, images, batch=256):
    """Use the frozen 64D penultimate ReLU; clipping is an image-output transform."""
    device = next(classifier.parameters()).device
    features, predictions = [], []
    for x in images.split(batch):
        x = x.to(device).clamp(-1, 1)
        h = classifier.net[:-1](x)
        logits = classifier.net[-1](h)
        if not features:
            assert h.shape[1] == FEATURE_DIM
            assert torch.allclose(logits, classifier(x), atol=1e-6, rtol=1e-6)
        features.append(h.cpu()); predictions.append(logits.argmax(1).cpu())
    return torch.cat(features), torch.cat(predictions)


def balanced_indices(targets, pool, per_class, seed, offset=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(targets)
    selected = []
    for digit in range(10):
        candidate = np.asarray(pool)[y[np.asarray(pool)] == digit]
        candidate = rng.permutation(candidate)
        if len(candidate) < per_class+offset:
            raise ValueError(f'Too few examples of class {digit}')
        selected.append(candidate[offset:offset+per_class])
    # Classes 0..9 interleaved, so each block of 1000 has 100 examples per class.
    return torch.as_tensor(np.stack(selected).T.reshape(-1), dtype=torch.long)


class QualityBenchmark:
    def __init__(self, out=DEFAULT_OUT, stage='final'):
        if stage not in ('pilot', 'final'):
            raise ValueError('stage must be pilot or final')
        self.out = Path(out) / stage
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out/'samples'; self.cache.mkdir(exist_ok=True)
        self.stage, self.blocks = stage, 1 if stage == 'pilot' else 5
        self.per_class, self.batch = self.blocks*100, 100
        self.budgets = (4,) if stage == 'pilot' else BUDGETS
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.set_num_threads(4)
        self.model, self.classifier, manifest = nm.load_snapshot(self.device)
        code = '\n'.join(inspect.getsource(f) for f in
                         (nm.ode_step, nm.integrate, nm.solve_model, mmd_u, distribution_statistics, extract_features))
        self.config = {
            'version': 1, 'stage': stage, 'samples_per_setting': 10*self.per_class,
            'independent_balanced_blocks': self.blocks, 'samples_per_block': 1000,
            'methods': list(nm.STAGES), 'nfe_budgets': list(self.budgets), 'batch_size': self.batch,
            'noise_seed': 6221451 if stage == 'pilot' else 6221452,
            'real_selection_seed': 6221453, 'calibration_seed': 6221454,
            'run_order_seed': 6221455, 'knn_k': 3, 'device': str(self.device), 'dtype': 'float32',
            'weights_sha256': manifest['files_sha256'], 'implementation_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'split_sha256': nm.sha256(ROOT/'data/mnist_split_seed42.npz'),
            'raw_data_sha256': {p.name: nm.sha256(p) for p in sorted((ROOT/'data/MNIST/raw').glob('*ubyte'))},
            'feature_space': 'frozen MNIST classifier, penultimate 64D ReLU; not Inception FID/KID',
            'kernel': 'RBF exp(-squared_distance/(2*bandwidth_squared)); per-class bandwidth_squared is median squared training-feature distance',
            'bandwidth_calibration': '500 training images per class, fixed before evaluation',
            'primary_mmd': 'mean of class-conditional unbiased MMD squared; report five disjoint block estimates and full-sample estimate',
            'spread_display': 'mean +/- 1 sample SD across disjoint 1000-image blocks; not CI or training-seed uncertainty',
            'test_data_used': stage == 'final', 'training_performed': False,
            'real_source': 'official training validation split' if stage == 'pilot' else 'official MNIST test set',
            'generation': 'unclipped ODE state; clip [-1,1] only for features, classification and images',
            'timing': 'GPU-synchronized integration only; per 100-image batch after warm-up, no feature extraction or I/O',
        }
        path = self.out/'protocol.json'
        if path.exists() and json.loads(path.read_text()) != self.config:
            raise RuntimeError('Protocol changed; preserve old run and use another output directory')
        nm.save_json(self.config, path)
        self.config_id = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.labels = torch.arange(10).repeat(self.per_class)
        self.noise = torch.randn(len(self.labels), 1, 28, 28,
                                 generator=torch.Generator().manual_seed(self.config['noise_seed']))
        nm.save_tensor({'noise': self.noise, 'labels': self.labels, 'config_id': self.config_id}, self.out/'noise_bank.pt')
        self.prepare_real_data()

    def prepare_real_data(self):
        path = self.out/'real_features.pt'
        if path.exists():
            data = torch.load(path, map_location='cpu', weights_only=True)
            assert data['config_id'] == self.config_id
        else:
            train = datasets.MNIST(ROOT/'data', train=True, download=False)
            split = np.load(ROOT/'data/mnist_split_seed42.npz')
            tr, va = split['train'], split['validation']
            assert len(tr) == 55000 and len(va) == 5000
            assert len(np.unique(np.r_[tr, va])) == 60000
            calibration_ix = balanced_indices(train.targets, tr, 500, self.config['calibration_seed'])
            calibration_images = train.data[calibration_ix].unsqueeze(1).float()/127.5-1
            calibration, _ = extract_features(self.classifier, calibration_images)
            bandwidths = []
            for digit in range(10):
                f = calibration[train.targets[calibration_ix] == digit].double().numpy()
                value = float(np.median(pdist(f, metric='sqeuclidean')))
                assert np.isfinite(value) and value > 0
                bandwidths.append(value)
            prototype = torch.stack([train.data[tr[train.targets[tr].numpy() == digit]].float().mean(0)/127.5-1
                                     for digit in range(10)]).unsqueeze(1)
            proto_features, proto_predictions = extract_features(self.classifier, prototype)
            dataset = train if self.stage == 'pilot' else datasets.MNIST(ROOT/'data', train=False, download=False)
            pool = va if self.stage == 'pilot' else np.arange(len(dataset))
            assert self.stage == 'pilot' or len(dataset) == 10000
            real_ix = balanced_indices(dataset.targets, pool, self.per_class, self.config['real_selection_seed'])
            control_count = 100 if self.stage == 'pilot' else 300
            control_ix = balanced_indices(dataset.targets, pool, control_count, self.config['real_selection_seed'], self.per_class)
            assert len(np.intersect1d(real_ix, control_ix)) == 0
            real_images = dataset.data[real_ix].unsqueeze(1).float()/127.5-1
            control_images = dataset.data[control_ix].unsqueeze(1).float()/127.5-1
            real_features, real_predictions = extract_features(self.classifier, real_images)
            control_features, control_predictions = extract_features(self.classifier, control_images)
            assert torch.equal(dataset.targets[real_ix], self.labels)
            data = {'config_id': self.config_id, 'real_features': real_features, 'real_predictions': real_predictions,
                    'real_images': real_images, 'real_indices': real_ix, 'real_labels': dataset.targets[real_ix],
                    'control_features': control_features, 'control_predictions': control_predictions,
                    'control_labels': dataset.targets[control_ix], 'control_indices': control_ix,
                    'prototype_images': prototype, 'prototype_features': proto_features, 'prototype_predictions': proto_predictions,
                    'calibration_indices_in_training_set': calibration_ix, 'bandwidths_squared': bandwidths}
            nm.save_tensor(data, path)
        self.real = data
        self.real_accuracy = float((data['real_predictions'] == self.labels).float().mean())
        nm.save_json({'real_reference_samples': len(self.labels), 'disjoint_real_control_samples': len(data['control_labels']),
                      'reference_classifier_accuracy': self.real_accuracy, 'bandwidths_squared': data['bandwidths_squared'],
                      'real_source': self.config['real_source'], 'test_data_used': self.stage == 'final',
                      'training_validation_split_disjoint': True, 'feature_dimension': FEATURE_DIM,
                      'data_identity_note': 'Train and official-test indices belong to different datasets.'}, self.out/'data_audit.json')

    def metrics(self, features, predictions, labels, name, nfe=0):
        features = features.double().numpy()
        labels, predictions = labels.numpy(), predictions.numpy()
        real = self.real['real_features'].double().numpy()
        real_labels = self.real['real_labels'].numpy()
        rows, block_rows = [], []
        for digit in range(10):
            x, y = features[labels == digit], real[real_labels == digit]
            bandwidth = self.real['bandwidths_squared'][digit]
            stat = distribution_statistics(x, y, bandwidth, self.config['knn_k'])
            rows.append({'method': name, 'nfe': nfe, 'digit': digit, **stat,
                         'label_agreement': float((predictions[labels == digit] == digit).mean())})
            for block in range(min(len(x), len(y))//100):
                xs, ys = x[block*100:(block+1)*100], y[block*100:(block+1)*100]
                stat = distribution_statistics(xs, ys, bandwidth, self.config['knn_k'])
                hit = predictions[labels == digit][block*100:(block+1)*100] == digit
                block_rows.append({'method': name, 'nfe': nfe, 'block': block, 'digit': digit,
                                   **stat, 'label_agreement': float(hit.mean())})
        per_class = pd.DataFrame(rows)
        per_block = pd.DataFrame(block_rows).groupby(['method', 'nfe', 'block'], sort=False).mean(numeric_only=True).drop(columns='digit').reset_index()
        metrics = ['mmd2', 'feature_precision', 'feature_recall', 'feature_variance_ratio', 'label_agreement']
        summary = {'method': name, 'nfe': nfe, 'samples': len(features),
                   **{key: float(per_class[key].mean()) for key in metrics},
                   'mmd2_block_mean': float(per_block.mmd2.mean()),
                   'mmd2_block_sd': float(per_block.mmd2.std(ddof=1)) if len(per_block) > 1 else 0.,
                   'blocks': len(per_block)}
        return summary, per_class, per_block

    def controls(self):
        path = self.out/'controls.json'
        if path.exists():
            return json.loads(path.read_text())
        results = []
        r = self.real
        cases = [('Real vs real', r['control_features'], r['control_predictions'], r['control_labels']),
                 ('Class mean', r['prototype_features'][self.labels], r['prototype_predictions'][self.labels], self.labels)]
        noise_features, noise_predictions = extract_features(self.classifier, self.noise)
        cases.append(('Clipped Gaussian noise', noise_features, noise_predictions, self.labels))
        for name, features, predictions, labels in cases:
            summary, _, _ = self.metrics(features, predictions, labels, name)
            results.append(summary)
        assert results[1]['feature_recall'] < .05, 'Collapsed class means should have little coverage'
        assert results[0]['mmd2'] < results[1]['mmd2'], 'Metric failed real-vs-mean sanity check'
        assert results[0]['mmd2'] < results[2]['mmd2'], 'Metric failed real-vs-noise sanity check'
        nm.save_json(results, path)
        pd.DataFrame(results).to_csv(self.out/'controls.csv', index=False)
        return results

    def generate(self, method, budget):
        path = self.cache/f'{method}_nfe{budget:03d}.pt'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['config_id'] == self.config_id
            return payload
        steps = budget//nm.STAGES[method]
        nm.solve_model(self.model, self.noise[:self.batch], self.labels[:self.batch], 2, method)
        images, times = [], []
        for start in range(0, len(self.labels), self.batch):
            stop = start+self.batch
            result, seconds, calls = nm.solve_model(self.model, self.noise[start:stop], self.labels[start:stop], steps, method)
            assert calls == budget
            images.append(result); times.append(seconds)
            if (start//self.batch+1) % 10 == 0:
                print(f'{method} NFE={budget}: {stop}/{len(self.labels)} images', flush=True)
        images = torch.cat(images)
        features, predictions = extract_features(self.classifier, images)
        payload = {'config_id': self.config_id, 'method': method, 'steps': steps, 'nfe': budget,
                   'images': images, 'features': features, 'predictions': predictions,
                   'seconds_per_batch': times, 'batch_size': self.batch,
                   'out_of_range_fraction': float((images.abs() > 1).float().mean())}
        nm.save_tensor(payload, path)
        return payload

    def run(self):
        verify_metric_implementation()
        controls = self.controls()
        cases = [(m, b) for m in nm.STAGES for b in self.budgets]
        order = np.random.default_rng(self.config['run_order_seed']).permutation(len(cases))
        nm.save_json([{'method': cases[i][0], 'nfe': cases[i][1]} for i in order], self.out/'run_order.json')
        summaries, classes, blocks, times = [], [], [], []
        for run_index, i in enumerate(order):
            method, budget = cases[i]
            generated = self.generate(method, budget)
            summary, per_class, per_block = self.metrics(generated['features'], generated['predictions'], self.labels, method, budget)
            summary.update(steps=generated['steps'], median_seconds_per_100=float(np.median(generated['seconds_per_batch'])),
                           total_generation_seconds=float(sum(generated['seconds_per_batch'])),
                           out_of_range_fraction=generated['out_of_range_fraction'])
            summaries.append(summary); classes.append(per_class); blocks.append(per_block)
            times.extend({'method': method, 'nfe': budget, 'run_order': run_index, 'batch': j,
                          'batch_size': self.batch, 'seconds': value} for j, value in enumerate(generated['seconds_per_batch']))
            pd.DataFrame(summaries).to_csv(self.out/'quality_summary.csv', index=False)
            pd.concat(classes).to_csv(self.out/'quality_per_class.csv', index=False)
            pd.concat(blocks).to_csv(self.out/'quality_blocks.csv', index=False)
            pd.DataFrame(times).to_csv(self.out/'generation_times.csv', index=False)
            print(f'Completed {run_index+1}/{len(cases)}: {method}, NFE={budget}, MMD2={summary["mmd2"]:.5f}, '
                  f'precision={summary["feature_precision"]:.3f}, recall={summary["feature_recall"]:.3f}, '
                  f'label={summary["label_agreement"]:.3f}', flush=True)
        report = {'complete': True, 'config_id': self.config_id, 'stage': self.stage,
                  'settings_completed': len(cases), 'samples_per_setting': len(self.labels),
                  'generated_total': len(cases)*len(self.labels), 'nfe_budgets': list(self.budgets),
                  'real_classifier_accuracy': self.real_accuracy, 'controls': controls,
                  'test_data_used': self.stage == 'final', 'model_retrained': False,
                  'limitation': 'One frozen model and one learned feature space. Block variation is sampling variation, not training-seed robustness. Metrics are MNIST feature metrics, not standard FID/KID.'}
        nm.save_json(report, self.out/'report.json')
        render(self.out)
        return report


def render(out=DEFAULT_OUT/'final'):
    out = Path(out)
    t = pd.read_csv(out/'quality_summary.csv')
    controls = pd.read_csv(out/'controls.csv')
    baseline = controls[controls.method == 'Real vs real'].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in nm.STAGES:
        part = t[t.method == method].sort_values('nfe')
        for ax, key in zip(axes, ['nfe', 'median_seconds_per_100']):
            ax.errorbar(part[key], part.mmd2_block_mean, yerr=part.mmd2_block_sd,
                        fmt='o-', capsize=3, color=nm.COLORS[method], label=method)
            ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=.002)
    for ax in axes:
        ax.axhline(baseline.mmd2_block_mean, color='gray', linestyle='--', label='Disjoint real vs real')
        ax.set_ylabel('Class-conditional MNIST-feature MMD squared')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set(xlabel='Network evaluations (NFE)', title='Distribution quality versus network cost')
    axes[0].set_xticks(sorted(t.nfe.unique()), [str(n) for n in sorted(t.nfe.unique())])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set(xlabel='Seconds per 100-image batch', title='Distribution quality versus measured runtime')
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Lower is better; bars: +/- 1 SD across disjoint 1,000-image blocks (not CI)', fontsize=10)
    fig.tight_layout()
    paths = [nm.figure_file(fig, out, '01_quality_cost')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for method in nm.STAGES:
        part = t[t.method == method].sort_values('nfe')
        for ax, metric in zip(axes, ['feature_precision', 'feature_recall', 'label_agreement']):
            ax.semilogx(part.nfe, part[metric], 'o-', color=nm.COLORS[method], label=method)
    for ax, title in zip(axes, ['Feature precision (realism proxy)', 'Feature recall (coverage proxy)', 'Requested-label agreement']):
        ax.set(xlabel='NFE', title=title, ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1.)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.set_xticks(sorted(t.nfe.unique()), [str(n) for n in sorted(t.nfe.unique())]); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout(); paths.append(nm.figure_file(fig, out, '02_quality_coverage_condition'))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    names = ['Real vs real', 'Class mean', 'Clipped Gaussian noise']
    control = controls.set_index('method').loc[names]
    for ax, metric, title in zip(axes, ['mmd2', 'feature_recall', 'label_agreement'],
                               ['MMD squared: lower is better', 'Coverage proxy', 'Label agreement']):
        ax.bar(np.arange(3), control[metric], color=['#666666', '#CC79A7', '#999933'])
        ax.set_xticks(np.arange(3), ['Real vs real', 'Class mean', 'Noise'], fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=.2)
        if metric != 'mmd2':
            ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(PercentFormatter(1.))
    fig.suptitle('Metric controls: recognizable class averages can have no sample coverage', fontsize=11)
    fig.tight_layout(); paths.append(nm.figure_file(fig, out, '03_metric_controls'))
    for budget in [b for b in (4, 20, 160) if b in t.nfe.unique()]:
        fig, axes = plt.subplots(3, 10, figsize=(13, 4.3))
        for row, method in enumerate(nm.STAGES):
            data = torch.load(out/'samples'/f'{method}_nfe{budget:03d}.pt', map_location='cpu', weights_only=True)
            for digit in range(10):
                ax = axes[row, digit]
                ax.imshow(data['images'][digit, 0], cmap='gray', vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0: ax.set_title(f'y={digit}')
                if digit == 0:
                    ax.set_ylabel(f'{method}\n{data["steps"]} steps', rotation=0,
                                  ha='right', va='center', labelpad=12, fontsize=9)
        fig.suptitle(f'NFE={budget}: first fixed noise per class; same noises across methods')
        fig.tight_layout(); paths.append(nm.figure_file(fig, out, f'04_samples_nfe{budget:03d}'))
    nm.save_json([p.name for p in paths], out/'figure_files.json')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'final'], default='final')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        render(args.out/args.stage)
    else:
        result = QualityBenchmark(args.out, args.stage).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
