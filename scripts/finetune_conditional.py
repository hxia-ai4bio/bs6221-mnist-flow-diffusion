"""Continue the existing conditional models with lower learning rates.

No architecture/data/sampler changes. Select EMA checkpoints using 1,024 held-out
images and two fixed noise draws. Generated-image metrics are descriptive only.
All baseline weights and notebook outputs are backed up before training.

Training costs O(updates * batch * network cost), with one batch on the GPU.
Evaluation costs O(samples * sampling steps * network cost); images stay on CPU.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets

from visualize_sampling_branches import ROOT, load_definitions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(value, path):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def save_checkpoint(value, path):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def prepare_run(run):
    run.mkdir(parents=True, exist_ok=True)
    baseline = run/'baseline'
    manifest = run/'baseline_manifest.json'
    if manifest.exists():
        for name, sha in json.loads(manifest.read_text())['checkpoint_sha256'].items():
            if digest(baseline/name) != sha:
                raise ValueError(f'Baseline changed: {name}')
        return
    baseline.mkdir(exist_ok=True)
    hashes = {}
    for kind in ('flow', 'diffusion'):
        name = f'conditional_{kind}.pt'
        target = baseline/name
        if target.exists():
            raise ValueError('Incomplete backup exists; inspect before reusing this run directory')
        shutil.copy2(ROOT/'checkpoints/conditional'/name, target)
        hashes[name] = digest(target)
    shutil.copy2(ROOT/'checkpoints/conditional/digit_classifier.pt', baseline/'digit_classifier.pt')
    for source in ('outputs/conditional', 'outputs/sampling_branches', 'notebooks'):
        shutil.copytree(ROOT/source, baseline/Path(source).name)
    save_json({'checkpoint_sha256': hashes, 'test_set_used': False}, manifest)


class Experiment:
    def __init__(self, run):
        self.run = run
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device.type != 'mps':
            raise RuntimeError('MPS is unavailable. Run with host GPU access; do not silently train on CPU.')
        self.ns = load_definitions({'DigitClassifier', 'sample_flow', 'sample_diffusion', 'validate_labels'})
        self.ns.update(DEVICE=self.device, DIFFUSION_STEPS=100)
        self.schedule = {k: v.to(self.device) for k, v in self.ns['cosine_schedule'](100).items()}
        self.ns['schedule'] = self.schedule
        mnist = datasets.MNIST(ROOT/'data', train=True, download=False)
        split = np.load(ROOT/'data/mnist_split_seed42.npz')
        train_ix = torch.from_numpy(split['train'].astype(np.int64))
        val_ix = torch.from_numpy(split['validation'].astype(np.int64))
        assert len(train_ix) == 55000 and len(val_ix) == 5000
        assert len(np.intersect1d(train_ix.numpy(), val_ix.numpy())) == 0
        self.train_x = mnist.data[train_ix].unsqueeze(1).float()/127.5-1
        self.train_y = mnist.targets[train_ix]
        self.val_x = mnist.data[val_ix[:1024]].unsqueeze(1).float()/127.5-1
        self.val_y = mnist.targets[val_ix[:1024]]
        self.split_hash = digest(ROOT/'data/mnist_split_seed42.npz')
        self.fixed_pairs = {}
        for kind in ('flow', 'diffusion'):
            # CPU generation gives the same noise/time pairs at every validation.
            self.fixed_pairs[kind] = [self.pair(self.val_x, kind, torch.Generator().manual_seed(seed), cpu=True)
                                      for seed in (3701, 3702)]
        self.classifier = self.ns['DigitClassifier']().to(self.device).eval()
        ck = torch.load(run/'baseline/digit_classifier.pt', map_location='cpu', weights_only=True)
        assert ck['config']['split_sha256'] == self.split_hash
        self.classifier.load_state_dict(ck['model'])
        print(f'Device={self.device}; train=55000; selection=1024 images x 2 fixed draws; test unused', flush=True)

    def load(self, path, ema=True):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if ck['config']['split_sha256'] != self.split_hash:
            raise ValueError('Checkpoint uses another data split')
        assert ck['config']['architecture'] == 'conditional_tiny_unet_v1'
        assert ck['config']['diffusion_steps'] == 100
        model = self.ns['ConditionalTinyUNet'](ck['config']['channels']).to(self.device)
        model.load_state_dict(ck['ema' if ema else 'model'], strict=True)
        return model.eval(), ck

    def pair(self, clean, kind, rng, cpu=False):
        device = torch.device('cpu') if cpu else self.device
        clean = clean.to(device)
        noise = torch.randn(clean.shape, generator=rng).to(device)
        if kind == 'flow':
            t = torch.rand(len(clean), generator=rng).to(device)
            return (1-t[:, None, None, None])*noise+t[:, None, None, None]*clean, t, clean-noise
        steps = torch.randint(100, (len(clean),), generator=rng)
        a = self.schedule['abar'].to(device)[steps.to(device)][:, None, None, None]
        return a.sqrt()*clean+(1-a).sqrt()*noise, steps.to(device).float()/99, noise

    @torch.no_grad()
    def validation(self, model, kind, legacy=False):
        model.eval()
        pairs = ([self.pair(self.val_x[:256], kind, torch.Generator().manual_seed(1701), cpu=True)]
                 if legacy else self.fixed_pairs[kind])
        total, count = 0., 0
        for x, t, target in pairs:
            for start in range(0, len(x), 64):
                s = slice(start, start+64)
                predicted = model(x[s].to(self.device), t[s].to(self.device), self.val_y[s].to(self.device))
                assert predicted.shape == target[s].shape and torch.isfinite(predicted).all()
                errors = (predicted-target[s].to(self.device)).square().flatten(1).mean(1)
                total += errors.sum().item()
                count += len(errors)
        return total/count

    def train(self, kind, args):
        latest, best = self.run/f'{kind}_latest.pt', self.run/f'{kind}_best.pt'
        baseline = self.run/f'baseline/conditional_{kind}.pt'
        model, ck = self.load(latest if latest.exists() else baseline, ema=False)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        ema.load_state_dict(ck['ema'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(ck['optimizer'])
        rng = torch.Generator()
        rng.set_state(ck['generator_state'])
        config = {'updates': args.updates, 'lr_start': args.lr, 'lr_end': args.min_lr,
                  'validation_images': 1024, 'validation_seeds': [3701, 3702], 'eval_every': 1000}
        if latest.exists():
            info = copy.deepcopy(ck['fine_tuning'])
            if info['config'] != config:
                raise ValueError('Resume settings must match the original continuation schedule')
        else:
            baseline_val = self.validation(ema, kind)
            info = {'config': config, 'start_step': ck['completed_steps'], 'records': [],
                    'baseline_validation': baseline_val, 'best_validation': baseline_val,
                    'best_step': ck['completed_steps'], 'baseline_sha256': digest(baseline)}
            save_checkpoint({**ck, 'fine_tuning': info}, best)
        start_step = info['start_step']
        target = start_step+min(args.updates, args.stop_after or args.updates)
        if ck['completed_steps'] >= target:
            print(f'{kind}: requested stage already complete', flush=True)
            return
        begin = time.perf_counter()
        history = copy.deepcopy(ck['history'])
        print(f'{kind}: continuing {ck["completed_steps"]} -> {target}; baseline expanded MSE={info["baseline_validation"]:.6f}', flush=True)
        for step in range(ck['completed_steps']+1, target+1):
            progress = (step-start_step-1)/max(args.updates-1, 1)
            lr = args.min_lr+.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            ix = torch.randint(len(self.train_x), (64,), generator=rng)
            x, t, target_noise = self.pair(self.train_x[ix], kind, rng)
            labels = self.train_y[ix].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, t, labels)
            assert prediction.shape == target_noise.shape
            loss = F.mse_loss(prediction, target_noise)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at {kind} {step}')
            loss.backward()
            if step == ck['completed_steps']+1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert model.label_embedding.weight.grad.abs().sum() > 0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for averaged, live in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(live, 1-ck['config']['ema_decay'])
            if step % 250 == 0 or step == target:
                legacy = self.validation(ema, kind, legacy=True)
                history.append({'step': step, 'training_loss': loss.item(), 'validation_loss': legacy,
                                'gradient_norm_before_clip': norm.item(), 'learning_rate': lr})
                torch.mps.synchronize()
                print(f'{kind} {step}/{start_step+args.updates}: legacy val={legacy:.6f}, lr={lr:.2g}, elapsed={time.perf_counter()-begin:.1f}s', flush=True)
            if (step-start_step) % 1000 == 0 or step == target:
                val = self.validation(ema, kind)
                info['records'].append({'step': step, 'validation_loss': val, 'learning_rate': lr})
                improved = val < info['best_validation']
                if improved:
                    info.update(best_validation=val, best_step=step)
                payload = {'kind': kind, 'config': ck['config'], 'completed_steps': step,
                           'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           'ema': {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                           'optimizer': optimizer.state_dict(), 'generator_state': rng.get_state(),
                           'history': history, 'training_seconds': ck['training_seconds']+time.perf_counter()-begin,
                           'fine_tuning': info}
                save_checkpoint(payload, latest)
                if improved:
                    save_checkpoint(payload, best)
                save_json(info, self.run/f'{kind}_training.json')
                print(f'  expanded val={val:.6f}; best={info["best_validation"]:.6f} at {info["best_step"]}', flush=True)
        del model, ema, optimizer
        torch.mps.empty_cache()

    @torch.no_grad()
    def generated(self, model, kind, per_class):
        labels = torch.arange(10).repeat_interleave(per_class)
        noise = torch.randn(len(labels), 1, 28, 28, generator=torch.Generator().manual_seed(42026))
        chunks = []
        for i in range(0, len(labels), 100):
            s = slice(i, i+100)
            if kind == 'flow':
                samples, _ = self.ns['sample_flow'](model, noise[s], labels[s], steps=80)
            else:
                # Fixed chunking and per-chunk step seeds pair before/after paths.
                samples, _ = self.ns['sample_diffusion'](model, noise[s], labels[s], seed=52027+i)
            assert torch.isfinite(samples).all()
            chunks.append(samples)
        images = torch.cat(chunks)
        predictions = torch.cat([self.classifier(x.to(self.device).clamp(-1, 1)).argmax(1).cpu()
                                 for x in images.split(100)])
        metrics = {'label_agreement': float((predictions == labels).float().mean()),
                   'agreement_by_class': [(predictions[labels == d] == d).float().mean().item() for d in range(10)],
                   'within_class_pixel_variance': [images[labels == d].clamp(-1, 1).var(0, unbiased=False).mean().item()
                                                   for d in range(10)]}
        return {'images': images, 'labels': labels, 'predictions': predictions, 'initial_noise': noise}, metrics

    def evaluate(self, args):
        out = self.run/args.tag
        out.mkdir(exist_ok=True)
        summary = {'samples_per_class': args.samples_per_class, 'sample_seed': 42026,
                   'independent_initial_noise_per_image': True, 'test_set_used': False,
                   'checkpoint_selection': 'expanded fixed-noise validation MSE only',
                   'flow_sampler': 'Heun 80', 'diffusion_sampler': 'DDPM 100',
                   'quality_metric_note': 'Classifier agreement and pixel variance are proxies, not a complete quality/diversity benchmark.',
                   'models': {}}
        for kind in ('flow', 'diffusion'):
            samples, metrics = {}, {}
            for version, path in [('before', self.run/f'baseline/conditional_{kind}.pt'),
                                  ('after', self.run/f'{kind}_best.pt')]:
                model, ck = self.load(path)
                samples[version], metrics[version] = self.generated(model, kind, args.samples_per_class)
                metrics[version].update(training_steps=ck['completed_steps'], checkpoint_sha256=digest(path),
                                        validation_loss=self.validation(model, kind))
                print(f'{kind} {version}: {json.dumps(metrics[version])}', flush=True)
                del model
                torch.mps.empty_cache()
            assert torch.equal(samples['before']['initial_noise'], samples['after']['initial_noise'])
            summary['models'][kind] = metrics
            save_checkpoint(samples, out/f'{kind}_paired_samples.pt')
            count = min(4, args.samples_per_class)
            fig, axes = plt.subplots(10, count*2, figsize=(10, 11), squeeze=False)
            for digit in range(10):
                for version_i, version in enumerate(('before', 'after')):
                    for col in range(count):
                        ax = axes[digit, version_i*count+col]
                        image = samples[version]['images'][digit*args.samples_per_class+col, 0]
                        ax.imshow(image, cmap='gray', vmin=-1, vmax=1)
                        ax.set_xticks([]); ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        if digit == 0:
                            ax.set_title(f'{version.title()} {col+1}', fontsize=10)
                        if version_i == 0 and col == 0:
                            ax.set_ylabel(str(digit), rotation=0, labelpad=12)
            fig.suptitle(f'{kind.upper()}: same noise, labels and sampler before/after training', fontsize=13)
            fig.text(.5, .012, 'First four fixed samples per class; no selection. Full paired samples and per-class metrics are saved.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .03, 1, .97))
            fig.savefig(out/f'{kind}_before_after.png', dpi=160)
            plt.close(fig)
        save_json(summary, out/'comparison.json')
        fig, axes = plt.subplots(4, 10, figsize=(12, 5.4))
        for k, kind in enumerate(('flow', 'diffusion')):
            paired = torch.load(out/f'{kind}_paired_samples.pt', map_location='cpu', weights_only=True)
            for v, version in enumerate(('before', 'after')):
                for digit in range(10):
                    ax = axes[2*k+v, digit]
                    ax.imshow(paired[version]['images'][digit*args.samples_per_class, 0],
                              cmap='gray', vmin=-1, vmax=1)
                    ax.set_xticks([]); ax.set_yticks([])
                    if k == v == 0:
                        ax.set_title(str(digit))
                    if digit == 0:
                        ax.set_ylabel(f'{kind.upper()}\n{version}', fontsize=10)
        fig.suptitle('Before / after continuation: first fixed sample for each digit', fontsize=13)
        fig.text(.5, .015, 'Same labels, initial noise and sampling settings. Examples are not selected by quality.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(out/'comparison_preview.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, kind in zip(axes, ('flow', 'diffusion')):
            info = json.loads((self.run/f'{kind}_training.json').read_text())
            x = [info['start_step']]+[r['step'] for r in info['records']]
            y = [info['baseline_validation']]+[r['validation_loss'] for r in info['records']]
            ax.plot(x, y, 'o-', color='#0072B2')
            ax.scatter(info['best_step'], info['best_validation'], marker='*', s=140, color='#D55E00', zorder=5)
            ax.set(title=f'{kind.upper()} / held-out validation', xlabel='Total optimizer updates', ylabel='MSE (model-specific target)')
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out/'validation_progress.png', dpi=160)
        plt.close(fig)

    def promote(self):
        promoted = {}
        for kind in ('flow', 'diffusion'):
            best = self.run/f'{kind}_best.pt'
            ck = torch.load(best, map_location='cpu', weights_only=True)
            info = ck['fine_tuning']
            if not info['best_validation'] < info['baseline_validation']:
                raise RuntimeError(f'{kind} did not improve; keep baseline')
            target = ROOT/f'checkpoints/conditional/conditional_{kind}.pt'
            if digest(target) not in (info['baseline_sha256'], digest(best)):
                raise RuntimeError(f'{target} changed outside this run; preserve it and review')
            tmp = target.with_suffix('.pt.tmp')
            shutil.copy2(best, tmp)
            tmp.replace(target)
            promoted[kind] = {'step': ck['completed_steps'], 'sha256': digest(target),
                              'baseline_validation': info['baseline_validation'], 'validation': info['best_validation']}
        save_json(promoted, self.run/'promoted.json')
        print('Promoted validated EMA checkpoints:', json.dumps(promoted), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate', 'promote'])
    parser.add_argument('--run-dir', type=Path, default=ROOT/'outputs/training_improvement/20261002')
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--tag', default='final')
    args = parser.parse_args()
    if args.updates < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('Update counts must be positive')
    if not 0 < args.min_lr <= args.lr or args.samples_per_class < 2:
        parser.error('Require 0 < min-lr <= lr and at least two samples per class')
    prepare_run(args.run_dir)
    experiment = Experiment(args.run_dir)
    if args.mode == 'train':
        for kind in ('flow', 'diffusion'):
            experiment.train(kind, args)
    elif args.mode == 'evaluate':
        experiment.evaluate(args)
    else:
        experiment.promote()


if __name__ == '__main__':
    main()
