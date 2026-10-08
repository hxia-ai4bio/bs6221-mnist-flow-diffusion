"""Show cached Euler outputs across fixed noises; no training or integration."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets


def main():
    out = ROOT/'outputs/numerical_analysis/flow_v1'
    bank = torch.load(out/'initial_conditions.pt', map_location='cpu', weights_only=True)
    labels = bank['labels']
    reference = torch.load(out/'cache/RK4_05120.pt', map_location='cpu', weights_only=True)
    steps = (1, 2, 3, 5, 10)
    states, rows = {}, []
    for n in steps:
        result = torch.load(out/f'cache/Euler_{n:05d}.pt', map_location='cpu', weights_only=True)
        assert result['config_id'] == reference['config_id']
        states[n] = result['images']
        for digit in range(10):
            images = result['images'][labels == digit].double()
            ref = reference['images'][labels == digit].double()
            # RMS pixel standard deviation over the fixed noises of one class.
            spread = images.var(0, unbiased=False).mean().sqrt().item()
            ref_spread = ref.var(0, unbiased=False).mean().sqrt().item()
            rows.append({'digit': digit, 'steps': n, 'samples': len(images),
                         'rms_pixel_std': spread, 'reference_rms_pixel_std': ref_spread,
                         'spread_ratio_to_reference': spread/ref_spread})
    table = pd.DataFrame(rows)
    table.to_csv(out/'euler_within_class_spread.csv', index=False)
    digit = 3
    indices = torch.where(labels == digit)[0][:6]
    fig, axes = plt.subplots(len(steps)+1, len(indices), figsize=(8, 7.5))
    for row, n in enumerate((*steps, None)):
        images = reference['images'] if n is None else states[n]
        for col, ix in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[ix, 0], cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'Noise {col+1}', fontsize=10)
            if col == 0:
                ax.set_ylabel('Reference' if n is None else f'Euler N={n}', fontsize=10)
    fig.suptitle('Label 3: recognizable shape versus variation across noises', fontsize=12)
    fig.text(.5, .012, 'Same noise down each column; first six fixed noises, no selection.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .97))
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'13_euler_noise_diversity.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(table[table.digit == digit].to_string(index=False))
    # Independent check against the training-set class mean; official test set unused.
    data = datasets.MNIST(ROOT/'data', train=True, download=False)
    train_ix = np.load(ROOT/'data/mnist_split_seed42.npz')['train']
    chosen = torch.as_tensor(train_ix, dtype=torch.long)
    chosen = chosen[data.targets[chosen] == digit]
    mean_image = data.data[chosen].double().mean(0)/127.5-1
    error = (states[1][labels == digit, 0].double()-mean_image).square().flatten(1).mean(1).sqrt()
    print('Euler 1-step RMSE to training class mean: median', float(error.median()))


if __name__ == '__main__':
    main()
