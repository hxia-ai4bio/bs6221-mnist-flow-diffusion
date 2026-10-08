"""Download/check MNIST and save a reproducible, disjoint train/validation split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "outputs" / "setup"
    output_root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST(data_root, train=True, download=args.download, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=args.download, transform=transform)
    if len(train) != 60_000 or len(test) != 10_000:
        raise RuntimeError("Unexpected official MNIST dataset sizes")

    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(42)).numpy()
    train_indices, val_indices = indices[:55_000], indices[55_000:]
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("Training and validation indices overlap")
    np.savez(data_root / "mnist_split_seed42.npz", train=train_indices, validation=val_indices)

    images = torch.stack([train[int(i)][0] for i in train_indices[:64]])
    if images.shape != (64, 1, 28, 28) or not torch.isfinite(images).all():
        raise RuntimeError("Invalid image batch")
    if images.min() < -1 or images.max() > 1:
        raise RuntimeError("Normalization must produce values in [-1, 1]")
    save_image((images + 1) / 2, output_root / "mnist_preview.png", nrow=8)

    raw_files = {}
    for path in sorted((data_root / "MNIST" / "raw").iterdir()):
        if path.is_file():
            raw_files[str(path.relative_to(PROJECT_ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {
        "dataset": "MNIST", "loader": "torchvision.datasets.MNIST",
        "seed": 42, "train_count": 55_000, "validation_count": 5_000,
        "test_count": 10_000, "test_source": "official test set; not used for tuning",
        "image_shape": [1, 28, 28], "normalization": "float32 in [-1, 1]",
        "split_file": "data/mnist_split_seed42.npz",
        "raw_files": raw_files,
    }
    (data_root / "mnist_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "raw_files"}, indent=2))


if __name__ == "__main__":
    main()
