"""Restore full-project MNIST from the bundled files without network access.

Files are checked before copying; existing mismatching files are never overwritten.
Streaming extraction uses O(chunk size) memory and O(total data bytes) time.
"""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def install_file(source, destination, compressed=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        candidate = Path(scratch) / destination.name
        opener = gzip.open if compressed else open
        with opener(source, 'rb') as stream, candidate.open('wb') as target:
            shutil.copyfileobj(stream, target, length=1024 * 1024)
        if destination.exists():
            if digest(destination) != digest(candidate):
                raise RuntimeError(f'Existing file differs; not overwritten: {destination}')
        else:
            candidate.replace(destination)


def main():
    package = ROOT / 'Flow_Group_Package'
    manifest = json.loads((package / 'package_manifest.json').read_text())
    # Check all assets before restoring any data.
    for name, record in manifest['files'].items():
        if not name.startswith('data/'):
            continue
        source = package / name
        if source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise RuntimeError(f'Bundled asset failed verification: {name}')
    for source in sorted((package / 'data/MNIST/raw').glob('*.gz')):
        install_file(source, ROOT / 'data/MNIST/raw' / source.name)
        install_file(source, ROOT / 'data/MNIST/raw' / source.stem, compressed=True)
    install_file(package / 'data/mnist_split_seed42.npz', ROOT / 'data/mnist_split_seed42.npz')
    print('Verified bundled assets; root MNIST and the fixed split are ready (offline).')


if __name__ == '__main__':
    main()
