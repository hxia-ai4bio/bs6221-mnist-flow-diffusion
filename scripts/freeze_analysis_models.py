"""Export a versioned, self-contained inference snapshot without overwriting it."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT/'checkpoints/frozen/20261002_flow15000_ddpm20000'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = SNAPSHOT/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest['files_sha256'].items():
            if sha256(SNAPSHOT/name) != digest:
                raise RuntimeError(f'Frozen snapshot changed: {name}')
        print(f'Existing snapshot verified: {SNAPSHOT}')
        return
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    if any(SNAPSHOT.iterdir()):
        raise RuntimeError('Incomplete snapshot exists. Inspect it before retrying.')
    source = ROOT/'bs6221_flow/models.py'
    code = source.read_text()
    (SNAPSHOT/'model_definitions.py').write_text(code)
    spec = importlib.util.spec_from_file_location('frozen_definitions', SNAPSHOT/'model_definitions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = {}
    for kind, expected_steps in [('flow', 15000), ('diffusion', 20000)]:
        path = ROOT/f'checkpoints/conditional/conditional_{kind}.pt'
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['completed_steps'] == expected_steps
        model = module.ConditionalTinyUNet(ck['config']['channels'])
        model.load_state_dict(ck['ema'], strict=True)
        assert all(torch.isfinite(v).all() for v in ck['ema'].values())
        payload = {'kind': kind, 'state_dict': ck['ema'], 'config': ck['config'],
                   'completed_steps': ck['completed_steps'], 'weight_type': 'EMA',
                   'source_checkpoint_sha256': sha256(path)}
        torch.save(payload, SNAPSHOT/f'{kind}_ema.pt')
        loaded = torch.load(SNAPSHOT/f'{kind}_ema.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(v, loaded['state_dict'][k]) for k, v in ck['ema'].items())
        records[kind] = {'training_steps': expected_steps, 'parameters': sum(p.numel() for p in model.parameters()),
                         'source': str(path.relative_to(ROOT)), 'source_sha256': sha256(path),
                         'expanded_validation_mse': ck['fine_tuning']['best_validation']}
    path = ROOT/'checkpoints/conditional/digit_classifier.pt'
    ck = torch.load(path, map_location='cpu', weights_only=True)
    torch.save({'state_dict': ck['model'], 'config': ck['config'], 'real_validation_accuracy': .9818},
               SNAPSHOT/'digit_classifier.pt')
    manifest = {'snapshot': SNAPSHOT.name, 'weight_type': 'EMA', 'normalization': 'pixel / 127.5 - 1',
                'image_shape': [1, 28, 28], 'models': records, 'test_data_used': False,
                'architecture_source': str(source.relative_to(ROOT)), 'architecture_source_sha256': sha256(source),
                'files_sha256': {p.name: sha256(p) for p in SNAPSHOT.iterdir() if p.is_file()}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Frozen EMA weights and architecture saved: {SNAPSHOT}')


if __name__ == '__main__':
    main()
