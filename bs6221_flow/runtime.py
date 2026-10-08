"""Paths and explicit device selection shared by the Python CLI."""
from pathlib import Path
import hashlib
import torch
ROOT = Path(__file__).resolve().parents[1]


def choose_device(preference='auto'):
    available = {'cpu': True, 'cuda': torch.cuda.is_available(),
                 'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}
    if preference == 'auto':
        return torch.device(next(x for x in ('cuda','mps','cpu') if available[x]))
    if preference not in available or not available[preference]:
        raise RuntimeError(f'Device {preference} unavailable; select auto or cpu.')
    return torch.device(preference)


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''): result.update(chunk)
    return result.hexdigest()
