"""Headless PNG output and terminal tables; no interactive runtime needed."""
import re,os
from pathlib import Path
_CACHE = Path(__file__).resolve().parents[1]/'.cache/matplotlib'
_CACHE.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(_CACHE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .runtime import ROOT

_FOLDER = ROOT/'outputs/demo'
_COUNTER = 0

def configure_output(folder):
    global _FOLDER, _COUNTER
    _FOLDER = Path(folder)
    _FOLDER.mkdir(parents=True, exist_ok=True)
    _COUNTER = 0
    return _FOLDER


def finish_figure(fig=None):
    global _COUNTER
    fig = plt.gcf() if fig is None else fig
    _FOLDER.mkdir(parents=True, exist_ok=True)
    _COUNTER += 1
    title = fig._suptitle.get_text() if fig._suptitle else 'figure'
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_')[:80] or 'figure'
    path = _FOLDER/f'{_COUNTER:02d}_{slug}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved figure:', path)
    return path


def print_table(value):
    print(value.to_string() if hasattr(value, 'to_string') else str(value))
