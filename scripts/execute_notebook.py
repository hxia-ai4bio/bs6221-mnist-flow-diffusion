"""Execute a display notebook with training disabled and save real outputs."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def execute(path, kernel, output=None):
    root = Path(__file__).resolve().parents[1]
    path = Path(path)
    target = Path(output) if output else path
    target.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(path, as_version=4)
    os.environ['BS6221_RUN_TRAINING'] = '0'
    os.environ['MPLCONFIGDIR'] = str(root/'.cache/matplotlib')
    os.environ['IPYTHONDIR'] = str(root/'.cache/ipython')
    def saved(cell, cell_index, **kwargs):
        print(f'Cell {cell_index + 1}/{len(notebook.cells)} complete', flush=True)
    client = NotebookClient(notebook, timeout=1200, kernel_name=kernel,
                            resources={'metadata': {'path': str(root)}},
                            on_cell_executed=saved)
    client.execute()
    nbformat.write(notebook, target)
    print(f'Executed notebook saved: {target}', flush=True)


def main(default='notebooks/00_project_demo.ipynb'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default=default)
    parser.add_argument('--kernel', default='bs6221-flow-shared')
    parser.add_argument('--output')
    args = parser.parse_args()
    execute(args.notebook, args.kernel, args.output)


if __name__ == '__main__':
    main()
