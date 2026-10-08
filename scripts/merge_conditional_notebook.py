"""Compatibility entry: rebuild display notebooks from static Python APIs."""
from build_display_notebooks import build

def merge_notebooks(root=None):
    build()

if __name__ == "__main__":
    build()
