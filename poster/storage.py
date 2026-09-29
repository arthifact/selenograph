"""All optional poster outputs stay inside this removable package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def output_path(path):
    root = ROOT.resolve()
    result = Path(path).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("Choose an output subdirectory inside poster/")
    return result
