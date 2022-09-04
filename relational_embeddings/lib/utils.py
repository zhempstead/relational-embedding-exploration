import os
from pathlib import Path


def all_csv_in_path(path, exclude_base=False):
    files = [f for f in path.iterdir() if f.is_file() and f.name.endswith(".csv")]
    if exclude_base:
        files = [f for f in files if not f.name.startswith("base")]
    return files


def dataset_dir(dataset_name):
    return Path(__file__).resolve().parent.parent.parent / "data" / dataset_name


def make_symlink(source, link):
    link.symlink_to(os.path.relpath(source, start=link.parent))


def prev_stage_dir(cwd, prev_stage):
    while not cwd.name.startswith(prev_stage):
        cwd = cwd.parent
    return cwd


def get_sweep_vars(outdir):
    '''
    Get relevant sweep variables and their values by parsing outdir
    '''
    sweep_vars = {}
    for subdir in outdir.parts:
        parts = subdir.split(',')
        if len(parts) == 1:
            continue
        stage = parts[0]
        for part in parts[1:]:
            var, value = part.split('=')
            try:
                value = int(value)
            except ValueError:
                pass
            sweep_vars[f'{stage}.{var}'] = value
    return sweep_vars
