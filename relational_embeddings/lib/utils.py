import os
from pathlib import Path


def all_csv_in_path(path, exclude_base=False, exclude_er_map=False):
    files = [f for f in path.iterdir() if f.is_file() and f.name.endswith(".csv")]
    if exclude_base:
        files = [f for f in files if not f.name.startswith("base")]
    if exclude_er_map:
        files = [f for f in files if not f.name == 'er_mapping.csv']
    return files


def dataset_dir(dataset_name):
    return Path(__file__).resolve().parent.parent.parent / "data" / dataset_name


def make_symlink(source, link):
    link.symlink_to(os.path.relpath(source, start=link.parent))


def prev_stage_dir(cwd, prev_stage):
    while not cwd.name.startswith(prev_stage):
        if cwd.parent == cwd:
            raise ValueError(f"Unrecognized previous stage '{prev_stage}'")
        cwd = cwd.parent
    return cwd

def get_rootdir(cwd):
    while not (cwd / "multirun.yaml").exists():
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
            if var.startswith("global."):
                key = var
            else:
                key = f'{stage}.{var}'
            sweep_vars[key] = value
    return sweep_vars


def tee(fout, text):
    print(text)
    fout.write(text)
    fout.write("\n")
