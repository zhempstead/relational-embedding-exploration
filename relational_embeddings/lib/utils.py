import os
from pathlib import Path

def all_csv_in_path(path, exclude_base=False):
  files = [f for f in path.iterdir() if f.is_file() and f.name.endswith('.csv')]
  if exclude_base:
    files = [f for f in files if not f.name.startswith('base')]
  return files

def dataset_dir(dataset_name):
  return Path(__file__).resolve().parent.parent.parent / 'data' / dataset_name

def make_symlink(source, link):
  link.symlink_to(os.path.relpath(source, start=link.parent))
