from collections import defaultdict
import datetime as dt
import json
from relational_embeddings.lib.utils import all_csv_in_path, dataset_dir, make_symlink

def real_dataset(outdir, cfg):
    """
    Symlink over real dataset
    """
    indir = dataset_dir(cfg.name)
    
    for path in all_csv_in_path(indir, exclude_base=False):
        make_symlink(path, outdir / path.name)
