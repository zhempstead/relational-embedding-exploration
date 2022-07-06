from pathlib import Path

import hydra
from hydra.utils import get_original_cwd

from relational_embeddings.normalize.leva import leva_normalize

@hydra.main(version_base=None, config_path='../../hydra_conf', config_name='run')
def normalize(cfg):
    '''
    Normalize input tables
    '''
    datadir = (Path(get_original_cwd()) / __file__).parent.parent.parent / 'data' / cfg.dataset.name
    # CWD set by hydra
    outdir = Path.cwd() / 'normalize'
    outdir.mkdir()

    print(f"Normalizing using '{cfg.normalization.method}' method...")

    if cfg.normalization.method == 'leva':
      leva_normalize(datadir, outdir, cfg.normalization)
    else:
      raise ValueError("Unrecognized normalize method '{cfg.normalize.method}'")
    
    print(f"Done normalizing! Output at '{outdir}'")

if __name__ == '__main__':
    normalize()
