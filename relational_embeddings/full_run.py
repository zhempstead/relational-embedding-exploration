from pathlib import Path

import hydra
from hydra.utils import get_original_cwd

from relational_embeddings.normalize import normalize
from relational_embeddings.table2graph import table2graph

@hydra.main(version_base=None, config_path='../hydra_conf', config_name='run')
def run(cfg):
    '''
    Build embedding and evaluate on downstream model
    '''
    normalize(cfg)
    table2graph(cfg)


if __name__ == '__main__':
    run()
