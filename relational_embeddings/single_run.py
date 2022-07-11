from pathlib import Path

import hydra

from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph

STAGE2FUNC = {
    'normalize': normalize,
    'table2graph': table2graph,
}

@hydra.main(version_base=None, config_path='../hydra_conf', config_name='single_run')
def run(cfg):
    '''
    Build embedding and evaluate on downstream model
    '''
    outdir = Path.cwd()
    indir = None
    for stage in cfg.pipeline:
      outdir = Path.cwd() / stage
      outdir.mkdir()
      stage_func = STAGE2FUNC[stage]
      stage_func(cfg, outdir, indir=indir)
      indir = outdir

if __name__ == '__main__':
    run()
