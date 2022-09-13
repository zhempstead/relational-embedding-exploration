from pathlib import Path
import shutil
import sys

import hydra
from omegaconf import OmegaConf

from relational_embeddings.lib.utils import get_rootdir, get_sweep_vars
from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph
from relational_embeddings.pipeline.graph2model import graph2model
from relational_embeddings.pipeline.graph2text import graph2text
from relational_embeddings.pipeline.text2model import text2model
from relational_embeddings.pipeline.model2emb import model2emb
from relational_embeddings.pipeline.downstream import downstream

STAGE2FUNC = {
    "normalize": normalize,
    "table2graph": table2graph,
    "graph2model": graph2model,
    "graph2text": graph2text,
    "text2model": text2model,
    "model2emb": model2emb,
    "downstream": downstream,
}

def run(outdir):
    stage = outdir.name.split(',')[0]
    overrides = get_overrides(outdir)

    with hydra.initialize(version_base=None, config_path="../hydra_conf"):
        cfg = hydra.compose(config_name="run", overrides=overrides, return_hydra_config=True)
    stage_idx = cfg.hydra.sweeper.pipeline.index(stage)
    if stage_idx == 0:
        indir = None
    else:
        indir = outdir.parent
    run_stage(stage, cfg, outdir, indir=indir)

def get_overrides(outdir):
    # Sweep overrides
    sweep_vars = get_sweep_vars(outdir)
    overrides = [f'{var}={val}' for var, val in sweep_vars.items()]

    # Non-sweep overrides
    rootdir = get_rootdir(outdir)
    multirun_cfg = OmegaConf.load(rootdir / 'multirun.yaml')
    overrides += [ov for ov in multirun_cfg.hydra.overrides.task if len(ov.split('=')[0].split('.')) == 1]

    return overrides

def run_stage(stage, cfg, outdir, indir=None):
    outdir.mkdir(exist_ok=True)
    stage_func = STAGE2FUNC[stage]
    stage_func(cfg, outdir, indir=indir)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
