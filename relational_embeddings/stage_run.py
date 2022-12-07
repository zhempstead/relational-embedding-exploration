import importlib
from pathlib import Path
import shutil
import sys

import hydra
from omegaconf import OmegaConf

from relational_embeddings.lib.utils import get_rootdir, get_sweep_vars

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
    overrides += [ov for ov in multirun_cfg.hydra.overrides.task if ',' not in ov]

    return overrides

def run_stage(stage, cfg, outdir, indir=None):
    outdir.mkdir(exist_ok=True)
    stage_func = getattr(importlib.import_module(f"relational_embeddings.pipeline.{stage}"), stage)
    stage_func(cfg, outdir, indir=indir)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
