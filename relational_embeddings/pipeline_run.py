from pathlib import Path

import hydra

from relational_embeddings.stage_run import run_stage


@hydra.main(version_base=None, config_path="../hydra_conf", config_name="single_run")
def run(cfg):
    """
    Build embedding and evaluate on downstream model
    """
    outdir = Path.cwd()
    indir = None
    for stage in cfg.pipeline:
        outdir = Path.cwd() / stage
        run_stage(stage, cfg, outdir, indir=indir)
        indir = outdir


if __name__ == "__main__":
    run()
