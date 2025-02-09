import importlib
import os
from pathlib import Path

import hydra


@hydra.main(version_base=None, config_path="../hydra_conf", config_name="run")
def run(cfg):
    """
    Build embedding and evaluate on downstream model
    """
    if cfg.resume_workdir is not None:
        resume_subdir = Path(cfg.resume_workdir) / cfg.pipeline_subdir
        resume_subdir.mkdir(exist_ok=True)
        os.chdir(resume_subdir)
        print(f"Subdir changed to '{Path.cwd()}'")

    donefile = Path.cwd() / 'DONE'
    if donefile.exists():
        print(f"{donefile} already exists... skipping this job")
        return

    if cfg.pipeline_stage != cfg.pipeline[0]:
        parent_donefile = Path.cwd() / '..' / 'DONE'
        if not parent_donefile.exists():
            print(f"Parent's {parent_donefile} doesn't exist... aborting this job!")
            raise RuntimeError("Parent's donefile doesn't exist")

    stage_func = getattr(importlib.import_module(f"relational_embeddings.pipeline.{cfg.pipeline_stage}"), cfg.pipeline_stage)
    stage_func(cfg, Path.cwd())
    donefile.touch()


if __name__ == "__main__":
    run()
